#!/usr/bin/env python3
"""WP4: locate the sector near-degeneracy locus, then cut radially through it.

The winding-free Hamiltonian is block diagonal in transport sector, so one
sector owns the band bottom.  The interesting locus is where that ownership is
about to change hands -- where the lowest two sector ground energies become
nearly degenerate.  WP4 asks what else happens there: does the residue `Z`, the
band gap, the ring amplitude `K_6`, the loop ratio `K_8/K_6` and the continuum
edge all respond, or does only the sector gap move?

**The pre-registered kill criterion, from the 6->2 lesson.**  The cubic-16
6->2 ground-multiplicity crossing looked like physics and turned out to be a
host pathology -- FCC-32 shows no such crossing at any coupling.  So: *if only
the sector gap sharpens with system size while `Z`, `K_6`, `K_8/K_6` and the
continuum edge stay flat through the locus, it is a host effect and is
discarded without ceremony.*  That criterion is written down here, before the
scan runs, precisely so it cannot be renegotiated after seeing the answer.

Two modes:

  --scan   per-sector bordered ground energies over a `(Jpm, Jpmpm)` grid,
           giving the sector gap, the bottom sector's identity and dimension,
           the bottom degeneracy, and the continuum edge.  This is what
           LOCATES the ring; it does not fold, so it is comparatively cheap.

  --cut    the full observable set at points along a radial line: everything
           above plus `Z` and the loop tomography, which need the fold.

Usage:
  run_wp4_sector_ring.py --scan <levels> <jpm> <jpmpm> [<jpm> <jpmpm> ...]
  run_wp4_sector_ring.py --cut  <levels> <jpm> <jpmpm> [<jpm> <jpmpm> ...]
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.sparse.linalg import eigsh

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "notes"))
sys.path.insert(0, str(ROOT / "campaign"))

import recompute_finite_size_artifact as geometry  # noqa: E402
from qsi_campaign.protocol import polarization_sector_labels  # noqa: E402
from qsi_campaign.tomography import (  # noqa: E402
    connected_class_amplitudes,
    graph_distance_matrix,
    loop_tomography,
    minimum_image_distance_matrix,
)
from run_feshbach_anatomy import hexagon_flip_operator  # noqa: E402
from run_truncated_feshbach import bfs_space, build_hamiltonian  # noqa: E402
from run_wp1_tomography import masked_map  # noqa: E402

OUTPUT = ROOT / "campaign" / "outputs" / "wp4_sector_ring"
DEGENERACY_TOL = 1e-9


def sector_landscape(cluster, space, labels, ice_rows, non_ice, hamiltonian,
                     flux_operator=None, n_hexagons: int = 1):
    """Per-sector bordered ground energies, ice weights, and hexagon flux.

    The flux is the one genuinely DISCRETE label available here: for `Jpm < 0`
    the winding-free ground state carries `<O_hex> = -1/4` per hexagon
    (pi flux) and for `Jpm > 0` it carries `+1/4` (0 flux), quantized rather
    than fitted.  It is evaluated on the ice component of the bordered ground
    state and normalised by that component's weight, since the flux operator is
    defined on the ice manifold.

    This is what makes a *diagram* possible on FCC-32 at all: energies and gaps
    move continuously and cannot by themselves mark a boundary, but a quantized
    label either holds or does not.
    """
    grounds, weights, dims, fluxes = {}, {}, {}, {}
    for sector in np.unique(labels):
        rows = np.concatenate([ice_rows[labels == sector], non_ice])
        block = hamiltonian[np.ix_(rows, rows)]
        values, vectors = eigsh(block, k=2, which="SA", tol=1e-10)
        order = np.argsort(values)
        ground = vectors[:, order[0]]
        n_ice = int((labels == sector).sum())
        grounds[int(sector)] = float(values[order[0]])
        # the residue: how much of the winding-free ground state is still on
        # the ice manifold rather than dressed into spinon space
        ice_part = ground[:n_ice]
        ice_weight = float(np.sum(ice_part ** 2))
        weights[int(sector)] = ice_weight
        dims[int(sector)] = n_ice
        if flux_operator is not None and ice_weight > 1e-12:
            rows_ice = ice_rows[labels == sector]
            sub = flux_operator[np.ix_(
                np.searchsorted(ice_rows, rows_ice),
                np.searchsorted(ice_rows, rows_ice))]
            fluxes[int(sector)] = float(
                ice_part @ (sub @ ice_part) / (ice_weight * n_hexagons))
    return grounds, weights, dims, fluxes


def main() -> int:
    mode = "cut" if "--cut" in sys.argv else "scan"
    tokens = [v for v in sys.argv[1:] if not v.startswith("--")]
    levels = int(tokens[0])
    values = [float(v) for v in tokens[1:]]
    points = list(zip(values[0::2], values[1::2]))

    # cubic-16 is available so the bordered-projection flux can be checked
    # against the untruncated answer, where the masked band gives exactly +-1/4
    name = "cubic16" if "--cubic16" in sys.argv else "fcc32"
    basis, shape = (("cubic", (1, 1, 1)) if name == "cubic16"
                    else ("fcc", (2, 2, 2)))
    cluster = geometry.build_cluster(basis, shape)
    labels = polarization_sector_labels(cluster)
    ice_states = np.sort(np.asarray(cluster.ice_states, dtype=np.uint64))
    graph_distance = graph_distance_matrix(cluster)
    real_distance = minimum_image_distance_matrix(cluster)
    flux_operator = hexagon_flip_operator(cluster, ice_states)
    n_hexagons = sum(1 for _, wind in cluster.hexes if tuple(wind) == (0, 0, 0))
    print(f"{name}: {cluster.n_ice} ice states, "
          f"{n_hexagons} contractible hexagons", flush=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)

    for jpm, jpmpm in points:
        started = time.perf_counter()
        space = bfs_space(cluster, levels, jpmpm=jpmpm)
        ice_rows = np.searchsorted(space, ice_states)
        non_ice = np.setdiff1d(np.arange(len(space)), ice_rows)
        hamiltonian = build_hamiltonian(cluster, space, jpm, jpmpm).tocsr()

        grounds, weights, dims, fluxes = sector_landscape(
            cluster, space, labels, ice_rows, non_ice, hamiltonian,
            flux_operator=flux_operator, n_hexagons=n_hexagons)
        order = sorted(grounds, key=grounds.get)
        lowest, second = grounds[order[0]], grounds[order[1]]
        degenerate = [s for s in order
                      if grounds[s] - lowest < DEGENERACY_TOL]

        q_block = hamiltonian[np.ix_(non_ice, non_ice)]
        edge = float(eigsh(q_block, k=1, which="SA", tol=1e-9,
                           return_eigenvectors=False)[0])

        record = {
            "jpm": jpm, "jpmpm": jpmpm, "levels": levels,
            "space": int(len(space)),
            "bottom_sector": int(order[0]),
            "bottom_sector_dim": dims[order[0]],
            "bottom_degeneracy": len(degenerate),
            "degenerate_sectors": [int(s) for s in degenerate],
            "sector_gap": float(second - lowest),
            "ground_energy": float(lowest),
            "continuum_edge": edge,
            "edge_gap": float(edge - lowest),
            "z_ground": weights[order[0]],
            "flux_ground": fluxes.get(order[0]),
            "flux_quantized": (
                None if order[0] not in fluxes
                else bool(abs(abs(fluxes[order[0]]) - 0.25) < 1e-6)),
            "sector_grounds": {str(k): v for k, v in grounds.items()},
            "sector_fluxes": {str(k): v for k, v in fluxes.items()},
        }

        if mode == "cut":
            f_masked, _ = masked_map(cluster, space, ice_rows, non_ice,
                                     hamiltonian, labels, lowest, chunk=256)
            summary = loop_tomography(f_masked, ice_states, cluster, labels,
                                      graph_distance, real_distance)
            amplitudes = connected_class_amplitudes(summary)
            k6 = amplitudes.get(6, {}).get("K", float("nan"))
            record["k6"] = float(k6)
            for length, entry in amplitudes.items():
                if length != 6 and k6:
                    record[f"k{length}_over_k6"] = float(entry["K"] / k6)

        record["runtime"] = time.perf_counter() - started
        record["cluster"] = name
        tag = f"{name}_L{levels}_jpm{jpm:+.4f}_pp{jpmpm:+.4f}".replace(".", "p")
        (OUTPUT / f"{mode}_{tag}.json").write_text(
            json.dumps(record, indent=1, default=float))
        print(f"jpm={jpm:+.3f} pp={jpmpm:+.3f} bottom sector "
              f"{record['bottom_sector']} (dim {record['bottom_sector_dim']}, "
              f"deg {record['bottom_degeneracy']}) | sector gap "
              f"{record['sector_gap']:.3e} | edge gap {record['edge_gap']:.4f} "
              f"| Z={record['z_ground']:.4f} "
              f"| flux={record['flux_ground'] if record['flux_ground'] is None else round(record['flux_ground'], 6)}"
              f"{' [QUANTIZED]' if record['flux_quantized'] else ''}"
              + (f" | K6={record['k6']:.4e} "
                 f"K8/K6={record.get('k8_over_k6', float('nan')):.4f}"
                 if mode == "cut" else "")
              + f" ({record['runtime']:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
