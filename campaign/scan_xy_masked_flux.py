#!/usr/bin/env python3
"""Winding-free flux over the (Jx, Jy) plane, normalised so it is +-1.

No counterterm anywhere in here.  The mask is the whole method: build the
exact Feshbach map F(z) of the ice block, delete every entry connecting
different transport sectors, and solve E = eig F(E).  The ground
multiplet's eigenvector is the ice-manifold component of the winding-free
state, and the contractible hexagon operator maps ice to ice, so the flux
is a 90x90 expectation value.

WHAT CHANGED, AND WHY IT IS THE RIGHT OBSERVABLE.  The raw expectation
<O_hex + h.c.> is amplitude times phase, not a flux.  On the subspace where
a hexagon is flippable the ring operator acts as sigma^x, with eigenvalues
+-1, so

    <O_h> = +- <N_h>,     N_h = projector onto "hexagon h is flippable",

and the flux is the ratio, not the numerator:

    Phi = <sum_h O_h> / <sum_h N_h>   ->   exactly +-1.

Phi = +1 is 0-flux, Phi = -1 is pi-flux.  The earlier +-1/4 was that +-1
diluted by the mean flippability <sum_h N_h>/N_hex, which this run measures
directly: the prediction implied by +-0.25 over sixteen hexagons is that
exactly four hexagons are flippable on average.

The ratio also resolves the region where <O_hex> vanished, which was
ambiguous between two very different states:
  * <N_h> nonzero: hexagons are flippable but do not resonate -- genuine
    destructive interference, Phi = 0;
  * <N_h> zero: no flippable hexagon at all, Phi = 0/0 undefined -- a
    ground state that is not a resonating gauge state in any sense.
Per-hexagon Phi_h is recorded too, so a non-uniform flux cannot hide inside
the average.

SYMMETRY.  Parity (-1)^(N_up) commutes with H at every (Jpm, Jpmpm), and
every ice state has N_up = 8 exactly (each site sits in one A-tetrahedron;
summing the ice rule over the four of them gives 4x2 = 8).  So P lies
entirely in the even sector and the self-energy cannot reach the odd one at
any order: the 32768 odd states are not a block to diagonalise separately,
they are irrelevant.  Dropping them takes QHQ from 65446 to 32678 and the
Klein blocks from ~16362 to ~8170 -- eight times fewer flops.  Pass
--full-basis to reproduce the old, slower path for validation.

Usage: scan_xy_masked_flux.py [--gpu] [--full-basis]
                              <points.json> <task_id> <n_tasks> <out_prefix>
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "notes"))
sys.path.insert(0, str(ROOT / "campaign"))

import recompute_finite_size_artifact as geometry  # noqa: E402
from qsi_campaign.exact_band import (  # noqa: E402
    cubic16_translation_permutations,
    full_spin_basis,
    microscopic_character_hamiltonian,
    translation_sector_bases,
)
from qsi_campaign.downfold import (  # noqa: E402
    build_blocks_klein, solve_roots)

OUTPUT = ROOT / "campaign" / "outputs" / "xy_phase_plane"
OUTPUT.mkdir(parents=True, exist_ok=True)

DEGEN_TOL = 1e-9
FLIP_TOL = 1e-9          # below this, "no flippable hexagon in this state"
# The continuum edge must be the lowest pole that actually couples to the
# ice manifold.  Decoupled poles have zero residue, so the self-energy is
# analytic across them and a band level cannot decay into them; using one
# as the edge silently discards every root above it.  On the full basis the
# lowest pole is an odd-parity state with exactly zero coupling.
COUPLING_FLOOR = 1e-8


def even_parity_basis(n_sites: int) -> np.ndarray:
    """States with even N_up.  Closed under both transverse moves."""
    states = full_spin_basis(n_sites)
    return states[np.bitwise_count(states) % 2 == 0]


def hexagon_operators(cluster):
    """Ring-flip and flippability operators per contractible hexagon.

    Both live in the 90-dimensional ice basis: a ring flip on a flippable
    hexagon takes an ice state to an ice state.  ``flip`` is the off-diagonal
    sigma^x on each flippable pair; ``count`` is the diagonal projector onto
    the flippable configurations, whose expectation normalises the flux.
    """
    ice = np.asarray(cluster.ice_states, dtype=np.uint64)
    hexagons = {tuple(sorted(sites)): sites
                for sites, wind in cluster.hexes if tuple(wind) == (0, 0, 0)}
    flips, counts = [], []
    for sites in hexagons.values():
        mask = 0
        pattern = 0
        for position, site in enumerate(sites):
            mask |= 1 << int(site)
            if position % 2 == 0:
                pattern |= 1 << int(site)
        other = mask ^ pattern
        occupied = ice & np.uint64(mask)
        active = np.where((occupied == np.uint64(pattern))
                          | (occupied == np.uint64(other)))[0]
        flip = np.zeros((ice.size, ice.size))
        count = np.zeros(ice.size)
        if active.size:
            partners = np.searchsorted(ice, ice[active] ^ np.uint64(mask))
            if not np.array_equal(ice[partners],
                                  ice[active] ^ np.uint64(mask)):
                raise RuntimeError("a hexagon flip left the ice manifold")
            flip[partners, active] = 1.0
            count[active] = 1.0
        flips.append(flip)
        counts.append(np.diag(count))
    return flips, counts


def multiplet_mean(operator, vectors):
    """Mean of <O> over the columns of ``vectors`` (an orthonormal set)."""
    return float(np.mean([v @ operator @ v for v in vectors.T]))


def branch_state(blocks, roots, flips, counts, *, masked: bool):
    """Ground multiplet of the map, with the normalised flux."""
    if not roots.converged.any():
        return None
    energies = np.where(roots.converged, roots.energies, np.inf)
    ground = float(np.min(energies))
    multiplet = np.where(energies <= ground + DEGEN_TOL)[0]
    above = energies[energies > ground + DEGEN_TOL]
    values, eigenvectors = np.linalg.eigh(blocks.f_matrix(ground,
                                                          masked=masked))
    vectors = eigenvectors[:, multiplet]

    per_hexagon_flip = [multiplet_mean(f, vectors) for f in flips]
    per_hexagon_count = [multiplet_mean(c, vectors) for c in counts]
    total_flip = float(np.sum(per_hexagon_flip))
    total_count = float(np.sum(per_hexagon_count))
    # Phi = <sum O_h> / <sum N_h>: +1 is 0-flux, -1 is pi-flux, 0 is
    # flippable-but-not-resonating, and NaN means nothing was flippable.
    flux = total_flip / total_count if total_count > FLIP_TOL else float("nan")
    per_hexagon = [f / c if c > FLIP_TOL else float("nan")
                   for f, c in zip(per_hexagon_flip, per_hexagon_count)]
    finite = [value for value in per_hexagon if np.isfinite(value)]

    weights = [1.0 / (1.0 - blocks.self_energy_slope(ground, v))
               for v in vectors.T]
    return {
        "energy": ground,
        "flux": flux,
        "flux_raw": total_flip / len(flips),   # the old, un-normalised number
        "flippable": total_count,              # <sum_h N_h>, mean over hexes
        "flux_per_hexagon_min": min(finite) if finite else float("nan"),
        "flux_per_hexagon_max": max(finite) if finite else float("nan"),
        "n_hexagons_flippable": int(np.sum(np.asarray(per_hexagon_count)
                                           > FLIP_TOL)),
        "ice_weight": float(np.mean(weights)),
        "degeneracy": int(multiplet.size),
        "gap": float(above.min() - ground) if above.size else float("nan"),
        "residual": float(values[multiplet[0]] - ground),
        "levels": np.sort(energies[np.isfinite(energies)])[:10].tolist(),
        "n_converged": int(roots.converged.sum()),
        "vectors": vectors.T.tolist(),         # keep them: cheap, and the
                                               # next observable is free
    }


def main() -> None:
    flags = {value for value in sys.argv[1:] if value.startswith("--")}
    arguments = [value for value in sys.argv[1:] if not value.startswith("--")]
    gpu = "--gpu" in flags
    full_basis = "--full-basis" in flags
    points_file, task_id, n_tasks, prefix = (
        arguments[0], int(arguments[1]), int(arguments[2]), arguments[3])
    requested = json.loads(Path(points_file).read_text())["points"]
    mine = sorted(requested[task_id::n_tasks],
                  key=lambda point: (point["jx"], point["jy"]))
    print(f"task {task_id}/{n_tasks}: {len(mine)} of {len(requested)} points",
          flush=True)
    if not mine:
        return

    cluster = geometry.build_cluster("cubic", (1, 1, 1))
    states = (full_spin_basis(cluster.n_sites) if full_basis
              else even_parity_basis(cluster.n_sites))
    ice_sorted = np.sort(np.asarray(cluster.ice_states, dtype=np.uint64))
    ice_rows = np.searchsorted(states, ice_sorted)
    if not np.array_equal(states[ice_rows], ice_sorted):
        raise RuntimeError("ice states are missing from the working basis")
    sector_bases = translation_sector_bases(
        states, cubic16_translation_permutations(cluster))
    flips, counts = hexagon_operators(cluster)
    print(f"basis {len(states)} ({'full' if full_basis else 'even parity'}), "
          f"{len(sector_bases)} Klein irreps, {len(flips)} hexagons",
          flush=True)

    destination = OUTPUT / f"{prefix}_{task_id:03d}.json"
    records = []
    if destination.exists():
        records = json.loads(destination.read_text())["points"]
        done = {(round(r["jx"], 9), round(r["jy"], 9)) for r in records}
        mine = [p for p in mine
                if (round(p["jx"], 9), round(p["jy"], 9)) not in done]
        print(f"resuming: {len(records)} on disk, {len(mine)} to go",
              flush=True)

    for point in mine:
        started = time.perf_counter()
        jx, jy = float(point["jx"]), float(point["jy"])
        jpm, jpmpm = -(jx + jy) / 4.0, (jx - jy) / 4.0
        hamiltonian = microscopic_character_hamiltonian(
            cluster, states, jpm, np.zeros(3), jpmpm=jpmpm)
        entry = {"jx": jx, "jy": jy, "jpm": jpm, "jpmpm": jpmpm}
        try:
            blocks = build_blocks_klein(cluster, states, ice_rows,
                                        hamiltonian, sector_bases, gpu=gpu)
        except RuntimeError as failure:
            entry["error"] = str(failure)
            entry["runtime"] = time.perf_counter() - started
            records.append(entry)
            print(f"jx={jx:+.3f} jy={jy:+.3f} UNRESOLVED: {failure}",
                  flush=True)
            with open(destination, "w") as handle:
                json.dump({"points": records}, handle, indent=1, default=float)
            continue

        for masked in (True, False):
            tag = "masked" if masked else "unmasked"
            state = branch_state(
                blocks,
                solve_roots(blocks, masked=masked,
                            coupling_floor=COUPLING_FLOOR),
                flips, counts, masked=masked)
            if state is None:
                entry[f"{tag}_flux"] = float("nan")
                entry[f"{tag}_energy"] = float("nan")
                entry[f"{tag}_degeneracy"] = 0
                entry[f"{tag}_n_converged"] = 0
                continue
            for name, value in state.items():
                entry[f"{tag}_{name}"] = value
        entry["runtime"] = time.perf_counter() - started
        records.append(entry)
        print(f"jx={jx:+.3f} jy={jy:+.3f} jpm={jpm:+.4f} pp={jpmpm:+.4f} "
              f"| masked Phi={entry.get('masked_flux', float('nan')):+.6f} "
              f"<N>={entry.get('masked_flippable', float('nan')):.4f} "
              f"deg={entry.get('masked_degeneracy', 0)} "
              f"({entry.get('masked_n_converged', 0)}/90) "
              f"| unmasked Phi={entry.get('unmasked_flux', float('nan')):+.6f} "
              f"[{entry['runtime']:.0f}s]", flush=True)
        with open(destination, "w") as handle:
            json.dump({"points": records}, handle, indent=1, default=float)

    print(f"wrote {destination} ({len(records)} points)", flush=True)


if __name__ == "__main__":
    main()
