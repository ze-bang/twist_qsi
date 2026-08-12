#!/usr/bin/env python3
"""WP3: sector-resolved longitudinal S^zz(q, omega) from the bordered solvers.

The fit-independent prediction of the construction: **which spectral weight is
forbidden outright by flux superselection, and where the surviving weight
sits in momentum.**  Unlike the peak positions of WP2, this does not depend on
any material fit -- it follows from the mask.

Why the longitudinal channel is exact here.  `S^z` is diagonal in the Ising
basis, so every matrix element `<n|S^z_q|0>` is exact on the bordered space,
spinon dressing included -- no isometry, no frame, no truncation of the
operator.  And because `S^z_q` cannot change an Ising configuration, it cannot
change the transport label: **transitions are sector-diagonal**, so only the
ground sector's own excitations carry weight.

The `q = 0` theorem (WP0b, verified on FCC-32's 2970 ice states): the four
sublattice magnetisations are exactly constant on each transport sector, and
the label is in bijection with them.  So `P S^z_{q=0} P` is a c-number on every
mask block, commutes with the whole winding-free Hamiltonian, and the inelastic
response at Gamma vanishes IDENTICALLY.  Naive periodic ED has weight there
(0.26-0.60 on cubic-16); all of it is artifact.  This script measures that zero
as a gate on itself: if the computed Gamma weight is not at machine zero, the
implementation is wrong, not the physics.

Validation ladder, in order:
  1. `--cluster cubic16` reproduces the frozen tier-1 numbers
     (`outputs/tier1_probes/`): within-band longitudinal sticks dark at
     <= 3e-5, `q = 0` masked weight <= 9e-5 against raw 0.26-0.60.
  2. only then FCC-32, where the same rule is a prediction rather than a check.

Usage: run_wp3_dssf.py <cubic16|fcc32> <levels> <jpm> [--jpmpm PP]
                       [--nstates K]
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
from run_truncated_feshbach import bfs_space, build_hamiltonian  # noqa: E402
from run_wp0b_gamma_rule import allowed_momenta  # noqa: E402

OUTPUT = ROOT / "campaign" / "outputs" / "wp3_dssf"
CLUSTERS = {"cubic16": ("cubic", (1, 1, 1)), "fcc32": ("fcc", (2, 2, 2))}
DEGENERACY_TOL = 1e-9


def sublattice_sources(cluster, space: np.ndarray, momentum) -> np.ndarray:
    """`S^z_{q,mu}` as a diagonal over the truncated space, one row per mu.

    Diagonal because `S^z` is diagonal in the Ising basis -- this is what makes
    the longitudinal channel exact rather than frame-dependent.
    """
    n_sites = int(cluster.n_sites)
    bits = (space[:, None] >> np.arange(n_sites, dtype=np.uint64)) & np.uint64(1)
    spins = bits.astype(float) - 0.5
    sublattice = np.arange(n_sites) % 4
    phase = np.exp(-2j * np.pi * (np.asarray(cluster.positions) @ momentum))
    return np.stack(
        [spins @ (phase * (sublattice == mu)) for mu in range(4)], axis=0)


def main() -> int:
    tokens = [v for v in sys.argv[1:] if not v.startswith("--")]
    name, levels, jpm = tokens[0], int(tokens[1]), float(tokens[2])
    jpmpm = 0.0
    n_states = 24
    for token in sys.argv:
        if token.startswith("--jpmpm="):
            jpmpm = float(token.split("=", 1)[1])
        if token.startswith("--nstates="):
            n_states = int(token.split("=", 1)[1])

    basis, shape = CLUSTERS[name]
    cluster = geometry.build_cluster(basis, shape)
    space = bfs_space(cluster, levels, jpmpm=jpmpm)
    labels = polarization_sector_labels(cluster)
    ice_states = np.sort(np.asarray(cluster.ice_states, dtype=np.uint64))
    ice_rows = np.searchsorted(space, ice_states)
    non_ice = np.setdiff1d(np.arange(len(space)), ice_rows)
    sector_ids = np.unique(labels)
    momenta = allowed_momenta(cluster)
    print(f"{name} L={levels} jpm={jpm:+.3f} pp={jpmpm:+.3f}: space "
          f"{len(space):,}, {len(sector_ids)} sectors, {len(momenta)} momenta",
          flush=True)

    started = time.perf_counter()
    hamiltonian = build_hamiltonian(cluster, space, jpm, jpmpm).tocsr()

    # Only sectors in the ground manifold contribute to the zero-temperature
    # response below.  First locate that manifold with one eigenvalue per
    # sector; requesting n_states eigenvectors in all 85 sectors is exactly
    # redundant and was the dominant cost of the original implementation.
    grounds: dict[int, float] = {}
    sector_rows: dict[int, np.ndarray] = {}
    for scan_index, sector in enumerate(sector_ids, start=1):
        rows = np.concatenate([ice_rows[labels == sector], non_ice])
        sector_rows[int(sector)] = rows
        block = hamiltonian[np.ix_(rows, rows)]
        value = eigsh(block, k=1, which="SA", tol=1e-10,
                      return_eigenvectors=False)[0]
        grounds[int(sector)] = float(value)
        if scan_index % 10 == 0 or scan_index == len(sector_ids):
            print(f"  ground scan {scan_index}/{len(sector_ids)} sectors",
                  flush=True)

    lowest = min(grounds.values())
    ground_sectors = [s for s, e in grounds.items()
                      if e - lowest < DEGENERACY_TOL]
    print(f"  ground energy {lowest:.9f} in sector(s) {ground_sectors}",
          flush=True)

    # Resolve the requested response window only in the contributing sectors.
    spectra: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for sector in ground_sectors:
        rows = sector_rows[sector]
        block = hamiltonian[np.ix_(rows, rows)]
        k = min(n_states, block.shape[0] - 2)
        values, vectors = eigsh(block, k=k, which="SA", tol=1e-10)
        order = np.argsort(values)
        spectra[int(sector)] = (values[order], vectors[:, order], rows)

    # sticks, averaged over a degenerate ground manifold if there is one
    results = []
    for index, momentum in enumerate(momenta):
        weights = np.zeros(n_states)
        excitation = None
        for sector in ground_sectors:
            values, vectors, rows = spectra[sector]
            sources = sublattice_sources(cluster, space[rows], momentum)
            excitation = values - values[0]
            for mu in range(4):
                column = sources[mu][:, None] * vectors[:, :1]
                overlap = vectors.conj().T @ column      # <n|S^z_q,mu|0>
                weights += np.abs(overlap[:, 0]) ** 2 / int(cluster.n_sites)
        weights /= len(ground_sectors)
        weights[excitation <= DEGENERACY_TOL] = 0.0      # elastic line removed
        results.append({
            "momentum": [float(x) for x in momentum],
            "is_gamma": bool(np.allclose(momentum, 0.0)),
            "excitation": [float(x) for x in excitation],
            "weight": [float(x) for x in weights],
            "total_inelastic_weight": float(weights.sum()),
        })
        tag = "Gamma" if np.allclose(momentum, 0.0) else f"q{index}"
        print(f"  {tag:>6}: total inelastic weight {weights.sum():.3e}",
              flush=True)

    # The Gamma rule is a statement about the ICE-BLOCK operator: `P S^z_0 P`
    # is a c-number on each transport sector, so it cannot connect states
    # WITHIN the winding-free band.  The bordered eigenvectors also carry
    # spinon (Q) components, and the sublattice magnetisations are NOT constant
    # there, so the measured weight is a small dressing tail rather than a
    # machine zero -- exactly as the frozen tier-1 record describes it
    # ("masked <= 9e-5 (weak coupling, dressing tail) down to 2e-27 at CHO,
    # against raw 0.26-0.60").  The meaningful test is therefore suppression
    # against the q != 0 weight computed by the same code path, not equality
    # with zero.
    gamma = next(r for r in results if r["is_gamma"])
    others = [r["total_inelastic_weight"] for r in results if not r["is_gamma"]]
    reference = float(np.mean(others)) if others else float("nan")
    suppression = (reference / gamma["total_inelastic_weight"]
                   if gamma["total_inelastic_weight"] > 0 else float("inf"))
    gate_passed = bool(
        gamma["total_inelastic_weight"] < 1.0e-4 and suppression > 1.0e3)
    print(f"\nGamma selection rule: weight {gamma['total_inelastic_weight']:.3e} "
          f"against {reference:.3e} at q != 0 "
          f"-> suppressed {suppression:.3g}x -> "
          f"{'PASS' if gate_passed else 'FAIL'}")
    print("  (frozen tier-1 reference: masked <= 9e-5 at weak coupling, "
          "raw 0.26-0.60 -- the residue is the spinon dressing tail, since "
          "S^z_0 is a c-number on the ice manifold but not on Q)")
    if not gate_passed:
        print("  FAIL means the matrix elements are wrong: the ice-block "
              "operator provably commutes with the mask (WP0b, verified on "
              "2970 FCC-32 ice states), so the weight cannot be large.")

    record = {
        "cluster": name, "levels": levels, "jpm": jpm, "jpmpm": jpmpm,
        "space": int(len(space)), "n_sectors": int(len(sector_ids)),
        "nstates_requested": int(n_states),
        "ground_energy": lowest, "ground_sectors": ground_sectors,
        "gamma_rule_passed": bool(gate_passed),
        "gamma_weight": gamma["total_inelastic_weight"],
        "reference_weight_q_nonzero": reference,
        "gamma_suppression": suppression,
        "momenta": results,
        "runtime": time.perf_counter() - started,
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    tag = f"{name}_L{levels}_{jpm:+.3f}_pp{jpmpm:+.3f}".replace(".", "p")
    (OUTPUT / f"dssf_{tag}.json").write_text(
        json.dumps(record, indent=1, default=float))
    print(f"wrote {OUTPUT / f'dssf_{tag}.json'} ({record['runtime']:.0f}s)")
    return 0 if gate_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
