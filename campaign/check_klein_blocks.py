#!/usr/bin/env python3
"""Validate the Klein-blocked downfold assembly against eigensolver-free truth.

On the fixed-Sz basis (also closed under the four FCC translations) the
blocked ``build_blocks_klein`` must reproduce, to near machine precision:

1. the exact folded map ``F(z) = PHP - QHP^T (QHQ - z)^{-1} QHP`` evaluated
   directly from the sparse microscopic Hamiltonian by conjugate-gradient
   solves -- no eigendecomposition anywhere in the reference; and
2. the tracked masked-band oracle levels (origin == "below") at Jpm = -0.15.

This validates the entire pole/coupling assembly (SVD ice orthonormalization,
projector deflation, overlap classification, per-irrep concatenation) before
the full-basis production runs depend on it.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.sparse import identity
from scipy.sparse.linalg import cg

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "notes"))

import recompute_finite_size_artifact as geometry  # noqa: E402
from qsi_campaign.exact_band import (  # noqa: E402
    cubic16_translation_permutations,
    fixed_magnetization_basis,
    microscopic_character_hamiltonian,
    translation_sector_bases,
)
from qsi_campaign.downfold import build_blocks_klein, solve_roots  # noqa: E402


def main() -> None:
    cluster = geometry.build_cluster("cubic", (1, 1, 1))
    states = fixed_magnetization_basis(cluster.n_sites, cluster.n_sites // 2)
    index = {int(s): i for i, s in enumerate(states)}
    ice_rows = np.asarray([index[int(s)] for s in cluster.ice_states])
    bases = translation_sector_bases(
        states, cubic16_translation_permutations(cluster))
    print("Klein sector dims:", {k: v.shape[1] for k, v in bases.items()},
          flush=True)

    worst = 0.0
    for jpm in (-0.125, -0.150):
        hamiltonian = microscopic_character_hamiltonian(
            cluster, states, jpm, np.zeros(3))
        started = time.perf_counter()
        blocks = build_blocks_klein(cluster, states, ice_rows,
                                    hamiltonian, bases)
        print(f"jpm={jpm}: klein blocks in {time.perf_counter()-started:.1f}s,"
              f" {len(blocks.poles)} poles", flush=True)

        real = hamiltonian.real.tocsr()
        complement = np.setdiff1d(np.arange(len(states)), ice_rows)
        qhq = real[complement][:, complement].tocsr()
        qhp = real[complement][:, ice_rows].toarray()
        php = real[ice_rows][:, ice_rows].toarray()
        edge = float(blocks.poles.min())
        for z in (edge - 0.5, edge - 2.0):
            shifted = (qhq - z * identity(qhq.shape[0], format="csr")).tocsr()
            solved = np.empty_like(qhp)
            for column in range(qhp.shape[1]):
                solved[:, column], info = cg(
                    shifted, qhp[:, column], rtol=1e-12, atol=0.0,
                    maxiter=20000)
                assert info == 0, f"CG failed at column {column}: {info}"
            reference = php - qhp.T @ solved
            deviation = float(
                np.abs(blocks.f_matrix(z, masked=False) - reference).max())
            worst = max(worst, deviation)
            print(f"  z={z:+.3f}: max|F_klein - F_cg| = {deviation:.3e}",
                  flush=True)
            assert deviation < 1e-9, "Klein F(z) disagrees with sparse CG"

        if abs(jpm + 0.150) < 1e-12:
            result = solve_roots(blocks, masked=True)
            mine = np.sort(result.energies[result.converged])
            tracked = json.load(open(
                ROOT / "campaign" / "outputs" / "masked_band_complete"
                / "band_-0p150.json"))
            levels = np.asarray(tracked["levels"], float)
            below = np.sort(levels[np.asarray(tracked["origin"]) == "below"])
            assert len(mine) == len(below), \
                f"{len(mine)} roots vs {len(below)} tracked"
            deviation = float(np.abs(mine - below).max())
            print(f"  solve_roots vs tracked oracle: {len(mine)} roots, "
                  f"max deviation {deviation:.3e}", flush=True)
            assert deviation < 1e-9, "Klein roots disagree with the oracle"

    print(f"KLEIN VALIDATION OK (worst F deviation {worst:.3e})", flush=True)


if __name__ == "__main__":
    main()
