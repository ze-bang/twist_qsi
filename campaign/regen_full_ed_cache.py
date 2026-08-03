#!/usr/bin/env python3
"""Regenerate the exact microscopic complement cache for the QMC benchmark.

Replaces the retired ``full_ed_cubic16_jpm_0p046.npz``, whose spectrum was
inconsistent with the current microscopic Hamiltonian (its ground state lay
below the true one, violating the variational bound).  The spectrum here is
produced by the verified symmetry-block path: Klein translation x S^z parity,
eight real symmetric blocks, cross-checked against sparse ``eigsh`` on the
lowest states.  Energies are in the same convention as
``microscopic_character_hamiltonian`` so the band splice is exactly consistent.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "notes"))

import recompute_finite_size_artifact as geometry  # noqa: E402
from qsi_campaign.exact_band import (  # noqa: E402
    fixed_magnetization_basis,
    full_spin_basis,
    microscopic_character_hamiltonian,
    translation_sector_bases,
    cubic16_translation_permutations,
)
from qsi_campaign.material_scan import _sector_blocks  # noqa: E402


def main() -> None:
    jpm = 0.046
    cluster = geometry.build_cluster("cubic", (1, 1, 1))
    states = full_spin_basis(cluster.n_sites)
    started = time.perf_counter()
    hamiltonian = microscopic_character_hamiltonian(
        cluster, states, jpm, np.zeros(3), jpmpm=0.0
    )
    sectors = translation_sector_bases(
        states, cubic16_translation_permutations(cluster)
    )
    full = np.sort(np.concatenate(
        [np.linalg.eigvalsh(block) for block in _sector_blocks(sectors, states, hamiltonian)]
    ))
    elapsed = time.perf_counter() - started

    # Independent check: dense diagonalization of the fixed-Sz translation
    # sectors (no iterative solver) must reproduce the bottom of the assembled
    # spectrum.  Sparse eigsh is deliberately NOT used as the referee: ARPACK
    # under-resolves degenerate multiplicities (observed here: 4 of 7 copies
    # of the first excited level), which fails a naive elementwise comparison.
    sz_states = fixed_magnetization_basis(cluster.n_sites, cluster.n_sites // 2)
    sz_h = microscopic_character_hamiltonian(cluster, sz_states, jpm, np.zeros(3))
    sz_sectors = translation_sector_bases(
        sz_states, cubic16_translation_permutations(cluster)
    )
    referee = np.sort(np.concatenate([
        np.linalg.eigvalsh((basis.conj().T @ (sz_h @ basis)).toarray().real)
        for basis in sz_sectors.values()
    ]))
    residual = float(np.max(np.abs(referee[:50] - full[:50])))
    if residual > 1e-9:
        raise RuntimeError(f"full spectrum fails the dense cross-check: {residual:.3e}")

    out = ROOT / "campaign" / "cache" / "full_ed_cubic16_jpm_0p046.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, E_full=full)
    print(f"jpm={jpm}: {len(full)} levels in {elapsed:.0f}s, "
          f"E0={full[0]:.8f}, sparse cross-check residual {residual:.2e}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
