#!/usr/bin/env python3
"""Low spectrum vs Jpm in the magnetized total-Sz sectors.

Note: the emergent gauge charge is the staggered divergence, whose total
vanishes identically on the bipartite torus; these are magnetization
sectors, whose excitations are gauge-neutral spinon-antispinon pairs that
carry net physical spin and therefore cannot annihilate into the ice
manifold.  Usage:
    run_jpm_levels_charged.py <sz> <jpm> [<jpm> ...]
Levels are stored absolutely; the figure references them to the neutral
ground state.  Spectra at -sz are identical by the global spin flip.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from scipy.linalg import eigh

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "notes"))

import recompute_finite_size_artifact as geometry  # noqa: E402
from qsi_campaign.exact_band import (  # noqa: E402
    fixed_magnetization_basis,
    microscopic_character_hamiltonian,
    translation_sector_bases,
    cubic16_translation_permutations,
)

OUTPUT = ROOT / "campaign" / "outputs" / "jpm_levels"
OUTPUT.mkdir(parents=True, exist_ok=True)
N_KEEP = 60


def main() -> None:
    sz = int(sys.argv[1])
    couplings = [float(v) for v in sys.argv[2:]]
    cluster = geometry.build_cluster("cubic", (1, 1, 1))
    states = fixed_magnetization_basis(cluster.n_sites,
                                       cluster.n_sites // 2 + sz)
    sectors = translation_sector_bases(
        states, cubic16_translation_permutations(cluster))

    for jpm in couplings:
        started = time.perf_counter()
        hamiltonian = microscopic_character_hamiltonian(
            cluster, states, jpm, np.zeros(3))
        values = []
        for basis in sectors.values():
            block = (basis.conj().T @ (hamiltonian @ basis)).toarray()
            block = 0.5 * (block + block.conj().T)
            seek = min(N_KEEP, block.shape[0]) - 1
            values.append(eigh(block, subset_by_index=(0, seek),
                               eigvals_only=True))
        levels = np.sort(np.concatenate(values))[:N_KEEP]
        tag = f"{jpm:+.3f}".replace(".", "p")
        np.savez_compressed(OUTPUT / f"levels_sz{sz}_{tag}.npz",
                            jpm=jpm, sz=sz, levels=levels)
        print(f"sz={sz} jpm={jpm:+.3f} E0={levels[0]:.6f} "
              f"({time.perf_counter()-started:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
