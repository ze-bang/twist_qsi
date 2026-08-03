#!/usr/bin/env python3
"""Low spectrum at theta = 0 as a function of Jpm, with spinon density.

Same dense per-sector diagonalization as run_theta_bands.py, swept over the
coupling instead of the twist.  Usage: run_jpm_levels.py <jpm> [<jpm> ...].
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
N_KEEP = 150


def main() -> None:
    couplings = [float(v) for v in sys.argv[1:]]
    cluster = geometry.build_cluster("cubic", (1, 1, 1))
    states = fixed_magnetization_basis(cluster.n_sites, cluster.n_sites // 2)
    index = {int(s): i for i, s in enumerate(states)}
    ice_rows = np.asarray([index[int(s)] for s in cluster.ice_states])
    sectors = translation_sector_bases(
        states, cubic16_translation_permutations(cluster))
    ising = np.asarray(microscopic_character_hamiltonian(
        cluster, states, 0.0, np.zeros(3)).diagonal(), dtype=float)

    for jpm in couplings:
        started = time.perf_counter()
        hamiltonian = microscopic_character_hamiltonian(
            cluster, states, jpm, np.zeros(3))
        values, ice_w, pair_w = [], [], []
        for basis in sectors.values():
            block = (basis.conj().T @ (hamiltonian @ basis)).toarray()
            block = 0.5 * (block + block.conj().T)
            vals, vecs = eigh(block, subset_by_index=(0, N_KEEP - 1))
            density = np.abs(np.asarray(basis @ vecs)) ** 2
            values.append(vals)
            ice_w.append(density[ice_rows, :].sum(axis=0))
            pair_w.append(ising @ density)
        values = np.concatenate(values)
        ice_w = np.concatenate(ice_w)
        pair_w = np.concatenate(pair_w)
        order = np.argsort(values)[:N_KEEP]
        tag = f"{jpm:+.3f}".replace(".", "p")
        np.savez_compressed(OUTPUT / f"levels_{tag}.npz",
                            jpm=jpm, levels=values[order],
                            ice_weight=ice_w[order],
                            spinon_pairs=pair_w[order])
        print(f"jpm={jpm:+.3f} E0={values[order][0]:.6f} "
              f"({time.perf_counter()-started:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
