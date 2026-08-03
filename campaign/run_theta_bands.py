#!/usr/bin/env python3
"""Low spectrum of the sourced Hamiltonian H(theta) with ice character.

For a set of couplings, diagonalizes H(theta) along the line
theta = (0, 0, t), t in [0, pi] (the spectrum at -t is identical by complex
conjugation, and t = pi is an exact gauge corner), keeping the lowest levels
and, for each eigenstate, its ice weight <psi|P_ice|psi>.  Dense
diagonalization of the four translation sectors is used throughout, so the
result is exact at any coupling, including past the isolation boundary.

Output: campaign/outputs/theta_bands_jpm<t>.npz per coupling with arrays
``t`` (n_t), ``levels`` (n_t, n_keep), ``ice_weight`` (n_t, n_keep).
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

OUTPUT = ROOT / "campaign" / "outputs"
N_KEEP = 150
N_T = 9


def main() -> None:
    couplings = [float(v) for v in sys.argv[1:]] or [-0.05, -0.18, -0.22]
    cluster = geometry.build_cluster("cubic", (1, 1, 1))
    states = fixed_magnetization_basis(cluster.n_sites, cluster.n_sites // 2)
    index = {int(s): i for i, s in enumerate(states)}
    ice_rows = np.asarray([index[int(s)] for s in cluster.ice_states])
    sectors = translation_sector_bases(
        states, cubic16_translation_permutations(cluster))

    # Ising diagonal of the basis states: (1/2) sum_t Q_t^2 in units of Jzz,
    # i.e. the charge content; one elementary +-1 spinon pair contributes 1.
    ising = np.asarray(microscopic_character_hamiltonian(
        cluster, states, 0.0, np.zeros(3)).diagonal(), dtype=float)

    ts = np.linspace(0.0, np.pi, N_T)
    for jpm in couplings:
        levels = np.zeros((N_T, N_KEEP))
        weights = np.zeros((N_T, N_KEEP))
        pairs = np.zeros((N_T, N_KEEP))
        for it, t in enumerate(ts):
            started = time.perf_counter()
            hamiltonian = microscopic_character_hamiltonian(
                cluster, states, jpm, np.array([0.0, 0.0, t]))
            values, ice_w, pair_w = [], [], []
            for basis in sectors.values():
                block = (basis.conj().T @ (hamiltonian @ basis)).toarray()
                block = 0.5 * (block + block.conj().T)
                seek = min(N_KEEP, block.shape[0]) - 1
                vals, vecs = eigh(block, subset_by_index=(0, seek))
                density = np.abs(np.asarray(basis @ vecs)) ** 2
                values.append(vals)
                ice_w.append(density[ice_rows, :].sum(axis=0))
                pair_w.append(ising @ density)
            values = np.concatenate(values)
            ice_w = np.concatenate(ice_w)
            pair_w = np.concatenate(pair_w)
            order = np.argsort(values)[:N_KEEP]
            levels[it] = values[order]
            weights[it] = ice_w[order]
            pairs[it] = pair_w[order]
            print(f"jpm={jpm:+.3f} t={t:.3f} "
                  f"E0={levels[it,0]:.6f} gap90={levels[it,90]-levels[it,89]:.4f} "
                  f"({time.perf_counter()-started:.0f}s)", flush=True)
        tag = f"{jpm:+.3f}".replace(".", "p")
        np.savez_compressed(OUTPUT / f"theta_bands_jpm{tag}.npz",
                            t=ts, levels=levels, ice_weight=weights,
                            spinon_pairs=pairs, jpm=jpm)
        print(f"wrote theta_bands_jpm{tag}.npz", flush=True)


if __name__ == "__main__":
    main()
