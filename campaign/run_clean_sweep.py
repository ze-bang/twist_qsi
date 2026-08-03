#!/usr/bin/env python3
"""Winding-free construction on the clean (non-proliferated) manifold.

Third reference-frame formulation: instead of the bare ice projector or the
fixed-rank dressed band, the reference is the set of pool states that remain
ice-descended by character -- eigenstates of the lowest pool whose spinon
density sits below the largest gap in the pool's density distribution.  The
rank d(lambda) is variable: states claimed by hybridization drop out (the
spectral-flow analog of the Feshbach multiplet loss), and states pushed
above the 90th level by level repulsion are recovered by character rather
than energy ordering.

Diagnostics per coupling: the selected rank, the bimodality gap of the
density distribution (the frame's own validity measure), and the corner
overlap sigma_min of the selected manifold.

Usage: run_clean_sweep.py <jpm> [<jpm> ...]
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from scipy.linalg import eigh, eigvalsh, svd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "notes"))

import recompute_finite_size_artifact as geometry  # noqa: E402
from qsi_campaign.exact_band import (  # noqa: E402
    basis_character_phases,
    fixed_magnetization_basis,
    microscopic_character_hamiltonian,
    translation_sector_bases,
    cubic16_translation_permutations,
    uniform_character_grid,
)
from qsi_campaign.protocol import sector_project  # noqa: E402

OUTPUT = ROOT / "campaign" / "outputs" / "clean_sweep"
OUTPUT.mkdir(parents=True, exist_ok=True)
N_POOL = 150


def band_peak(levels, lo=1e-7, hi=0.2, n=2600):
    shifted = np.asarray(levels) - np.min(levels)
    grid = np.geomspace(lo, hi, n)
    beta = 1.0 / grid[:, None]
    w = np.exp(-beta * shifted[None, :])
    z = w.sum(axis=1)
    mean = (w * shifted[None, :]).sum(axis=1) / z
    second = (w * shifted[None, :] ** 2).sum(axis=1) / z
    heat = (second - mean * mean) / grid**2
    return float(grid[int(np.argmax(heat))])


def select_clean(density, window=(0.08, 1.15)):
    """Split the pool at the largest density gap whose center is in window."""
    order = np.argsort(density)
    ordered = density[order]
    centers = 0.5 * (ordered[:-1] + ordered[1:])
    inside = np.where((centers >= window[0]) & (centers <= window[1]))[0]
    if len(inside) == 0:
        return np.arange(len(density)), 0.0
    gaps = ordered[inside + 1] - ordered[inside]
    cut = inside[int(np.argmax(gaps))]
    return order[: cut + 1], float(np.max(gaps))


def main() -> None:
    couplings = [float(v) for v in sys.argv[1:]]
    cluster = geometry.build_cluster("cubic", (1, 1, 1))
    states = fixed_magnetization_basis(cluster.n_sites, cluster.n_sites // 2)
    sectors = translation_sector_bases(
        states, cubic16_translation_permutations(cluster))
    ising = np.asarray(microscopic_character_hamiltonian(
        cluster, states, 0.0, np.zeros(3)).diagonal(), dtype=float)
    bits = ((np.asarray(states, dtype=np.uint64)[:, None]
             >> np.arange(cluster.n_sites, dtype=np.uint64)) & np.uint64(1))
    polarization = (bits.astype(float) - 0.5) @ np.asarray(
        cluster.positions, dtype=float)
    corner_thetas = list(uniform_character_grid(2))
    weights = np.array([1.0, np.sqrt(2.0), np.sqrt(3.0)])
    index = {int(s): i for i, s in enumerate(states)}
    ice_rows = np.asarray([index[int(s)] for s in cluster.ice_states])
    ideal_values = np.unique(np.round(polarization[ice_rows] @ weights, 9))

    for jpm in couplings:
        started = time.perf_counter()
        hamiltonian = microscopic_character_hamiltonian(
            cluster, states, jpm, np.zeros(3))
        pool_vals, pool_vecs = [], []
        for basis in sectors.values():
            block = (basis.conj().T @ (hamiltonian @ basis)).toarray()
            block = 0.5 * (block + block.conj().T)
            vals, vecs = eigh(block, subset_by_index=(0, N_POOL - 1))
            pool_vals.append(vals)
            pool_vecs.append(np.asarray(basis @ vecs))
        pool_vals = np.concatenate(pool_vals)
        pool_vecs = np.concatenate(pool_vecs, axis=1)
        order = np.argsort(pool_vals)[:N_POOL]
        energies_pool = pool_vals[order]
        psi_pool = pool_vecs[:, order]
        density = np.einsum("i,in->n", ising, np.abs(psi_pool) ** 2)

        chosen, bimodality = select_clean(density)
        chosen = np.sort(chosen)
        rank = len(chosen)
        energies = energies_pool[chosen]
        psi = psi_pool[:, chosen]

        raw_overlaps = []
        for theta in corner_thetas:
            phases = basis_character_phases(cluster, states, np.asarray(theta))
            raw_overlaps.append(psi.conj().T @ (phases[:, None] * psi))

        # Gauge-closure pruning: drop combinations whose corner images leave
        # the selected span, until the manifold is closed under the corner
        # action (the algebraic content of "not spinon proliferated").
        keep = np.eye(rank, dtype=complex)
        pruned = 0
        for _ in range(rank - 40):
            reduced = [keep.conj().T @ o @ keep for o in raw_overlaps]
            coherence = np.mean([r @ r.conj().T for r in reduced], axis=0)
            coh_vals, coh_vecs = np.linalg.eigh(coherence)
            if coh_vals[0] > 0.60:
                break
            keep = keep @ coh_vecs[:, 1:]
            pruned += 1
        rank = keep.shape[1]
        h_zero = keep.conj().T @ (energies[:, None] * keep)
        raw_selected = np.linalg.eigvalsh(0.5 * (h_zero + h_zero.conj().T))

        # Pool gauge-coherence spectrum: the sorted eigenvalues of the
        # averaged corner closure operator on the full pool; the size of its
        # large-eigenvalue block is the rank of the most gauge-coherent
        # subspace that exists at this coupling, over all possible frames.
        pool_overlaps = []
        for theta in corner_thetas:
            phases = basis_character_phases(cluster, states, np.asarray(theta))
            pool_overlaps.append(
                psi_pool.conj().T @ (phases[:, None] * psi_pool))
        pool_coherence = np.mean(
            [o @ o.conj().T for o in pool_overlaps], axis=0)
        coherence_spectrum = np.sort(np.linalg.eigvalsh(
            0.5 * (pool_coherence + pool_coherence.conj().T)))[::-1]

        corners, sigma_min = [], 1.0
        for o in raw_overlaps:
            reduced = keep.conj().T @ o @ keep
            u, s, vh = svd(reduced)
            sigma_min = min(sigma_min, float(s.min()))
            rotation = u @ vh
            corners.append(rotation @ h_zero @ rotation.conj().T)
        averaged = np.mean(corners, axis=0)
        projected = np.stack([
            keep.conj().T @ (psi.conj().T
                             @ (polarization[:, a][:, None] * psi)) @ keep
            for a in range(3)])
        mixer = (weights[0] * projected[0] + weights[1] * projected[1]
                 + weights[2] * projected[2])
        mixer = 0.5 * (mixer + mixer.conj().T)
        mix_vals, mix_vecs = np.linalg.eigh(mixer)
        labels = np.argmin(np.abs(mix_vals[:, None]
                                  - ideal_values[None, :]), axis=1)
        drift = float(np.max(np.abs(mix_vals - ideal_values[labels])))
        rotated = mix_vecs.conj().T @ averaged @ mix_vecs
        clean_band = np.linalg.eigvalsh(sector_project(rotated, labels))

        record = {
            "jpm": jpm, "rank": rank, "pruned": pruned,
            "bimodality": bimodality,
            "clean_band": clean_band, "clean_peak": band_peak(clean_band),
            "corner_sigma_min": sigma_min, "sector_drift": drift,
            "density_pool": density, "energies_pool": energies_pool,
            "raw_selected": raw_selected,
            "coherence_spectrum": coherence_spectrum,
        }
        tag = f"{jpm:+.3f}".replace(".", "p")
        np.savez_compressed(OUTPUT / f"clean_{tag}.npz", **record)
        g6 = 12.0 * abs(jpm) ** 3
        print(f"jpm={jpm:+.3f} rank={rank}(-{pruned}) peak/g6={record['clean_peak']/g6:.4f} "
              f"bimod={bimodality:.3f} sigma_min={sigma_min:.3f} "
              f"drift={drift:.3f} ({time.perf_counter()-started:.0f}s)",
              flush=True)


if __name__ == "__main__":
    main()
