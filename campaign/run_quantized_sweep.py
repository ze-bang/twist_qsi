#!/usr/bin/env python3
"""Winding-free construction on the maximal transport-quantized subspace.

Fifth (and principled) formulation: the reference manifold is the largest
subspace of the low-energy pool on which polarization transport is integer
quantized -- the domain where the character average is genuinely topological.
Construction: diagonalize the pool-projected polarization (joint transport
eigenbasis); keep every vector whose full-space transport mean sits on the
half-integer lattice (deviation < DTOL) with small full-space transport
variance (< VTOL); run the corner average + mask on that span.

Usage: run_quantized_sweep.py [--probe] <jpm> [<jpm> ...]
--probe prints the variance/deviation distributions instead of selecting.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from scipy.linalg import eigh, svd

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

OUTPUT = ROOT / "campaign" / "outputs" / "quantized_sweep"
OUTPUT.mkdir(parents=True, exist_ok=True)
N_POOL = 150
VTOL = 0.20
DTOL = 0.12


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


def main() -> None:
    argv = sys.argv[1:]
    probe = "--probe" in argv
    couplings = [float(v) for v in argv if not v.startswith("--")]
    cluster = geometry.build_cluster("cubic", (1, 1, 1))
    states = fixed_magnetization_basis(cluster.n_sites, cluster.n_sites // 2)
    sectors = translation_sector_bases(
        states, cubic16_translation_permutations(cluster))
    bits = ((np.asarray(states, dtype=np.uint64)[:, None]
             >> np.arange(cluster.n_sites, dtype=np.uint64)) & np.uint64(1))
    polarization = (bits.astype(float) - 0.5) @ np.asarray(
        cluster.positions, dtype=float)
    corner_thetas = list(uniform_character_grid(2))
    weights = np.array([1.0, np.sqrt(2.0), np.sqrt(3.0)])

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
        energies = pool_vals[order]
        psi = pool_vecs[:, order]

        # joint transport eigenbasis of the pool
        projected = np.stack([
            psi.conj().T @ (polarization[:, a][:, None] * psi)
            for a in range(3)])
        mixer = sum(w * p for w, p in zip(weights, projected))
        mixer = 0.5 * (mixer + mixer.conj().T)
        _, mix_vecs = np.linalg.eigh(mixer)
        columns = psi @ mix_vecs                       # full-space vectors
        density = np.abs(columns) ** 2
        mean_p = np.stack([density.T @ polarization[:, a] for a in range(3)],
                          axis=1)
        var_p = np.stack([density.T @ polarization[:, a] ** 2
                          for a in range(3)], axis=1) - mean_p**2
        variance = var_p.sum(axis=1)
        nearest = np.round(2.0 * mean_p) / 2.0
        deviation = np.max(np.abs(mean_p - nearest), axis=1)

        if probe:
            for pct in (10, 50, 75, 90):
                print(f"jpm={jpm:+.3f} pct{pct}: var={np.percentile(variance, pct):.4f} "
                      f"dev={np.percentile(deviation, pct):.4f}", flush=True)
            chosen = (variance < VTOL) & (deviation < DTOL)
            print(f"  -> would select rank {int(chosen.sum())}", flush=True)
            continue

        chosen = np.where((variance < VTOL) & (deviation < DTOL))[0]
        rank = len(chosen)
        if rank < 6:
            tag = f"{jpm:+.3f}".replace(".", "p")
            np.savez_compressed(OUTPUT / f"quantized_{tag}.npz",
                                jpm=jpm, rank=rank, variance=variance,
                                deviation=deviation)
            print(f"jpm={jpm:+.3f} rank={rank} EMPTY "
                  f"({time.perf_counter()-started:.0f}s)", flush=True)
            continue
        keep = mix_vecs[:, chosen]
        h_zero = keep.conj().T @ (energies[:, None] * keep)
        labels_q = nearest[chosen]
        _, labels = np.unique(labels_q, axis=0, return_inverse=True)

        corners, sigma_min = [], 1.0
        for theta in corner_thetas:
            phases = basis_character_phases(cluster, states, np.asarray(theta))
            overlap = psi.conj().T @ (phases[:, None] * psi)
            reduced = keep.conj().T @ overlap @ keep
            u, s, vh = svd(reduced)
            sigma_min = min(sigma_min, float(s.min()))
            rotation = u @ vh
            corners.append(rotation @ h_zero @ rotation.conj().T)
        averaged = np.mean(corners, axis=0)
        quantized_band = np.linalg.eigvalsh(sector_project(averaged, labels))

        tag = f"{jpm:+.3f}".replace(".", "p")
        raw_selected = np.linalg.eigvalsh(0.5 * (h_zero + h_zero.conj().T))
        np.savez_compressed(
            OUTPUT / f"quantized_{tag}.npz",
            jpm=jpm, rank=rank,
            variance=variance, deviation=deviation,
            raw_selected=raw_selected,
            quantized_band=quantized_band,
            quantized_peak=band_peak(quantized_band),
            corner_sigma_min=sigma_min)
        g6 = 12.0 * abs(jpm) ** 3
        print(f"jpm={jpm:+.3f} rank={rank} "
              f"peak/g6={band_peak(quantized_band)/g6:.4f} "
              f"sigma_min={sigma_min:.3f} "
              f"({time.perf_counter()-started:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
