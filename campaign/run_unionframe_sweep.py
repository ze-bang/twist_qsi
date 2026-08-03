#!/usr/bin/env python3
"""Winding-free construction on the surviving ice sector below the charge gap.

Fourth formulation (the physically canonical one): the reference manifold is
every neutral (S^z_tot = 0) eigenstate lying below the ground state of the
S^z_tot = 1 sector -- i.e. everything beneath the charge gap.  The cutoff is
an energy, not a rank, so the manifold adapts as singlet pair states descend,
and at the M = 2 gauge corners the criterion is exactly corner-consistent
because the corner Hamiltonians are isospectral to theta = 0.

Requires the charged-sector floors from run_jpm_levels_charged
(levels_sz1_*.npz); computes them inline when absent.

Usage: run_gapframe_sweep.py <jpm> [<jpm> ...]
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

OUTPUT = ROOT / "campaign" / "outputs" / "unionframe_sweep"
OUTPUT.mkdir(parents=True, exist_ok=True)
CHARGED = ROOT / "campaign" / "outputs" / "jpm_levels"
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


def charged_floor(cluster, jpm):
    tag = f"{jpm:+.3f}".replace(".", "p")
    path = CHARGED / f"levels_sz1_{tag}.npz"
    if path.exists():
        return float(np.load(path)["levels"][0])
    states = fixed_magnetization_basis(cluster.n_sites, cluster.n_sites // 2 + 1)
    sectors = translation_sector_bases(
        states, cubic16_translation_permutations(cluster))
    hamiltonian = microscopic_character_hamiltonian(
        cluster, states, jpm, np.zeros(3))
    floor = np.inf
    for basis in sectors.values():
        block = (basis.conj().T @ (hamiltonian @ basis)).toarray()
        block = 0.5 * (block + block.conj().T)
        floor = min(floor, float(eigh(block, subset_by_index=(0, 0),
                                      eigvals_only=True)[0]))
    return floor


def main() -> None:
    couplings = [float(v) for v in sys.argv[1:]]
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
    index = {int(s): i for i, s in enumerate(states)}
    ice_rows = np.asarray([index[int(s)] for s in cluster.ice_states])
    ideal_values = np.unique(np.round(polarization[ice_rows] @ weights, 9))

    ising = np.asarray(microscopic_character_hamiltonian(
        cluster, states, 0.0, np.zeros(3)).diagonal(), dtype=float)

    for jpm in couplings:
        started = time.perf_counter()
        cutoff = charged_floor(cluster, jpm)
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
        # ice-descended by character: below the largest density gap whose
        # center lies in the mixing window
        order_d = np.argsort(density)
        ordered = density[order_d]
        centers = 0.5 * (ordered[:-1] + ordered[1:])
        inside = np.where((centers >= 0.08) & (centers <= 1.15))[0]
        icelike = np.zeros(N_POOL, dtype=bool)
        if len(inside):
            cut = inside[int(np.argmax(ordered[inside + 1] - ordered[inside]))]
            icelike[order_d[: cut + 1]] = True
        else:
            icelike[:] = True
        chosen = icelike & (energies_pool < cutoff)
        rank = int(chosen.sum())
        energies = energies_pool[chosen]
        psi = psi_pool[:, chosen]
        window_margin = float(cutoff - energies[-1]) if rank else np.nan

        corners, sigma_min = [], 1.0
        raw_overlaps = []
        for theta in corner_thetas:
            phases = basis_character_phases(cluster, states, np.asarray(theta))
            overlap = psi.conj().T @ (phases[:, None] * psi)
            raw_overlaps.append(overlap)
            u, s, vh = svd(overlap)
            sigma_min = min(sigma_min, float(s.min()))
            rotation = u @ vh
            corners.append(rotation @ np.diag(energies) @ rotation.conj().T)
        averaged = np.mean(corners, axis=0)
        coherence = np.mean([o @ o.conj().T for o in raw_overlaps], axis=0)
        coh_spectrum = np.sort(np.linalg.eigvalsh(
            0.5 * (coherence + coherence.conj().T)))[::-1]

        projected = np.stack([
            psi.conj().T @ (polarization[:, a][:, None] * psi)
            for a in range(3)])
        mixer = (weights[0] * projected[0] + weights[1] * projected[1]
                 + weights[2] * projected[2])
        mixer = 0.5 * (mixer + mixer.conj().T)
        mix_vals, mix_vecs = np.linalg.eigh(mixer)
        labels = np.argmin(np.abs(mix_vals[:, None]
                                  - ideal_values[None, :]), axis=1)
        drift = float(np.max(np.abs(mix_vals - ideal_values[labels])))
        rotated = mix_vecs.conj().T @ averaged @ mix_vecs
        gap_band = np.linalg.eigvalsh(sector_project(rotated, labels))

        tag = f"{jpm:+.3f}".replace(".", "p")
        np.savez_compressed(
            OUTPUT / f"unionframe_{tag}.npz",
            jpm=jpm, rank=rank, cutoff=cutoff,
            window_margin=window_margin,
            raw_selected=energies, union_band=gap_band,
            union_peak=band_peak(gap_band),
            corner_sigma_min=sigma_min, sector_drift=drift,
            coherence_spectrum=coh_spectrum)
        g6 = 12.0 * abs(jpm) ** 3
        print(f"jpm={jpm:+.3f} rank={rank} peak/g6={band_peak(gap_band)/g6:.4f} "
              f"sigma_min={sigma_min:.3f} coh_med={np.median(coh_spectrum):.3f} "
              f"drift={drift:.3f} margin={window_margin:.3f} "
              f"({time.perf_counter()-started:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
