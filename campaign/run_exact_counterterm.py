#!/usr/bin/env python3
"""Exact winding-free ED at any coupling: the mask realised as a Hamiltonian.

A counterterm supported entirely inside the ice manifold P does not touch QHQ
or QHP, so it enters the Feshbach map additively and without dressing:

    F_C(z) = P(H + C)P + PHQ (z - QHQ)^{-1} QHP = F(z) + C .

Choosing

    C = - (transport-changing part of F(z_0)) ,   z_0 = E_0 ,

therefore produces a genuine Hermitian operator H_wf = H + C on the FULL
Hilbert space whose ice-block Feshbach map is exactly the polarization-masked
map -- winding-free to all orders and in every loop class at once, with no
band, no frame, no Gram matrix, no spectral isolation and no loop-class
bookkeeping.  Everything ordinary ED provides (spectra, observables,
thermodynamics, the spinon sector) is then available from H_wf directly.

Per coupling this script builds C, diagonalizes H_wf, and reports the band, its
ice weight and isolation, the ground-multiplet hexagon flux and winding
amplitude, the ring heat-capacity peak, the corner coherence of the band, and
the agreement between the H_wf spectrum and the self-consistent masked Feshbach
roots (the two constructions are the same object seen from two sides; their
difference is the residual z-dependence of the self-energy).

Usage: run_exact_counterterm.py <jpm> [<jpm> ...] [--iterations N]
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.linalg import eigh, svdvals
from scipy.sparse import csr_matrix

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "notes"))
sys.path.insert(0, str(ROOT / "campaign"))

import recompute_finite_size_artifact as geometry  # noqa: E402
from qsi_campaign.exact_band import (  # noqa: E402
    basis_character_phases,
    fixed_magnetization_basis,
    microscopic_character_hamiltonian,
    translation_sector_bases,
    cubic16_translation_permutations,
    uniform_character_grid,
)
from qsi_campaign.downfold import build_blocks, solve_roots  # noqa: E402
from qsi_campaign.protocol import polarization_sector_labels  # noqa: E402
from run_winding_residual import feshbach_matrix  # noqa: E402
from run_strong_coupling_gs import (  # noqa: E402
    loop_expectation,
    transverse_density_matrix,
)
from run_feshbach_anatomy import band_peak  # noqa: E402

OUTPUT = ROOT / "campaign" / "outputs" / "exact_counterterm"
OUTPUT.mkdir(parents=True, exist_ok=True)
N_POOL = 150
N_BAND = 90


def embed(block: np.ndarray, rows: np.ndarray, dimension: int) -> csr_matrix:
    """Embed a dense ice-block operator into the full Hilbert space."""
    size = len(rows)
    row_index = np.repeat(rows, size)
    column_index = np.tile(rows, size)
    return csr_matrix((block.ravel(), (row_index, column_index)),
                      shape=(dimension, dimension))


def main() -> None:
    arguments = [v for v in sys.argv[1:] if not v.startswith("--")]
    iterations = 3 if "--iterations" not in sys.argv else int(
        sys.argv[sys.argv.index("--iterations") + 1])
    couplings = [float(v) for v in arguments]

    cluster = geometry.build_cluster("cubic", (1, 1, 1))
    states = fixed_magnetization_basis(cluster.n_sites, cluster.n_sites // 2)
    index = {int(s): i for i, s in enumerate(states)}
    sectors = translation_sector_bases(
        states, cubic16_translation_permutations(cluster))
    ice_rows = np.asarray([index[int(s)] for s in cluster.ice_states])
    outside = np.ones(len(states), dtype=bool)
    outside[ice_rows] = False
    complement = np.where(outside)[0]
    labels = polarization_sector_labels(cluster)
    cross = labels[:, None] != labels[None, :]
    defects = (np.bitwise_count(
        np.asarray(states, dtype=np.uint64)[:, None]
        & cluster.tet_masks[None, :]) != 2).sum(axis=1).astype(float)
    contractible = [sites for sites, wind in cluster.hexes
                    if tuple(wind) == (0, 0, 0)]
    winding4 = [sites for sites, _ in cluster.loops4]
    corner_phases = [basis_character_phases(cluster, states, np.asarray(t))
                     for t in uniform_character_grid(2)]

    def spectrum(hamiltonian, count):
        values, vectors = [], []
        for basis in sectors.values():
            block = (basis.conj().T @ (hamiltonian @ basis)).toarray()
            block = 0.5 * (block + block.conj().T)
            block = np.ascontiguousarray(block.real)
            val, vec = eigh(block, subset_by_index=(0, count - 1))
            values.append(val)
            vectors.append(np.asarray(basis @ vec))
        values = np.concatenate(values)
        vectors = np.concatenate(vectors, axis=1)
        order = np.argsort(values)[:count]
        return values[order], vectors[:, order]

    for jpm in couplings:
        started = time.perf_counter()
        bare = microscopic_character_hamiltonian(
            cluster, states, jpm, np.zeros(3)).tocsc().real
        energy, _ = spectrum(bare, 1)
        anchor = float(energy[0])

        history = []
        for _ in range(iterations):
            f_matrix = feshbach_matrix(bare, ice_rows, complement, anchor)
            counterterm = -np.where(cross, f_matrix, 0.0)
            hamiltonian = (bare + embed(counterterm, ice_rows,
                                        len(states))).tocsc()
            levels, pool = spectrum(hamiltonian, N_POOL)
            history.append(float(levels[0]))
            if abs(levels[0] - anchor) < 1e-10:
                anchor = float(levels[0])
                break
            anchor = float(levels[0])

        band = pool[:, :N_BAND]
        multiplet = pool[:, levels - levels[0] < 1e-9]

        ice_weight = float(np.mean([
            np.sum(np.abs(multiplet[ice_rows, column]) ** 2)
            for column in range(multiplet.shape[1])]))
        band_ice = float(np.mean(
            np.sum(np.abs(band[ice_rows, :]) ** 2, axis=0)))
        defect_number = float(np.mean([
            np.sum(defects * np.abs(multiplet[:, column]) ** 2)
            for column in range(multiplet.shape[1])]))
        flux = float(np.mean([
            loop_expectation(states, index, sites, multiplet)
            for sites in contractible]))
        winding = float(np.mean([
            loop_expectation(states, index, sites, multiplet)
            for sites in winding4]))
        density = transverse_density_matrix(states, index, cluster.n_sites,
                                            multiplet)
        occupations = np.linalg.eigvalsh(density)[::-1]

        overlaps = [band.conj().T @ (phase[:, None] * band)
                    for phase in corner_phases]
        closure = np.mean([o @ o.conj().T for o in overlaps], axis=0)
        coherence = np.sort(np.linalg.eigvalsh(
            0.5 * (closure + closure.conj().T)))[::-1]
        sigma_min = min(float(svdvals(o).min()) for o in overlaps)

        # the same object from the other side: masked Feshbach roots
        blocks = build_blocks(cluster, states, ice_rows,
                              microscopic_character_hamiltonian(
                                  cluster, states, jpm, np.zeros(3)))
        masked = solve_roots(blocks, masked=True)
        roots = np.sort(masked.energies[masked.converged])
        n_roots = int(masked.converged.sum())
        if n_roots:
            common = min(n_roots, N_BAND)
            agreement = float(np.max(np.abs(
                roots[:common] - levels[:common])))
            relative = agreement / max(
                float(levels[N_BAND - 1] - levels[0]), 1e-300)
        else:
            agreement = np.nan
            relative = np.nan

        record = {
            "jpm": jpm,
            "iterations": len(history),
            "energy": float(levels[0]),
            "energy_history": history,
            "degeneracy": int(multiplet.shape[1]),
            "band_width": float(levels[N_BAND - 1] - levels[0]),
            "band_gap": float(levels[N_BAND] - levels[N_BAND - 1]),
            "ice_weight": ice_weight,
            "band_ice_weight": band_ice,
            "defect_number": defect_number,
            "hexagon_flux": flux,
            "winding4_amplitude": winding,
            "condensate_fraction": float(occupations[0] / cluster.n_sites),
            "ring_peak": band_peak(levels[:N_BAND]),
            "ring_peak_over_g6": band_peak(levels[:N_BAND])
            / (12.0 * abs(jpm) ** 3),
            "coherence_min": float(coherence[-1]),
            "coherence_rank_0p8": int((coherence > 0.8).sum()),
            "corner_sigma_min": sigma_min,
            "counterterm_norm": float(np.linalg.norm(counterterm)),
            "masked_roots": n_roots,
            "root_agreement": agreement,
            "root_agreement_relative": relative,
            "runtime": time.perf_counter() - started,
        }
        tag = f"{jpm:+.3f}".replace(".", "p")
        with open(OUTPUT / f"exact_ct_{tag}.json", "w") as handle:
            json.dump(record, handle, indent=1, default=float)
        np.savez_compressed(OUTPUT / f"exact_ct_{tag}.npz", jpm=jpm,
                            levels=levels, roots=roots, coherence=coherence,
                            counterterm=counterterm, occupations=occupations)
        print(f"jpm={jpm:+.3f} E={levels[0]:+.6f} ice={ice_weight:.4f} "
              f"band_ice={band_ice:.4f} flux6={flux:+.4f} wind4={winding:+.4f} "
              f"peak/g6={record['ring_peak_over_g6']:.3f} "
              f"width={record['band_width']:.5f} gap={record['band_gap']:.5f} "
              f"roots={n_roots} dE={agreement:.2e} "
              f"({relative:.1e} of width) ({record['runtime']:.0f}s)",
              flush=True)


if __name__ == "__main__":
    main()
