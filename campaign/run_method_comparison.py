#!/usr/bin/env python3
"""Specific heat from every available method, on one footing, with checks.

For each coupling in [-0.30, +0.05] this computes C(T)/N over the COMPLETE
65536-state spectrum (all magnetization sectors, Klein-blocked) for

  periodic   the raw cluster -- the baseline, no winding removal;
  polar      the original construction: Loewdin/polar pullback of the exact
             band, masked, spliced back into the full spectrum;
  feshbach   the masked Feshbach roots, spliced the same way (frame-free);
  hc         the counterterm Hamiltonian H + C, whose own spectrum needs no
             splice.

Convergence and physicality are checked, not assumed:

  * entropy sum rule, integral of C/T dT = N ln 2, for every route;
  * temperature-grid refinement: peak positions recomputed on a doubled grid;
  * mask convergence: the exact transport-sector mask against the M = 2
    polarization-parity mask of Theorem B;
  * frame health for the polar route (Gram minimum, gap above the band);
  * completeness for the Feshbach route (number of sub-continuum roots);
  * anchoring residue for H + C.

Usage: run_method_comparison.py <jpm> [<jpm> ...]
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.linalg import eigh, eigvalsh
from scipy.signal import argrelmax

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "notes"))
sys.path.insert(0, str(ROOT / "campaign"))

import recompute_finite_size_artifact as geometry  # noqa: E402
from qsi_campaign.exact_band import (  # noqa: E402
    extract_exact_band,
    fixed_magnetization_basis,
    microscopic_character_hamiltonian,
    translation_sector_bases,
    cubic16_translation_permutations,
)
from qsi_campaign.downfold import build_blocks, solve_roots  # noqa: E402
from qsi_campaign.protocol import (  # noqa: E402
    polarization_sector_labels,
    sector_project,
    full_hilbert_counterterm_spectrum,
)
from run_winding_residual import feshbach_matrix  # noqa: E402
from run_exact_counterterm import embed  # noqa: E402

OUTPUT = ROOT / "campaign" / "outputs" / "method_comparison"
OUTPUT.mkdir(parents=True, exist_ok=True)
N_SITES = 16
GRID = np.geomspace(1e-7, 4.0, 3000)
FINE = np.geomspace(1e-7, 4.0, 6000)
SGRID = np.geomspace(1e-8, 60.0, 6000)   # for the entropy sum rule


def heat_capacity(levels, grid=GRID):
    energies = np.sort(np.asarray(levels, dtype=float))
    energies = energies - energies[0]
    beta = 1.0 / grid[:, None]
    weight = np.exp(-beta * energies[None, :])
    partition = weight.sum(axis=1)
    mean = (weight * energies[None, :]).sum(axis=1) / partition
    second = (weight * energies[None, :] ** 2).sum(axis=1) / partition
    return (second - mean * mean) / (N_SITES * grid**2)


def entropy_sum_rule(levels, grid=SGRID):
    """Integral of (C/T) dT per site, plus the residual ground entropy.

    The total must be ln 2: the spectrum has 2^16 levels whatever the method
    did to them, so any deviation means a splice lost or duplicated a level.
    An exactly degenerate ground multiplet keeps ln(g0) of entropy frozen at
    T = 0, and that has to be added back before comparing.
    """
    energies = np.sort(np.asarray(levels, dtype=float))
    degeneracy = int(np.sum(energies - energies[0] < 1e-12))
    curve = heat_capacity(energies, grid)
    released = float(np.trapezoid(curve / grid, grid))
    return released, degeneracy, released + np.log(degeneracy) / N_SITES


def find_peaks(curve, grid=GRID, floor=1e-4, order=16):
    return [(float(grid[i]), float(curve[i]))
            for i in argrelmax(curve, order=order)[0] if curve[i] > floor]


def classify(peaks, grid, curve, dip=0.70):
    """(gauge, spinon, resolved) -- gauge is the tallest below the spinon."""
    if not peaks:
        return None, None, False
    spinon = max(peaks, key=lambda p: p[0])
    below = [p for p in peaks if p[0] < 0.5 * spinon[0]]
    if not below:
        return None, spinon, False
    gauge = max(below, key=lambda p: p[1])
    lo = int(np.argmin(np.abs(grid - gauge[0])))
    hi = int(np.argmin(np.abs(grid - spinon[0])))
    ratio = (float(curve[lo:hi].min()) / min(gauge[1], spinon[1])
             if hi > lo else np.nan)
    return gauge, spinon, bool(ratio < dip)


def full_spectrum(cluster, jpm, permutations):
    levels = []
    for n_up in range(cluster.n_sites + 1):
        states = fixed_magnetization_basis(cluster.n_sites, n_up)
        hamiltonian = microscopic_character_hamiltonian(
            cluster, states, jpm, np.zeros(3))
        if len(states) < 8:
            levels.append(eigvalsh(np.asarray(hamiltonian.todense()).real))
            continue
        for basis in translation_sector_bases(states, permutations).values():
            block = (basis.conj().T @ (hamiltonian @ basis)).toarray()
            block = 0.5 * (block + block.conj().T)
            levels.append(eigvalsh(np.ascontiguousarray(block.real)))
    return np.sort(np.concatenate(levels))


def main() -> None:
    couplings = [float(v) for v in sys.argv[1:]]
    cluster = geometry.build_cluster("cubic", (1, 1, 1))
    permutations = cubic16_translation_permutations(cluster)
    states = fixed_magnetization_basis(cluster.n_sites, cluster.n_sites // 2)
    index = {int(s): i for i, s in enumerate(states)}
    ice_rows = np.asarray([index[int(s)] for s in cluster.ice_states])
    sectors = translation_sector_bases(states, permutations)
    labels = polarization_sector_labels(cluster)
    cross = labels[:, None] != labels[None, :]
    outside = np.ones(len(states), dtype=bool)
    outside[ice_rows] = False
    complement = np.where(outside)[0]

    ice_states = np.asarray(cluster.ice_states, dtype=np.uint64)
    bits = ((ice_states[:, None] >> np.arange(16, dtype=np.uint64))
            & np.uint64(1))
    polarization = (bits.astype(float) - 0.5) @ np.asarray(cluster.positions)
    parity = np.round(np.mod(2 * polarization, 2)).astype(int)
    parity_labels = np.unique(parity, axis=0, return_inverse=True)[1]

    for jpm in couplings:
        started = time.perf_counter()
        record = {"jpm": jpm, "g4": 4.0 * jpm * jpm,
                  "g6": 12.0 * abs(jpm) ** 3}
        spectra = {}

        # ---------------------------------------------------------- periodic
        spectra["periodic"] = full_spectrum(cluster, jpm, permutations)
        hamiltonian = microscopic_character_hamiltonian(
            cluster, states, jpm, np.zeros(3))

        # ------------------------------------------------------------- polar
        try:
            band = extract_exact_band(hamiltonian, sectors, ice_rows, 90)
            operator = band.operator
            polar_levels = np.linalg.eigvalsh(
                sector_project(operator, labels)).real
            parity_levels = np.linalg.eigvalsh(
                sector_project(operator, parity_labels)).real
            spectra["polar"] = full_hilbert_counterterm_spectrum(
                spectra["periodic"], polar_levels)
            record["frame_ok"] = True
            record["gram_min"] = float(band.diagnostics["model_overlap_min"])
            record["gap_above_band"] = float(band.fixed_sz_gap_above_band)
            record["mask_convergence"] = float(
                np.max(np.abs(np.sort(polar_levels) - np.sort(parity_levels))))
            record["band_width_polar"] = float(np.ptp(polar_levels))
        except Exception as error:
            record["frame_ok"] = False
            record["frame_error"] = str(error)[:140]

        # ---------------------------------------------------------- feshbach
        blocks = build_blocks(cluster, states, ice_rows, hamiltonian)
        masked = solve_roots(blocks, masked=True)
        roots = np.sort(masked.energies[masked.converged])
        record["n_roots"] = int(len(roots))
        record["z_mean"] = float(np.nanmean(masked.ice_weight))
        if len(roots) == 90:
            spectra["feshbach"] = full_hilbert_counterterm_spectrum(
                spectra["periodic"], roots)

        # ---------------------------------------------------------------- HC
        anchor = float(np.min([
            eigh(np.ascontiguousarray(
                (basis.conj().T @ (hamiltonian @ basis)).toarray().real),
                subset_by_index=(0, 0), eigvals_only=True)[0]
            for basis in sectors.values()]))
        bare = hamiltonian.tocsc().real
        for _ in range(3):
            f_matrix = feshbach_matrix(bare, ice_rows, complement, anchor)
            corrected = (bare + embed(-np.where(cross, f_matrix, 0.0),
                                      ice_rows, len(states))).tocsc()
            zero_levels = np.sort(np.concatenate([
                eigvalsh(np.ascontiguousarray(
                    (basis.conj().T @ (corrected @ basis)).toarray().real))
                for basis in sectors.values()]))
            if abs(zero_levels[0] - anchor) < 1e-10:
                break
            anchor = float(zero_levels[0])
        # the S^z = 0 block is replaced wholesale; all other sectors are shared
        rest = full_spectrum_other(cluster, jpm, permutations)
        spectra["hc"] = np.sort(np.concatenate([zero_levels, rest]))
        if len(roots) == 90:
            record["hc_residue_over_width"] = float(
                np.max(np.abs(np.sort(zero_levels[:90]) - roots))
                / (zero_levels[89] - zero_levels[0]))

        # ------------------------------------------------------------ checks
        for name, levels in spectra.items():
            curve = heat_capacity(levels)
            fine = heat_capacity(levels, FINE)
            found = find_peaks(curve)
            found_fine = find_peaks(fine, FINE)
            gauge, spinon, resolved = classify(found, GRID, curve)
            gauge_f, spinon_f, _ = classify(found_fine, FINE, fine)
            record[f"{name}_n_levels"] = int(len(levels))
            released, degeneracy, total = entropy_sum_rule(levels)
            record[f"{name}_entropy_released"] = released
            record[f"{name}_ground_degeneracy"] = degeneracy
            record[f"{name}_entropy_total"] = total
            record[f"{name}_entropy_error"] = total - float(np.log(2.0))
            record[f"{name}_peaks"] = [(round(t, 7), round(h, 4))
                                       for t, h in found]
            record[f"{name}_gauge"] = gauge[0] if gauge else None
            record[f"{name}_spinon"] = spinon[0] if spinon else None
            record[f"{name}_resolved"] = resolved
            record[f"{name}_grid_shift"] = (
                abs(gauge_f[0] / gauge[0] - 1.0)
                if (gauge and gauge_f) else None)
            record[f"{name}_min_C"] = float(curve.min())

        record["runtime"] = time.perf_counter() - started
        tag = f"{jpm:+.3f}".replace(".", "p")
        with open(OUTPUT / f"compare_{tag}.json", "w") as handle:
            json.dump(record, handle, indent=1, default=float)
        np.savez_compressed(
            OUTPUT / f"compare_{tag}.npz", jpm=jpm, grid=GRID,
            **{f"{k}_curve": heat_capacity(v) for k, v in spectra.items()},
            **{f"{k}_levels": v for k, v in spectra.items()})

        line = " | ".join(
            f"{name}: {record[f'{name}_gauge']:.5f}"
            if record.get(f"{name}_gauge") else f"{name}: -"
            for name in ("periodic", "polar", "feshbach", "hc")
            if f"{name}_gauge" in record)
        print(f"jpm={jpm:+.3f} {line} | roots={record['n_roots']} "
              f"S/N={record['periodic_entropy_total']:.5f} "
              f"({record['runtime']:.0f}s)", flush=True)


def full_spectrum_other(cluster, jpm, permutations):
    """Every magnetization sector except S^z_tot = 0."""
    half = cluster.n_sites // 2
    levels = []
    for n_up in range(cluster.n_sites + 1):
        if n_up == half:
            continue
        states = fixed_magnetization_basis(cluster.n_sites, n_up)
        hamiltonian = microscopic_character_hamiltonian(
            cluster, states, jpm, np.zeros(3))
        if len(states) < 8:
            levels.append(eigvalsh(np.asarray(hamiltonian.todense()).real))
            continue
        for basis in translation_sector_bases(states, permutations).values():
            block = (basis.conj().T @ (hamiltonian @ basis)).toarray()
            block = 0.5 * (block + block.conj().T)
            levels.append(eigvalsh(np.ascontiguousarray(block.real)))
    return np.concatenate(levels)


if __name__ == "__main__":
    main()
