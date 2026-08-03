#!/usr/bin/env python3
"""Specific heat versus temperature across the coupling range, both peaks tracked.

For each coupling this computes the COMPLETE spectrum of the 16-site cluster --
every magnetization sector, each one block-diagonalized by the Klein group of
translations -- and from it the heat capacity per site

    C(T)/N = ( <E^2> - <E>^2 ) / (N T^2) .

Two Hamiltonians are treated: the raw periodic cluster, and the winding-free
Hamiltonian H + C of Proposition 1 (the in-block counterterm, which lives
entirely in the S^z_tot = 0 ice manifold, so every other magnetization sector
is shared and is diagonalized only once).

Both anomalies are located by a robust search for local maxima on a
logarithmic temperature grid: the lowest-temperature maximum is the gauge
(ring-exchange) anomaly, the highest-temperature one the spinon anomaly.  The
number of maxima found is recorded, so that a merged single-peak curve is
visible as such rather than being silently reported as a peak position.

Usage: run_specific_heat_scan.py <jpm> [<jpm> ...]
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
    fixed_magnetization_basis,
    microscopic_character_hamiltonian,
    translation_sector_bases,
    cubic16_translation_permutations,
)
from qsi_campaign.protocol import polarization_sector_labels  # noqa: E402
from run_winding_residual import feshbach_matrix  # noqa: E402
from run_exact_counterterm import embed  # noqa: E402

OUTPUT = ROOT / "campaign" / "outputs" / "specific_heat"
OUTPUT.mkdir(parents=True, exist_ok=True)

GRID = np.geomspace(1e-6, 3.0, 1600)


def sector_spectrum(hamiltonian, bases) -> np.ndarray:
    """Eigenvalues of a Hamiltonian, gathered over symmetry blocks."""
    levels = []
    for basis in bases:
        block = (basis.conj().T @ (hamiltonian @ basis)).toarray()
        block = 0.5 * (block + block.conj().T)
        if np.max(np.abs(block.imag), initial=0.0) < 1e-12:
            block = np.ascontiguousarray(block.real)
        levels.append(eigvalsh(block))
    return np.concatenate(levels)


def heat_capacity(levels: np.ndarray, n_sites: int) -> np.ndarray:
    shifted = np.asarray(levels) - float(np.min(levels))
    beta = 1.0 / GRID[:, None]
    # stable Boltzmann weights
    weight = np.exp(-beta * shifted[None, :])
    partition = weight.sum(axis=1)
    mean = (weight * shifted[None, :]).sum(axis=1) / partition
    second = (weight * shifted[None, :] ** 2).sum(axis=1) / partition
    return (second - mean * mean) / (n_sites * GRID**2)


def find_peaks(curve: np.ndarray, order: int = 12, floor: float = 1e-4):
    """All local maxima of C(T) on the log grid, tallest first."""
    indices = argrelmax(curve, order=order)[0]
    indices = [int(i) for i in indices if curve[i] > floor]
    peaks = [(float(GRID[i]), float(curve[i])) for i in indices]
    return sorted(peaks, key=lambda p: p[0])


def main() -> None:
    couplings = [float(v) for v in sys.argv[1:]]
    cluster = geometry.build_cluster("cubic", (1, 1, 1))
    n_sites = cluster.n_sites
    permutations = cubic16_translation_permutations(cluster)

    # basis and symmetry blocks for every magnetization sector, built once
    sectors = {}
    for n_up in range(n_sites + 1):
        states = fixed_magnetization_basis(n_sites, n_up)
        if len(states) < 8:
            sectors[n_up] = (states, None)
        else:
            bases = list(translation_sector_bases(states, permutations).values())
            sectors[n_up] = (states, bases)
    half = n_sites // 2
    zero_states = sectors[half][0]
    index = {int(s): i for i, s in enumerate(zero_states)}
    ice_rows = np.asarray([index[int(s)] for s in cluster.ice_states])
    outside = np.ones(len(zero_states), dtype=bool)
    outside[ice_rows] = False
    complement = np.where(outside)[0]
    labels = polarization_sector_labels(cluster)
    cross = labels[:, None] != labels[None, :]

    for jpm in couplings:
        started = time.perf_counter()
        raw_levels, other_levels = [], []
        for n_up, (states, bases) in sectors.items():
            hamiltonian = microscopic_character_hamiltonian(
                cluster, states, jpm, np.zeros(3))
            if bases is None:
                dense = np.asarray(hamiltonian.todense()).real
                values = eigvalsh(dense)
            else:
                values = sector_spectrum(hamiltonian, bases)
            raw_levels.append(values)
            if n_up != half:
                other_levels.append(values)
        raw_levels = np.concatenate(raw_levels)

        # winding-free: only the S^z = 0 block changes
        bare = microscopic_character_hamiltonian(
            cluster, zero_states, jpm, np.zeros(3)).tocsc().real
        anchor = float(np.min(sector_spectrum(bare, sectors[half][1])))
        for _ in range(3):
            f_matrix = feshbach_matrix(bare, ice_rows, complement, anchor)
            corrected = (bare + embed(-np.where(cross, f_matrix, 0.0),
                                      ice_rows, len(zero_states))).tocsc()
            zero_levels = sector_spectrum(corrected, sectors[half][1])
            new_anchor = float(np.min(zero_levels))
            if abs(new_anchor - anchor) < 1e-10:
                anchor = new_anchor
                break
            anchor = new_anchor
        wf_levels = np.concatenate(other_levels + [zero_levels])

        raw_curve = heat_capacity(raw_levels, n_sites)
        wf_curve = heat_capacity(wf_levels, n_sites)
        raw_peaks = find_peaks(raw_curve)
        wf_peaks = find_peaks(wf_curve)

        g4 = 4.0 * jpm * jpm
        g6 = 12.0 * abs(jpm) ** 3
        record = {
            "jpm": jpm, "g4": g4, "g6": g6,
            "n_levels": int(len(raw_levels)),
            "raw_peaks": raw_peaks, "wf_peaks": wf_peaks,
            "raw_n_peaks": len(raw_peaks), "wf_n_peaks": len(wf_peaks),
            "raw_low": raw_peaks[0][0] if raw_peaks else None,
            "raw_high": raw_peaks[-1][0] if raw_peaks else None,
            "wf_low": wf_peaks[0][0] if wf_peaks else None,
            "wf_high": wf_peaks[-1][0] if wf_peaks else None,
            "runtime": time.perf_counter() - started,
        }
        for key, scale in (("g4", g4), ("g6", g6)):
            for tag in ("raw_low", "wf_low"):
                value = record[tag]
                record[f"{tag}_over_{key}"] = value / scale if value else None

        tag = f"{jpm:+.3f}".replace(".", "p")
        with open(OUTPUT / f"heat_{tag}.json", "w") as handle:
            json.dump(record, handle, indent=1, default=float)
        np.savez_compressed(OUTPUT / f"heat_{tag}.npz", jpm=jpm, grid=GRID,
                            raw_curve=raw_curve, wf_curve=wf_curve,
                            raw_levels=raw_levels, wf_levels=wf_levels)
        print(f"jpm={jpm:+.3f} levels={len(raw_levels)} "
              f"raw peaks={[round(p[0], 5) for p in raw_peaks]} "
              f"wf peaks={[round(p[0], 5) for p in wf_peaks]} | "
              f"raw_low/g4={record['raw_low_over_g4']:.3f} "
              f"wf_low/g6={record['wf_low_over_g6'] or float('nan'):.3f} "
              f"({record['runtime']:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
