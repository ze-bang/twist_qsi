#!/usr/bin/env python3
"""Complete winding-free thermodynamics from H + C, with no splice.

Because the in-block counterterm makes the winding-free theory a genuine
Hermitian operator, the whole spectrum -- gauge band and spinon sector alike
-- comes from one diagonalization, and the heat capacity needs no stitching
of a projected band onto a microscopic complement.  This script computes the
full fixed-magnetization spectrum of H and of H + C and writes both heat
capacity curves together with the positions of the two anomalies.

The ensemble is the S^z_tot = 0 sector, as everywhere else in this campaign;
the spinon anomaly sits higher in this ensemble than in the full one.

Usage: run_wf_thermodynamics.py <jpm> [<jpm> ...]
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.linalg import eigh, eigvalsh

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

OUTPUT = ROOT / "campaign" / "outputs" / "wf_thermodynamics"
OUTPUT.mkdir(parents=True, exist_ok=True)


def heat_capacity(levels, grid):
    shifted = np.asarray(levels) - float(np.min(levels))
    beta = 1.0 / grid[:, None]
    weight = np.exp(-beta * shifted[None, :])
    partition = weight.sum(axis=1)
    mean = (weight * shifted[None, :]).sum(axis=1) / partition
    second = (weight * shifted[None, :] ** 2).sum(axis=1) / partition
    return (second - mean * mean) / grid**2


def peaks(grid, curve, split=0.05):
    low = grid < split
    high = ~low
    return (float(grid[low][np.argmax(curve[low])]) if low.any() else np.nan,
            float(grid[high][np.argmax(curve[high])]) if high.any() else np.nan)


def main() -> None:
    couplings = [float(v) for v in sys.argv[1:]]
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
    grid = np.geomspace(1e-5, 1.5, 900)

    def full_spectrum(hamiltonian):
        return np.sort(np.concatenate([
            eigvalsh(np.ascontiguousarray(
                (0.5 * ((basis.conj().T @ (hamiltonian @ basis)).toarray()
                        + (basis.conj().T
                           @ (hamiltonian @ basis)).toarray().conj().T)).real))
            for basis in sectors.values()]))

    for jpm in couplings:
        started = time.perf_counter()
        bare = microscopic_character_hamiltonian(
            cluster, states, jpm, np.zeros(3)).tocsc().real
        anchor = min(
            float(eigh(np.ascontiguousarray(
                (basis.conj().T @ (bare @ basis)).toarray().real),
                subset_by_index=(0, 0), eigvals_only=True)[0])
            for basis in sectors.values())
        for _ in range(3):
            f_matrix = feshbach_matrix(bare, ice_rows, complement, anchor)
            corrected = (bare + embed(-np.where(cross, f_matrix, 0.0),
                                      ice_rows, len(states))).tocsc()
            new_anchor = min(
                float(eigh(np.ascontiguousarray(
                    (basis.conj().T @ (corrected @ basis)).toarray().real),
                    subset_by_index=(0, 0), eigvals_only=True)[0])
                for basis in sectors.values())
            if abs(new_anchor - anchor) < 1e-10:
                anchor = new_anchor
                break
            anchor = new_anchor

        raw_levels = full_spectrum(bare)
        wf_levels = full_spectrum(corrected)
        raw_curve = heat_capacity(raw_levels, grid)
        wf_curve = heat_capacity(wf_levels, grid)
        raw_low, raw_high = peaks(grid, raw_curve)
        wf_low, wf_high = peaks(grid, wf_curve)

        record = {
            "jpm": jpm, "g6": 12.0 * abs(jpm) ** 3,
            "raw_low_peak": raw_low, "raw_high_peak": raw_high,
            "wf_low_peak": wf_low, "wf_high_peak": wf_high,
            "wf_low_over_g6": wf_low / (12.0 * abs(jpm) ** 3),
            "raw_low_over_g6": raw_low / (12.0 * abs(jpm) ** 3),
            "runtime": time.perf_counter() - started,
        }
        tag = f"{jpm:+.3f}".replace(".", "p")
        with open(OUTPUT / f"thermo_{tag}.json", "w") as handle:
            json.dump(record, handle, indent=1, default=float)
        np.savez_compressed(OUTPUT / f"thermo_{tag}.npz", jpm=jpm, grid=grid,
                            raw_curve=raw_curve, wf_curve=wf_curve,
                            raw_levels=raw_levels, wf_levels=wf_levels)
        print(f"jpm={jpm:+.3f} raw peaks {raw_low:.5f}/{raw_high:.4f} "
              f"wf peaks {wf_low:.5f}/{wf_high:.4f} "
              f"(wf low = {record['wf_low_over_g6']:.3f} g6, "
              f"raw low = {record['raw_low_over_g6']:.3f} g6) "
              f"({record['runtime']:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
