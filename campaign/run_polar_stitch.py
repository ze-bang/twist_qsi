#!/usr/bin/env python3
"""The original construction, stitched: polar pullback + mask, spliced into C(T).

The question this answers: does the frame-based winding-free band of
Sec. 5 -- the Loewdin/polar pullback of the exact band, masked -- give the same
two-anomaly heat capacity as the counterterm Hamiltonian, without the
anchoring residue and therefore without the spurious low-temperature peak?

It should, wherever the frame exists, because the masked pullback is a
Hermitian operator with exact transport-sector block structure and no energy
anchoring.  Its limitation is the frame itself: the band must be spectrally
isolated and the projected Gram matrix nonsingular.

Per coupling we compare three winding-free spectra spliced into the same full
spectrum, and the raw one:

  * polar    -- masked polar pullback of the exact band (the original method)
  * feshbach -- masked Feshbach roots (frame-free, exact where they exist)
  * H+C      -- the counterterm Hamiltonian's own lowest levels

Usage: run_polar_stitch.py <jpm> [<jpm> ...]
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.linalg import eigvalsh
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
from qsi_campaign.protocol import (  # noqa: E402
    polarization_sector_labels,
    sector_project,
    full_hilbert_counterterm_spectrum,
)

OUTPUT = ROOT / "campaign" / "outputs" / "polar_stitch"
OUTPUT.mkdir(parents=True, exist_ok=True)
HEAT = ROOT / "campaign" / "outputs" / "specific_heat"
GRID = np.geomspace(1e-6, 3.0, 1600)


def heat_capacity(levels, n_sites=16):
    energies = np.sort(np.asarray(levels, dtype=float))
    energies = energies - energies[0]
    beta = 1.0 / GRID[:, None]
    weight = np.exp(-beta * energies[None, :])
    partition = weight.sum(axis=1)
    mean = (weight * energies[None, :]).sum(axis=1) / partition
    second = (weight * energies[None, :] ** 2).sum(axis=1) / partition
    return (second - mean * mean) / (n_sites * GRID**2)


def peaks(curve, floor=1e-4):
    return [(float(GRID[i]), float(curve[i]))
            for i in argrelmax(curve, order=12)[0] if curve[i] > floor]


def main() -> None:
    couplings = [float(v) for v in sys.argv[1:]]
    cluster = geometry.build_cluster("cubic", (1, 1, 1))
    states = fixed_magnetization_basis(cluster.n_sites, cluster.n_sites // 2)
    index = {int(s): i for i, s in enumerate(states)}
    ice_rows = np.asarray([index[int(s)] for s in cluster.ice_states])
    sectors = translation_sector_bases(
        states, cubic16_translation_permutations(cluster))
    labels = polarization_sector_labels(cluster)

    for jpm in couplings:
        started = time.perf_counter()
        hamiltonian = microscopic_character_hamiltonian(
            cluster, states, jpm, np.zeros(3))
        record = {"jpm": jpm, "g6": 12.0 * abs(jpm) ** 3}

        # ---- the original construction: polar pullback of the exact band
        try:
            band = extract_exact_band(hamiltonian, sectors, ice_rows, 90)
            masked = sector_project(band.operator, labels)
            polar_levels = np.linalg.eigvalsh(masked).real
            record["frame_ok"] = True
            record["gram_min"] = float(
                band.diagnostics.get("model_overlap_min", np.nan))
            record["gap_above_band"] = float(band.fixed_sz_gap_above_band)
        except Exception as error:                       # frame has failed
            polar_levels = None
            record["frame_ok"] = False
            record["frame_error"] = str(error)[:120]

        tag = f"{jpm:+.3f}".replace(".", "p")
        full = np.sort(np.load(HEAT / f"heat_{tag}.npz")["raw_levels"])
        wf_full = np.sort(np.load(HEAT / f"heat_{tag}.npz")["wf_levels"])

        curves = {"raw": heat_capacity(full), "hc": heat_capacity(wf_full)}
        if polar_levels is not None:
            curves["polar"] = heat_capacity(
                full_hilbert_counterterm_spectrum(full, polar_levels))
        roots_path = (ROOT / "campaign" / "outputs" / "exact_counterterm"
                      / f"exact_ct_{tag}.npz")
        if roots_path.exists():
            roots = np.sort(np.load(roots_path)["roots"])
            curves["feshbach"] = heat_capacity(
                full_hilbert_counterterm_spectrum(full, roots))
            record["n_feshbach_roots"] = int(len(roots))

        for name, curve in curves.items():
            found = peaks(curve)
            record[f"{name}_peaks"] = [(round(t, 6), round(h, 4))
                                       for t, h in found]
            record[f"{name}_n_peaks"] = len(found)
        record["runtime"] = time.perf_counter() - started

        with open(OUTPUT / f"stitch_{tag}.json", "w") as handle:
            json.dump(record, handle, indent=1, default=float)
        np.savez_compressed(OUTPUT / f"stitch_{tag}.npz", jpm=jpm, grid=GRID,
                            **{f"{k}_curve": v for k, v in curves.items()},
                            polar_levels=polar_levels
                            if polar_levels is not None else np.zeros(0))
        g6 = record["g6"]
        summary = []
        for name in ("raw", "polar", "feshbach", "hc"):
            pk = record.get(f"{name}_peaks")
            if pk is None:
                summary.append(f"{name}: -")
            else:
                ring = sorted(pk)[0][0] if len(pk) < 3 else sorted(pk)[1][0]
                summary.append(f"{name}: {len(pk)} peaks, ring={ring/g6:.3f} g6")
        print(f"jpm={jpm:+.3f} frame={record['frame_ok']} | "
              + " | ".join(summary) + f" ({record['runtime']:.0f}s)",
              flush=True)


if __name__ == "__main__":
    main()
