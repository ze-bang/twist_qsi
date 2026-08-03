#!/usr/bin/env python3
"""Grid-free peak positions, and the band-level deviation, for the paper table.

run_counterterm_spine.py locates maxima on a 3000-point logarithmic grid, whose
spacing is 0.35% per step.  That is coarser than the agreement between the
counterterm and the exact masked reference, so a deviation reported from it is
grid-limited and reads as "0.00%" whenever the two land on the same grid point.
This script removes the grid from the comparison in two ways:

  * the ring-anomaly position is refined by fitting a parabola in log T to the
    three points around each maximum, on a grid four times finer, which brings
    the positional resolution to ~1e-4 relative;
  * the band levels themselves are compared directly, which involves no
    temperature grid at all and is the sharper statement.

Usage: refine_peaks.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.signal import argrelmax

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "campaign"))
SPINE = ROOT / "campaign" / "outputs" / "counterterm_spine"

N_SITES = 16
N_ICE = 90
GRID = np.geomspace(1e-7, 4.0, 12000)
SECTOR_MAX = 12.0


def heat_capacity(levels, grid=GRID, chunk=512):
    energies = np.sort(np.asarray(levels, float))
    energies = energies - energies[0]
    out = np.empty(len(grid))
    for start in range(0, len(grid), chunk):
        temperature = grid[start:start + chunk]
        weight = np.exp(-energies[None, :] / temperature[:, None])
        partition = weight.sum(1)
        mean = (weight * energies[None, :]).sum(1) / partition
        second = (weight * energies[None, :] ** 2).sum(1) / partition
        out[start:start + chunk] = ((second - mean * mean)
                                    / (N_SITES * temperature ** 2))
    return out


def effective_levels(levels, temperature):
    energies = np.sort(np.asarray(levels, float))
    energies = energies - energies[0]
    weight = np.exp(-energies / temperature)
    weight = weight / weight.sum()
    positive = weight[weight > 0]
    return float(np.exp(-np.sum(positive * np.log(positive))))


def refined_maxima(levels, grid=GRID, floor=1e-4):
    """Maxima with the position refined by a parabola in log T."""
    curve = heat_capacity(levels, grid)
    logt = np.log(grid)
    found = []
    for index in argrelmax(curve, order=48)[0]:
        if curve[index] <= floor or index == 0 or index == len(curve) - 1:
            continue
        left, middle, right = curve[index - 1], curve[index], curve[index + 1]
        denominator = left - 2.0 * middle + right
        shift = (0.5 * (left - right) / denominator
                 if abs(denominator) > 0 else 0.0)
        shift = float(np.clip(shift, -0.5, 0.5))
        step = logt[index + 1] - logt[index]
        temperature = float(np.exp(logt[index] + shift * step))
        found.append({"T": temperature, "height": float(middle),
                      "grid_T": float(grid[index]),
                      "active": effective_levels(levels, temperature)})
    return found


def level_shares(levels, temperature):
    """Share of the heat-capacity variance carried by each level at T.

    This must match analyse_peak_labels.level_shares: the classifier is
    validated against that definition, and substituting plain Boltzmann
    probabilities (which is easy to do by accident) mislabels the spinon
    anomaly as a ring anomaly at weak coupling, because the ninety band levels
    hold most of the *probability* there while carrying almost none of the
    *variance*.
    """
    energies = np.sort(np.asarray(levels, float))
    energies = energies - energies[0]
    weight = np.exp(-energies / temperature)
    weight /= weight.sum()
    mean = float(weight @ energies)
    share = weight * (energies - mean) ** 2
    total = share.sum()
    return share / total if total > 0 else share


def classify(levels, peaks, has_sectors=True):
    for peak in peaks:
        above = float(level_shares(levels, peak["T"])[N_ICE:].sum())
        if above > 0.5:
            peak["label"] = "spinon"
        elif not has_sectors:
            peak["label"] = "gauge"
        elif peak["active"] < SECTOR_MAX:
            peak["label"] = "sector"
        else:
            peak["label"] = "ring"
    return peaks


def named(peaks, label):
    return next((p["T"] for p in peaks if p["label"] == label), None)


def main() -> None:
    records = []
    print(f"grid: {len(GRID)} points, spacing "
          f"{(GRID[1] / GRID[0] - 1) * 100:.3f}% per step, "
          f"parabolic refinement in log T")
    print(f"{'jpm':>7} {'T_ring':>11} {'reference':>11} {'dev':>8} "
          f"{'band dev':>9} {'active':>7} {'T_spinon':>10} {'T_gauge(per)':>13}")
    for path in sorted(SPINE.glob("spine_*.npz"),
                       key=lambda p: -float(np.load(p)["jpm"])):
        data = np.load(path)
        jpm = float(data["jpm"])
        band = np.asarray(data["band"])
        upper = np.asarray(data["levels"])
        upper = upper[upper > band[-1]]
        reference_band = np.asarray(data["reference_band"])

        counterterm = classify(np.asarray(data["levels"]),
                              refined_maxima(np.asarray(data["levels"])))
        record = {"jpm": jpm, "g6": 12.0 * abs(jpm) ** 3,
                  "counterterm_ring": named(counterterm, "ring"),
                  "counterterm_spinon": named(counterterm, "spinon"),
                  "counterterm_sector": named(counterterm, "sector"),
                  "counterterm_n_peaks": len(counterterm),
                  "counterterm_active": next(
                      (p["active"] for p in counterterm
                       if p["label"] == "ring"), None)}

        if reference_band.size == N_ICE:
            reference_levels = np.sort(np.concatenate([reference_band, upper]))
            reference = classify(reference_levels,
                                 refined_maxima(reference_levels))
            record["reference_ring"] = named(reference, "ring")
            record["band_deviation"] = float(
                np.max(np.abs(band - reference_band))
                / (band[-1] - band[0]))
            if record["reference_ring"] and record["counterterm_ring"]:
                record["ring_deviation"] = abs(
                    record["counterterm_ring"] / record["reference_ring"] - 1.0)

        periodic = np.load(ROOT / "campaign" / "outputs" / "method_comparison"
                           / f"compare_{format(jpm, '+.3f').replace('.', 'p')}"
                             ".npz")["periodic_levels"]
        periodic_peaks = classify(periodic, refined_maxima(periodic),
                                  has_sectors=False)
        record["periodic_gauge"] = named(periodic_peaks, "gauge")
        record["periodic_spinon"] = named(periodic_peaks, "spinon")
        records.append(record)

        show = lambda v, f="11.6f": ("---".rjust(11) if v is None
                                     else format(v, f))
        print(f"{jpm:>7.3f} {show(record['counterterm_ring'])} "
              f"{show(record.get('reference_ring'))} "
              f"{('---' if record.get('ring_deviation') is None else format(record['ring_deviation'] * 100, '.3f') + '%'):>8} "
              f"{('---' if record.get('band_deviation') is None else format(record['band_deviation'] * 100, '.2f') + '%'):>9} "
              f"{(record['counterterm_active'] or float('nan')):>7.1f} "
              f"{show(record['counterterm_spinon'], '10.5f')} "
              f"{show(record['periodic_gauge'], '13.6f')}", flush=True)

    records.sort(key=lambda r: r["jpm"])
    with open(SPINE / "refined_peaks.json", "w") as handle:
        json.dump(records, handle, indent=1, default=float)

    couplings = np.array([r["jpm"] for r in records])
    ring = np.array([r["counterterm_ring"] or np.nan for r in records])
    gauge = np.array([r["periodic_gauge"] or np.nan for r in records])
    weak = np.abs(couplings) <= 0.031

    def slope(y):
        good = weak & np.isfinite(y)
        return float(np.polyfit(np.log(np.abs(couplings[good])),
                               np.log(y[good]), 1)[0])

    print(f"\npower-law fits over |jpm| <= 0.031 ({int(weak.sum())} points):")
    print(f"  counterterm  n = {slope(ring):.3f}   (ideal 3 for g6)")
    print(f"  periodic     n = {slope(gauge):.3f}   (ideal 2 for g4)")
    deviations = [r["ring_deviation"] for r in records
                  if r.get("ring_deviation") is not None]
    bands = [r["band_deviation"] for r in records
             if r.get("band_deviation") is not None]
    print(f"ring deviation from the reference: max {max(deviations)*100:.3f}% "
          f"over {len(deviations)} couplings with a complete reference")
    print(f"band-level deviation: max {max(bands)*100:.2f}% of the band width")
    print(f"sector anomalies found: "
          f"{sum(1 for r in records if r['counterterm_sector'])}")
    print(f"wrote {SPINE / 'refined_peaks.json'}")


if __name__ == "__main__":
    main()
