#!/usr/bin/env python3
"""Label every maximum of C(T) by the levels that produce it.

C(T)/N = (1/N T^2) sum_n p_n (E_n - <E>)^2, so each level carries a definite
share of the anomaly at any temperature.  Evaluating that share at a peak says
which states are responsible, and the answer is unambiguous in practice:

  sector    the weight sits almost entirely in the few lowest levels, which
            are the near-degenerate ground states of different transport
            sectors: a topological-sector Schottky anomaly, not a gauge
            excitation;
  ring      the weight is spread through the 90-state gauge band: the
            ring-exchange (photon) anomaly, the one that should track g_6;
  spinon    the weight comes from levels above the band: breaking ice rules.

A peak is additionally verified by surgery -- collapsing the levels blamed for
it and checking that it disappears while the others do not move.

Usage: analyse_peak_labels.py [--table] [--verify]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.signal import argrelmax

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "campaign" / "outputs" / "method_comparison"
N_SITES = 16
N_BAND = 90
GRID = np.geomspace(1e-9, 4.0, 4200)   # low enough to resolve the
                                       # sector splitting of H+C
METHODS = ("periodic", "polar", "feshbach", "hc")

# Peaks separate cleanly by the number of thermally active levels, exp(S):
# measured across the whole sweep, sector anomalies have 3-7, ring anomalies
# 19-26, spinon anomalies several hundred.  The gap between the first two
# groups is wide, so the dividing line is not delicate; it is quoted here
# rather than hidden, and ``active_levels`` is reported for every peak so the
# assignment can be checked.
SECTOR_MAX = 12.0


def heat_capacity(levels, grid=GRID, chunk=200, cutoff=60.0):
    """C(T)/N, evaluated in grid chunks with an energy cutoff.

    At temperature T a level lying more than ``cutoff`` * T above the ground
    state contributes nothing at double precision, so each chunk only needs the
    levels below that; without this the full 65536-level array on a fine grid
    is several gigabytes.
    """
    energies = np.sort(np.asarray(levels, float))
    energies = energies - energies[0]
    out = np.empty(len(grid))
    for start in range(0, len(grid), chunk):
        block = grid[start:start + chunk]
        keep = int(np.searchsorted(energies, cutoff * block[-1]) + 1)
        local = energies[:keep]
        beta = 1.0 / block[:, None]
        weight = np.exp(-beta * local[None, :])
        partition = weight.sum(axis=1)
        mean = (weight * local[None, :]).sum(axis=1) / partition
        second = (weight * local[None, :] ** 2).sum(axis=1) / partition
        out[start:start + chunk] = (second - mean * mean) / (N_SITES * block**2)
    return out


def level_shares(levels, temperature):
    """Share of the heat-capacity variance carried by each level at T."""
    energies = np.sort(np.asarray(levels, float))
    energies = energies - energies[0]
    weight = np.exp(-energies / temperature)
    weight /= weight.sum()
    mean = float(weight @ energies)
    share = weight * (energies - mean) ** 2
    total = share.sum()
    return share / total if total > 0 else share


def effective_levels(levels, temperature):
    """exp(S) at this temperature: how many levels are thermally active.

    This is the discriminator, and it needs no hand-tuned threshold.  A
    Schottky anomaly from the near-degenerate ground states of a handful of
    transport sectors has only those few levels active; the ring anomaly of
    the gauge band has tens; the spinon anomaly has thousands.

    Two earlier attempts are worth recording as failures.  Detecting the
    sector multiplet from gaps in the spectrum breaks on H + C, whose
    anchoring residue splits that multiplet so there is no gap to find.
    Collapsing the k lowest levels and asking whether the peak survives is
    constructive but too permissive: with only 90 levels in the band,
    collapsing 34 of them destroys the ring anomaly as well, and every peak
    is then attributed to the sectors.
    """
    energies = np.sort(np.asarray(levels, float))
    energies = energies - energies[0]
    weight = np.exp(-energies / temperature)
    weight /= weight.sum()
    keep = weight[weight > 1e-300]
    return float(np.exp(-np.sum(keep * np.log(keep))))


def removed_by_collapse(levels, temperature, grid, kmax=10):
    """Does flattening the few lowest levels remove this maximum?

    Only the transport-sector ground states are collapsed -- at most ``kmax``,
    of order the number of sectors that lie low on this cluster (seven).  The
    ring anomaly needs about twenty levels, so it survives; the sector anomaly
    does not.  Allowing k up to a third of the band, as an earlier version did,
    destroys the ring anomaly too and attributes every peak to the sectors.
    """
    for k in range(2, kmax + 1):
        surgery = np.array(levels, dtype=float)
        surgery[:k] = surgery[0]
        after = heat_capacity(surgery, grid)
        survivors = [float(grid[i]) for i in argrelmax(after, order=16)[0]
                     if after[i] > 1e-4]
        if not any(abs(np.log(t / temperature)) < 0.15 for t in survivors):
            return k
    return None


def label_peaks(levels, curve=None, grid=GRID, has_sectors=True):
    """Every maximum, labelled by the levels that produce it.

    ``has_sectors`` must be False for the raw periodic spectrum: that
    Hamiltonian does not conserve transport -- 99.97% of the off-diagonal
    weight of its ice block changes the transport label -- so it has no sector
    structure, and its low-temperature anomaly is the gauge anomaly, artifact
    contaminated but not a sector Schottky feature.
    """
    levels = np.sort(np.asarray(levels, float))
    curve = heat_capacity(levels, grid)
    peaks = [(float(grid[i]), float(curve[i]))
             for i in argrelmax(curve, order=16)[0] if curve[i] > 1e-4]
    labelled = []
    for temperature, height in peaks:
        share = level_shares(levels, temperature)
        f_band = float(share[:N_BAND].sum())
        f_above = float(share[N_BAND:].sum())
        active = effective_levels(levels, temperature)
        collapse = None
        if f_above > 0.5:
            name = "spinon"
        elif not has_sectors:
            name = "gauge"
        elif active < SECTOR_MAX:
            name = "sector"
        else:
            name = "ring"
        edge = (temperature < 3.0 * grid[0]) or (temperature > grid[-1] / 3.0)
        labelled.append({
            "T": temperature, "height": height, "label": name,
            "at_grid_edge": bool(edge), "active_levels": active,
            "collapse_k": collapse, "f_band": f_band, "f_above": f_above,
        })
    return labelled


def verify(levels, labelled, grid=GRID):
    """Surgery check: collapse the blamed levels, confirm the peak vanishes."""
    levels = np.sort(np.asarray(levels, float))
    checks = []
    for entry in labelled:
        if entry["label"] != "sector":
            continue
        surgery = levels.copy()
        surgery[:entry["collapse_k"]] = surgery[0]
        after = heat_capacity(surgery, grid)
        remaining = [float(grid[i]) for i in argrelmax(after, order=16)[0]
                     if after[i] > 1e-4]
        survived = any(abs(np.log(t / entry["T"])) < 0.15 for t in remaining)
        checks.append({"T": entry["T"], "survives_surgery": survived})
    return checks


def main() -> None:
    records = []
    for path in sorted(OUT.glob("compare_*.npz")):
        data = np.load(path)
        jpm = float(data["jpm"])
        with open(str(path).replace(".npz", ".json")) as handle:
            meta = json.load(handle)
        entry = {"jpm": jpm, "g4": 4 * jpm * jpm, "g6": 12 * abs(jpm) ** 3,
                 "meta": meta}
        for method in METHODS:
            key = f"{method}_levels"
            if key not in data:
                continue
            labelled = label_peaks(data[key],
                                   has_sectors=(method != "periodic"))
            entry[method] = labelled
            if "--verify" in sys.argv:
                entry[f"{method}_surgery"] = verify(data[key], labelled)
        records.append(entry)
    records.sort(key=lambda r: r["jpm"])

    print("every maximum of C(T), labelled by the levels that produce it")
    print(f"{'jpm':>7} {'method':>9} {'T_peak':>10} {'/g4':>8} {'/g6':>9} "
          f"{'label':>7} {'n_active':>8} {'f_band':>7} {'f_abv':>6}")
    for record in records:
        for method in METHODS:
            for peak in record.get(method, []):
                print(f"{record['jpm']:>7.3f} {method:>9} {peak['T']:>10.3e} "
                      f"{peak['T']/record['g4']:>8.4f} "
                      f"{peak['T']/record['g6']:>9.4f} "
                      f"{peak['label']:>7} {peak['active_levels']:>8.1f} "
                      f"{peak['f_band']:>7.2f} {peak['f_above']:>6.2f}"
                      + ("  (grid edge)" if peak["at_grid_edge"] else ""))
        print()

    with open(OUT / "peak_labels.json", "w") as handle:
        json.dump(records, handle, indent=1, default=float)
    print("written to", OUT / "peak_labels.json")


if __name__ == "__main__":
    main()
