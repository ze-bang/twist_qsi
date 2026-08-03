#!/usr/bin/env python3
"""Sector symmetrisation: put the counterterm's residue back where it came from.

The counterterm Hamiltonian H + C cancels the transporting part of the ice
block at a single energy z_0.  The error that leaves does not spread through
the band: it lands almost entirely on the transport-sector ground multiplet,
which the exact masked Feshbach map shows must be exactly degenerate (six
sector ground states at 0.00000 on cubic-16).  H + C splits them, and that
splitting is the spurious low-temperature anomaly.

Restoring the degeneracy -- replacing the k lowest levels by their mean --
removes the spurious anomaly, restores the residual entropy ln k, and leaves
the ring anomaly where it was, in fact improving it.  This script measures all
of that across the sweep.

Usage: run_sector_symmetrisation.py [--k N]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "campaign"))
OUT = ROOT / "campaign" / "outputs" / "method_comparison"

from analyse_peak_labels import label_peaks, heat_capacity, GRID  # noqa: E402

N_SITES = 16
SGRID = np.geomspace(1e-9, 60.0, 6000)


def entropy(levels):
    energies = np.sort(np.asarray(levels, float))
    energies = energies - energies[0]
    degeneracy = int(np.sum(energies < 1e-12))
    curve = heat_capacity(energies, SGRID)
    released = float(np.trapezoid(curve / SGRID, SGRID))
    return released + np.log(degeneracy) / N_SITES, degeneracy


def main() -> None:
    k = 6
    if "--k" in sys.argv:
        k = int(sys.argv[sys.argv.index("--k") + 1])

    with open(OUT / "peak_labels.json") as handle:
        labels = json.load(handle)
    reference = {round(r["jpm"], 4):
                 [p for p in r.get("feshbach", []) if p["label"] == "ring"]
                 for r in labels}

    records = []
    for path in sorted(OUT.glob("compare_*.npz")):
        data = np.load(path)
        jpm = float(data["jpm"])
        if jpm >= 0 or "hc_levels" not in data:
            continue
        g6 = 12.0 * abs(jpm) ** 3
        raw = np.sort(data["hc_levels"])
        symmetric = raw.copy()
        symmetric[:k] = symmetric[:k].mean()

        def summarise(levels):
            peaks = label_peaks(levels)
            ring = [p for p in peaks if p["label"] == "ring"]
            sector = [p for p in peaks if p["label"] == "sector"]
            spinon = [p for p in peaks if p["label"] == "spinon"]
            return {
                "ring": ring[0]["T"] / g6 if ring else None,
                "n_sector": len(sector),
                "sector_T": sector[0]["T"] if sector else None,
                "spinon": spinon[0]["T"] if spinon else None,
            }

        before, after = summarise(raw), summarise(symmetric)
        entropy_before, g_before = entropy(raw)
        entropy_after, g_after = entropy(symmetric)
        target = reference.get(round(jpm, 4))
        exact = target[0]["T"] / g6 if target else None

        record = {
            "jpm": jpm, "k": k, "g6": g6, "exact_ring": exact,
            "before": before, "after": after,
            "sector_splitting": float(raw[k - 1] - raw[0]),
            "band_width": float(raw[89] - raw[0]),
            "entropy_before": entropy_before, "entropy_after": entropy_after,
            "degeneracy_before": g_before, "degeneracy_after": g_after,
        }
        for tag, value in (("before", before["ring"]), ("after", after["ring"])):
            record[f"dev_{tag}"] = (abs(value / exact - 1.0)
                                    if (exact is not None and value is not None)
                                    else None)
        records.append(record)

    records.sort(key=lambda r: r["jpm"])
    with open(OUT / "sector_symmetrisation.json", "w") as handle:
        json.dump(records, handle, indent=1, default=float)

    print(f"sector symmetrisation, k = {k}")
    print(f"{'jpm':>7} {'exact':>8} {'H+C':>8} {'dev':>6} | {'H+C sym':>8} "
          f"{'dev':>6} | {'3rd pk':>7} {'->':>3} | {'split/width':>11} "
          f"| {'S/N':>7} {'->':>7} {'g0':>3}")
    for r in records:
        b, a = r["before"], r["after"]
        fmt = lambda v, s="8.4f": format(v, s) if v is not None else f"{'-':>8}"
        print(f"{r['jpm']:>7.3f} {fmt(r['exact_ring'])} {fmt(b['ring'])} "
              f"{(r['dev_before']*100 if r['dev_before'] is not None else float('nan')):>5.1f}% | "
              f"{fmt(a['ring'])} "
              f"{(r['dev_after']*100 if r['dev_after'] is not None else float('nan')):>5.1f}% | "
              f"{b['n_sector']:>7d} {a['n_sector']:>3d} | "
              f"{r['sector_splitting']/r['band_width']:>11.3f} | "
              f"{r['entropy_before']:>7.5f} {r['entropy_after']:>7.5f} "
              f"{r['degeneracy_after']:>3d}")
    print(f"\n(ln 2 = {np.log(2):.6f}; the entropy must return to it once the"
          f" frozen ln g0 is added back)")


if __name__ == "__main__":
    main()
