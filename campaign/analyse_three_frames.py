#!/usr/bin/env python3
"""Both heat-capacity peaks for the three winding-free architectures.

For every coupling: splice each corrected band (ice frame, dressed frame,
clean/adaptive frame) into the same fixed-Sz full spectrum via the trace
matched replacement, compute C(T), and record the ring (lower) and spinon
(upper) peak positions, together with the gauge-coherence rank of the pool.
The periodic (uncorrected) spectrum is carried as the baseline.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DRESSED = ROOT / "campaign" / "outputs" / "dressed_sweep"
CLEAN = ROOT / "campaign" / "outputs" / "clean_sweep"
OUTPUT = ROOT / "campaign" / "outputs"

TEMPERATURES = np.geomspace(1e-5, 3.0, 3200)


def heat_capacity(levels):
    shifted = np.asarray(levels) - np.min(levels)
    beta = 1.0 / TEMPERATURES[:, None]
    w = np.exp(-beta * shifted[None, :])
    z = w.sum(axis=1)
    mean = (w * shifted[None, :]).sum(axis=1) / z
    second = (w * shifted[None, :] ** 2).sum(axis=1) / z
    return (second - mean * mean) / TEMPERATURES**2


def two_peaks(curve):
    interior = (curve[1:-1] > curve[:-2]) & (curve[1:-1] > curve[2:])
    maxima = np.where(interior)[0] + 1
    if len(maxima) == 0:
        return None, None
    low = [m for m in maxima if TEMPERATURES[m] < 0.06]
    high = [m for m in maxima if TEMPERATURES[m] >= 0.06]
    t_low = float(TEMPERATURES[max(low, key=lambda m: curve[m])]) if low else None
    t_high = float(TEMPERATURES[max(high, key=lambda m: curve[m])]) if high else None
    return t_low, t_high


def spliced(full, raw, corrected):
    corrected = np.asarray(corrected) + (np.mean(raw) - np.mean(corrected))
    return np.concatenate([np.sort(full)[len(raw):], corrected])


def main() -> None:
    rows = []
    for path in sorted(DRESSED.glob("dressed_*.npz")):
        d = np.load(path)
        jpm = float(d["jpm"])
        tag = f"{jpm:+.3f}".replace(".", "p")
        clean_path = CLEAN / f"clean_{tag}.npz"
        if not clean_path.exists():
            continue
        c = np.load(clean_path)
        full = np.asarray(d["full_spectrum"])
        raw = np.asarray(d["energies"])

        row = {"jpm": jpm, "g6": 12.0 * abs(jpm) ** 3,
               "rank_clean": int(c["rank"]),
               "coherence_rank_0p7": int((c["coherence_spectrum"] > 0.7).sum()),
               "coherence_rank_0p5": int((c["coherence_spectrum"] > 0.5).sum()),
               "coherence_rank_0p85": int((c["coherence_spectrum"] > 0.85).sum())}

        curves = {"periodic": heat_capacity(full)}
        curves["dressed"] = heat_capacity(spliced(full, raw, d["dressed_band"]))
        if "ice_band" in d.files:
            curves["ice"] = heat_capacity(spliced(full, raw, d["ice_band"]))
        curves["clean"] = heat_capacity(
            spliced(full, np.asarray(c["raw_selected"]), c["clean_band"]))
        union_path = ROOT / "campaign" / "outputs" / "unionframe_sweep" / \
            f"unionframe_{tag}.npz"
        if union_path.exists():
            u = np.load(union_path)
            curves["union"] = heat_capacity(
                spliced(full, np.asarray(u["raw_selected"]), u["union_band"]))
            row["rank_union"] = int(u["rank"])
        for name, curve in curves.items():
            t_low, t_high = two_peaks(curve)
            row[f"{name}_low"] = t_low
            row[f"{name}_high"] = t_high
        rows.append(row)
        print(f"jpm={jpm:+.3f} rank={row['rank_clean']:3d} "
              f"coh0.7={row['coherence_rank_0p7']:3d} "
              + " ".join(f"{k}={row.get(k+'_low') and row[k+'_low']:.5g}"
                         if row.get(k + "_low") else f"{k}=--"
                         for k in ("periodic", "ice", "dressed", "clean")),
              flush=True)

    (OUTPUT / "three_frames.json").write_text(
        json.dumps({"points": rows}, indent=2, default=float) + "\n")
    print(f"wrote {OUTPUT / 'three_frames.json'} with {len(rows)} points")


if __name__ == "__main__":
    main()
