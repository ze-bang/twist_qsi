#!/usr/bin/env python3
"""Lower-peak position of the winding-free XXZ band as a function of Jpm.

Scans Jpm/Jzz from -0.30 (deep pi-flux) through +0.05 (zero-flux, QMC side),
recording for each point whether the winding-free construction is well defined
(isolated ice-descended band, nonsingular ice Gram), and when it is, the
positions of both heat-capacity maxima of the full-Hilbert replacement
spectrum, together with the band gap and minimum ice overlap.

Points reuse the ``solve_point`` cache shared with the Ce2Hf2O7 drivers, so a
stripe of the grid can run in each of several processes:

    python3 campaign/run_jpm_scan.py --stride 3 --offset 0   # etc.

The summary pass (--summary) collects every cached point into one JSON.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "notes"))

import recompute_finite_size_artifact as geometry  # noqa: E402
from qsi_campaign.material_scan import (  # noqa: E402
    KB_MEV_PER_K,
    heat_capacity_curves,
    solve_point,
    two_peak_summary,
)

CACHE = ROOT / "campaign" / "cache" / "ce2hf2o7_points"
OUTPUT = ROOT / "campaign" / "outputs"


def scan_grid() -> list[float]:
    """Jpm/Jzz values, pi-flux -0.30..-0.01 and zero-flux +0.01..+0.05.

    The extra half-step points at +-0.015 refine the approach to the
    perturbative limit; +-0.01 remain in the grid but their ring peak lies
    below the temperature floor at the reference Ja and is reported as
    unresolved.
    """
    negative = np.round(np.arange(-0.30, -0.005, 0.01), 6)
    positive = np.round(np.arange(0.01, 0.095, 0.01), 6)
    refine = np.array([-0.015, 0.015])
    return sorted(float(v) for v in np.concatenate([negative, positive, refine]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--ja-mev", type=float, default=0.050)
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args()

    cluster = geometry.build_cluster("cubic", (1, 1, 1))
    temperatures = np.geomspace(1.0e-5, 10.0, 2200)
    grid = scan_grid()

    if args.summary:
        records = []
        for jpm in grid:
            tag = f"jpm{jpm:+.6f}_pp{0.0:+.6f}".replace(".", "p")
            if not (CACHE / f"cubic16_xyz_{tag}.npz").exists():
                continue
            point = solve_point(cluster, jpm, 0.0, cache_dir=CACHE)
            record = {"jpm": jpm, "g6_Jzz": 12.0 * abs(jpm) ** 3,
                      "g4_Jzz": 4.0 * jpm * jpm,
                      "well_defined": bool(point.get("well_defined", True))}
            if not record["well_defined"]:
                record["failure"] = str(point["failure"])
            else:
                record.update({
                    "band_gap_Jzz": float(point["band_gap"]),
                    "model_overlap_min": float(point["model_overlap_min"]),
                })
                curves = heat_capacity_curves(point, args.ja_mev, temperatures)
                reduced = temperatures * KB_MEV_PER_K / args.ja_mev
                for label in ("winding_free", "periodic"):
                    peaks = two_peak_summary(reduced, curves[label])
                    record[label] = {
                        key.replace("_K", "_Jzz"): value for key, value in peaks.items()
                    }
            records.append(record)
        payload = {
            "protocol": "cubic-16 exact band, M=2 character + polarization blocks",
            "Ja_meV_for_reference_only": args.ja_mev,
            "temperature_grid_Jzz": [1.0e-5 * KB_MEV_PER_K / args.ja_mev,
                                     10.0 * KB_MEV_PER_K / args.ja_mev],
            "n_points": len(records),
            "points": records,
        }
        out = OUTPUT / "jpm_scan_lower_peak.json"
        out.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"wrote {out} with {len(records)} points")
        return

    selected = grid[args.offset :: args.stride]
    print(f"{len(selected)} of {len(grid)} points on stripe {args.offset}/{args.stride}",
          flush=True)
    for index, jpm in enumerate(selected, start=1):
        started = time.perf_counter()
        point = solve_point(cluster, jpm, 0.0, cache_dir=CACHE)
        elapsed = "cached" if point["cached"] else f"{time.perf_counter() - started:.0f}s"
        if not bool(point.get("well_defined", True)):
            print(f"[{index}/{len(selected)}] jpm={jpm:+.3f} {elapsed} NOT WELL DEFINED",
                  flush=True)
            continue
        curves = heat_capacity_curves(point, args.ja_mev, temperatures)
        reduced = temperatures * KB_MEV_PER_K / args.ja_mev
        summary = two_peak_summary(reduced, curves["winding_free"])
        print(f"[{index}/{len(selected)}] jpm={jpm:+.3f} {elapsed} "
              f"gap={float(point['band_gap']):.4f} "
              f"ovl={float(point['model_overlap_min']):.3f} "
              f"low_peak={summary.get('T_low_K', summary.get('T_low_Jzz', float('nan'))):.6g}",
              flush=True)


if __name__ == "__main__":
    main()
