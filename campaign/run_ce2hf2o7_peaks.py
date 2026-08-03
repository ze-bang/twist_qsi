#!/usr/bin/env python3
"""Scan the Smith et al. (Jpm/Ja, Jpmpm/Ja) triangle for the Ce2Hf2O7 two-peak structure.

Each parameter point is one cubic-16 XYZ calculation: the full 65,536-state
microscopic spectrum by symmetry blocks, the exact ice-descended band, and its
``M=2`` winding-free replacement.  ``Ja`` enters only as an energy scale, so a
cached point serves every ``Ja``.

Run a stripe of the grid with ``--stride N --offset I`` to use several
processes; the point cache deduplicates.
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
    abc_from_ratios,
    heat_capacity_curves,
    in_published_triangle,
    solve_point,
    two_peak_summary,
)

CACHE = ROOT / "campaign" / "cache" / "ce2hf2o7_points"
OUTPUT = ROOT / "campaign" / "outputs"


def grid_points(jpm_values: np.ndarray, jpmpm_values: np.ndarray) -> list[tuple[float, float]]:
    points = {
        (round(float(jpm), 6), round(float(jpmpm), 6))
        for jpm in jpm_values
        for jpmpm in jpmpm_values
        if in_published_triangle(float(jpm), float(jpmpm))
    }
    # Work outward from the weakly transverse corner.  The band construction is
    # only defined while the ice-descended band stays isolated, which fails deep
    # in the triangle; visiting the cheap, well-defined points first means a
    # partial scan is still a usable map.
    return sorted(points, key=lambda point: (abs(point[0]) + point[1], abs(point[0])))


def parse_points(text: str) -> list[tuple[float, float]]:
    out = []
    for item in text.split(";"):
        if not item.strip():
            continue
        jpm, jpmpm = (float(value) for value in item.split(","))
        out.append((round(jpm, 6), round(jpmpm, 6)))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--points", type=str, default=None, help="jpm,jpmpm;jpm,jpmpm;...")
    parser.add_argument("--jpm-range", type=float, nargs=3, default=(-0.45, -0.05, 9),
                        metavar=("LO", "HI", "N"))
    parser.add_argument("--jpmpm-range", type=float, nargs=3, default=(0.0, 0.30, 7),
                        metavar=("LO", "HI", "N"))
    parser.add_argument("--ja-mev", type=float, default=0.050,
                        help="published high-temperature scale, meV")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--summary", type=Path, default=None,
                        help="write a JSON summary of every cached point and exit")
    args = parser.parse_args()

    cluster = geometry.build_cluster("cubic", (1, 1, 1))
    # The ring-exchange peak falls below a milli-kelvin at weak transverse
    # coupling, so the grid has to reach well under the plotted range.
    temperatures = np.geomspace(1.0e-4, 10.0, 1800)

    if args.points is not None:
        points = parse_points(args.points)
    else:
        points = grid_points(
            np.linspace(args.jpm_range[0], args.jpm_range[1], int(args.jpm_range[2])),
            np.linspace(args.jpmpm_range[0], args.jpmpm_range[1], int(args.jpmpm_range[2])),
        )

    if args.summary is not None:
        records = []
        for jpm, jpmpm in points:
            tag = f"jpm{jpm:+.6f}_pp{jpmpm:+.6f}".replace(".", "p")
            if not (CACHE / f"cubic16_xyz_{tag}.npz").exists():
                continue
            point = solve_point(cluster, jpm, jpmpm, cache_dir=CACHE)
            ja, jb, jc = abc_from_ratios(args.ja_mev, jpm, jpmpm)
            if not bool(point.get("well_defined", True)):
                records.append({
                    "jpm": jpm, "jpmpm": jpmpm,
                    "Ja_meV": ja, "Jb_meV": jb, "Jc_meV": jc,
                    "well_defined": False,
                    "failure": str(point["failure"]),
                })
                continue
            curves = heat_capacity_curves(point, args.ja_mev, temperatures)
            records.append({
                "jpm": jpm,
                "jpmpm": jpmpm,
                "Ja_meV": ja, "Jb_meV": jb, "Jc_meV": jc,
                "well_defined": True,
                "band_gap_Ja": float(point["band_gap"]),
                "model_overlap_min": float(point["model_overlap_min"]),
                "periodic": two_peak_summary(temperatures, curves["periodic"]),
                "winding_free": two_peak_summary(temperatures, curves["winding_free"]),
            })
            # Ja is a pure energy scale, so the reduced peak positions and their
            # ratio are properties of (jpm, jpmpm) alone and can be rescaled to
            # any Ja without recomputing.
            for label in ("periodic", "winding_free"):
                peaks = records[-1][label]
                for key in ("T_low_K", "T_high_K"):
                    if key in peaks:
                        peaks[key.replace("_K", "_over_Ja")] = (
                            peaks[key] * KB_MEV_PER_K / args.ja_mev
                        )
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(json.dumps(
            {"Ja_meV": args.ja_mev, "n_points": len(records), "points": records}, indent=2) + "\n")
        print(f"wrote {args.summary} with {len(records)} points")
        return

    selected = points[args.offset :: args.stride]
    print(f"{len(selected)} of {len(points)} points on stripe {args.offset}/{args.stride}", flush=True)
    for index, (jpm, jpmpm) in enumerate(selected, start=1):
        started = time.perf_counter()
        point = solve_point(cluster, jpm, jpmpm, cache_dir=CACHE)
        elapsed = "cached" if point["cached"] else f"{time.perf_counter()-started:.0f}s"
        prefix = f"[{index}/{len(selected)}] jpm={jpm:+.4f} jpmpm={jpmpm:+.4f} {elapsed}"
        if not bool(point.get("well_defined", True)):
            print(f"{prefix} BAND NOT ISOLATED: {point['failure']}", flush=True)
            continue
        curves = heat_capacity_curves(point, args.ja_mev, temperatures)
        summary = two_peak_summary(temperatures, curves["winding_free"])
        print(
            f"{prefix} gap={float(point['band_gap']):.4f} "
            f"ovl={float(point['model_overlap_min']):.3f} "
            f"peaks={ {k: round(v, 5) for k, v in summary.items()} }",
            flush=True,
        )


if __name__ == "__main__":
    main()
