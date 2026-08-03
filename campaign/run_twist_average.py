#!/usr/bin/env python3
"""Observable-level twist-averaged heat capacity across the Jpm scan.

For each coupling the full 65,536-state cubic-16 XXZ spectrum is solved on a
reduced smooth-gauge twist grid (``M`` half-step points per axis on
``[0, pi)^3``, equivalent to a ``2M``-per-axis average over the full twist
torus through the corner gauge identity), the heat capacity is computed per
twist, and the twist average is taken at the observable level.  This needs no
band isolation and is therefore defined at every coupling, including where the
winding-free construction fails.

Comparison curves per point: bare periodic (theta = 0), the twist average,
the twist-resolved spread, and — where the cached winding-free band exists —
the full-Hilbert winding-free replacement curve.

    python3 campaign/run_twist_average.py             # M=2 production sweep
    python3 campaign/run_twist_average.py --grid 3 --jpm -0.05 -0.15 -0.25
    python3 campaign/run_twist_average.py --summary   # collect JSON
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
from qsi_campaign.protocol import full_hilbert_counterterm_spectrum  # noqa: E402
from qsi_campaign.thermodynamics import thermal_observables  # noqa: E402
from qsi_campaign.material_scan import two_peak_summary  # noqa: E402
from qsi_campaign.twist_average import (  # noqa: E402
    N_SITES,
    build_structures,
    full_spectrum,
    reduced_twist_grid,
)
from run_jpm_scan import scan_grid  # noqa: E402

CACHE = ROOT / "campaign" / "cache" / "twist_avg"
BAND_CACHE = ROOT / "campaign" / "cache" / "ce2hf2o7_points"
OUTPUT = ROOT / "campaign" / "outputs"

TEMPERATURES = np.geomspace(1.0e-5, 20.0, 2200)  # units of Jzz


def point_tag(jpm: float) -> str:
    return f"jpm{jpm:+.6f}".replace(".", "p")


def cached_spectrum(structures, jpm: float, theta: np.ndarray, label: str) -> np.ndarray:
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"spec_{point_tag(jpm)}_{label}.npz"
    if path.exists():
        with np.load(path) as saved:
            return saved["spectrum"]
    spectrum = full_spectrum(structures, jpm, theta)
    np.savez_compressed(path, spectrum=spectrum, theta=theta, jpm=jpm)
    return spectrum


def heat_capacity(spectrum: np.ndarray) -> np.ndarray:
    thermal = thermal_observables(spectrum, TEMPERATURES, n_sites=N_SITES)
    return thermal["heat_capacity_per_site"]


def winding_free_curve(jpm: float) -> np.ndarray | None:
    path = BAND_CACHE / f"cubic16_xyz_{point_tag(jpm)}_pp+0p000000.npz"
    if not path.exists():
        return None
    with np.load(path) as saved:
        if not bool(saved.get("well_defined", np.asarray(True))):
            return None
        replaced = full_hilbert_counterterm_spectrum(
            saved["full_spectrum"], saved["winding_free_band"]
        )
    return heat_capacity(replaced)


def solve_point(structures, jpm: float, grid_size: int) -> dict:
    started = time.perf_counter()
    grid = reduced_twist_grid(grid_size)
    bare = cached_spectrum(structures, jpm, np.zeros(3), "bare")
    curves, weights, low_peaks = [], [], []
    for index, (theta, weight) in enumerate(grid):
        spectrum = cached_spectrum(structures, jpm, theta, f"m{grid_size}_{index:02d}")
        curve = heat_capacity(spectrum)
        curves.append(curve)
        weights.append(float(weight))
        summary = two_peak_summary(TEMPERATURES, curve)
        low_peaks.append(summary.get("T_low_K"))
    curves = np.asarray(curves)
    weights = np.asarray(weights) / np.sum(weights)
    average = weights @ curves
    result = {
        "jpm": jpm,
        "grid_size": grid_size,
        "n_twists": len(grid),
        "bare_curve": heat_capacity(bare),
        "average_curve": average,
        "curve_spread": curves.max(axis=0) - curves.min(axis=0),
        "twist_low_peaks_Jzz": low_peaks,
        "winding_free_curve": winding_free_curve(jpm),
        "solve_seconds": time.perf_counter() - started,
    }
    return result


def summarize(result: dict) -> dict:
    record = {
        "jpm": result["jpm"],
        "g6_Jzz": 12.0 * abs(result["jpm"]) ** 3,
        "g4_Jzz": 4.0 * result["jpm"] ** 2,
        "grid_size": result["grid_size"],
        "n_twists": result["n_twists"],
        "twist_low_peaks_Jzz": result["twist_low_peaks_Jzz"],
    }
    for label in ("bare", "average", "winding_free"):
        curve = result[f"{label}_curve"]
        if curve is None:
            continue
        peaks = two_peak_summary(TEMPERATURES, curve)
        record[label] = {
            key.replace("_K", "_Jzz"): value for key, value in peaks.items()
        }
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", type=int, default=2,
                        help="half-step points per axis on [0, pi)^3")
    parser.add_argument("--jpm", type=float, nargs="*", default=None,
                        help="explicit couplings (default: the full scan grid)")
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args()

    couplings = args.jpm if args.jpm else scan_grid()
    cluster = geometry.build_cluster("cubic", (1, 1, 1))
    structures = build_structures(cluster)

    records, curve_store = [], {}
    for index, jpm in enumerate(couplings, start=1):
        result = solve_point(structures, jpm, args.grid)
        record = summarize(result)
        records.append(record)
        curve_store[point_tag(jpm)] = result
        average = record.get("average", {})
        print(f"[{index}/{len(couplings)}] jpm={jpm:+.3f} "
              f"{result['solve_seconds']:.0f}s "
              f"bare_low={record.get('bare', {}).get('T_low_Jzz', float('nan')):.4g} "
              f"avg_low={average.get('T_low_Jzz', float('nan')):.4g} "
              f"wf_low={record.get('winding_free', {}).get('T_low_Jzz', float('nan')):.4g}",
              flush=True)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    payload = {
        "protocol": (
            "observable-level twist average, smooth gauge, "
            f"reduced half-step grid M={args.grid} on [0,pi)^3 "
            f"(= {2 * args.grid}^3 average over the full twist torus)"
        ),
        "temperature_grid_Jzz": [float(TEMPERATURES[0]), float(TEMPERATURES[-1])],
        "n_points": len(records),
        "points": records,
    }
    out = OUTPUT / f"twist_average_scan_m{args.grid}.json"
    out.write_text(json.dumps(payload, indent=2) + "\n")
    arrays = {"temperatures": TEMPERATURES}
    for tag, result in curve_store.items():
        for name in ("bare_curve", "average_curve", "curve_spread", "winding_free_curve"):
            value = result[name]
            if value is not None:
                arrays[f"{tag}_{name}"] = np.asarray(value)
    np.savez_compressed(OUTPUT / f"twist_average_curves_m{args.grid}.npz", **arrays)
    print(f"wrote {out} with {len(records)} points")


if __name__ == "__main__":
    main()
