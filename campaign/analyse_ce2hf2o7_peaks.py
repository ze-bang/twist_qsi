#!/usr/bin/env python3
"""Ask whether winding-free cubic-16 XYZ can place both Ce2Hf2O7 peaks.

The measured maxima are ``T_2`` and ``T_1``.  Because ``Ja`` is a pure energy
scale, a parameter point is characterised by the Ja-independent ratio
``rho = T_2/T_1``; ``Ja`` afterwards slides both peaks together.  The question
therefore splits in two:

1. can any point in the well-defined region reach the measured ``rho``, and
2. what ``Ja`` does that point then require, compared with the published one?

Reported alongside is the pinned-``Ja`` branch, in which ``Ja`` is held at the
published high-temperature value and only ``(Jb,Jc)`` move.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qsi_campaign.material_scan import KB_MEV_PER_K, abc_from_ratios  # noqa: E402
from qsi_campaign.material_fit import load_digitized_series  # noqa: E402
from qsi_campaign.material_scan import local_maxima  # noqa: E402

PUBLISHED_JA_MEV = 0.050
PUBLISHED_SEEDS = {
    "NLC_A": (0.050, 0.021, 0.004),
    "NLC_B": (0.051, 0.008, -0.018),
}


def measured_peaks(path: Path, smoothing: int = 5) -> tuple[float, float]:
    """The two measured maxima, from the digitized Fig. 2(a) experimental trace.

    The raster trace is noisy at the pixel level, so it is boxcar-smoothed in
    log temperature before the maxima are located; the quoted values are stable
    against the smoothing width between 5 and 9 points.
    """
    temperature, heat = load_digitized_series(path)
    kernel = np.ones(smoothing) / smoothing
    smoothed = np.convolve(heat, kernel, "same")
    interior = slice(smoothing, -smoothing)
    peaks = [
        peak for peak in local_maxima(temperature[interior], smoothed[interior])
        if peak[0] < 0.2
    ]
    peaks.sort(key=lambda item: -item[1])
    low, high = sorted(peak[0] for peak in peaks[:2])
    return float(low), float(high)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path,
                        default=ROOT / "campaign" / "outputs" / "ce2hf2o7_peak_scan.json")
    parser.add_argument("--data", type=Path,
                        default=ROOT / "campaign" / "data" / "ce2hf2o7_smith2025_digitized.csv")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "campaign" / "outputs" / "ce2hf2o7_peak_analysis.json")
    args = parser.parse_args()

    t2, t1 = measured_peaks(args.data)
    target_ratio = t2 / t1
    scan = json.loads(args.summary.read_text())
    points = scan["points"]
    usable = [
        point for point in points
        if point.get("well_defined") and "peak_ratio" in point["winding_free"]
    ]
    if not usable:
        raise SystemExit("no well-defined two-peak points in the scan summary")

    records = []
    for point in usable:
        peaks = point["winding_free"]
        ratio = peaks["peak_ratio"]
        # Ja that would place the upper winding-free maximum on the measured T1
        required_ja = t1 * KB_MEV_PER_K / peaks["T_high_over_Ja"]
        ja, jb, jc = abc_from_ratios(required_ja, point["jpm"], point["jpmpm"])
        records.append({
            "jpm": point["jpm"],
            "jpmpm": point["jpmpm"],
            "band_gap_Ja": point["band_gap_Ja"],
            "model_overlap_min": point["model_overlap_min"],
            "peak_ratio": ratio,
            "ratio_shortfall": target_ratio / ratio,
            "pinned_Ja": {
                "Ja_meV": PUBLISHED_JA_MEV,
                "T_low_K": peaks["T_low_K"],
                "T_high_K": peaks["T_high_K"],
            },
            "rescaled_Ja": {
                "Ja_meV": required_ja, "Jb_meV": jb, "Jc_meV": jc,
                "T_low_K": t1 * ratio,
                "T_high_K": t1,
            },
        })

    records.sort(key=lambda item: -item["peak_ratio"])
    best = records[0]
    ill_defined = [point for point in points if not point.get("well_defined")]
    report = {
        "measured": {"T2_K": t2, "T1_K": t1, "ratio": target_ratio},
        "published_Ja_meV": PUBLISHED_JA_MEV,
        "published_seeds": PUBLISHED_SEEDS,
        "n_points": len(points),
        "n_well_defined": len(usable),
        "n_band_not_isolated": len(ill_defined),
        "largest_reachable_ratio": best["peak_ratio"],
        "reaches_measured_ratio": bool(best["peak_ratio"] >= target_ratio),
        "ranked": records,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")

    print(f"measured  T2 = {t2*1e3:.1f} mK, T1 = {t1*1e3:.1f} mK, ratio = {target_ratio:.4f}")
    print(f"scan      {len(usable)} well-defined of {len(points)} points, "
          f"{len(ill_defined)} with a non-isolated band")
    print(f"best ratio {best['peak_ratio']:.4f} at "
          f"(jpm, jpmpm) = ({best['jpm']:+.3f}, {best['jpmpm']:+.3f}), "
          f"short of the measured ratio by {best['ratio_shortfall']:.2f}x")
    print(f"{'jpm':>7} {'jpmpm':>7} {'ratio':>8} {'gap/Ja':>8} {'ovl':>6} "
          f"{'T2 pinned':>10} {'T1 pinned':>10} {'Ja needed':>10}")
    for record in records[:15]:
        print(f"{record['jpm']:+7.3f} {record['jpmpm']:+7.3f} {record['peak_ratio']:8.4f} "
              f"{record['band_gap_Ja']:8.4f} {record['model_overlap_min']:6.3f} "
              f"{record['pinned_Ja']['T_low_K']*1e3:9.2f}m {record['pinned_Ja']['T_high_K']*1e3:9.1f}m "
              f"{record['rescaled_Ja']['Ja_meV']:10.4f}")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
