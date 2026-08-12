#!/usr/bin/env python3
"""Rank scanned XYZ coupling points against the Ce2Hf2O7 heat capacity.

For a fixed (J_pm/J_zz, J_pmpm/J_zz) the winding-free heat capacity is a fixed
curve in reduced temperature, so J_zz alone sets where it lands in Kelvin.
Each scan point therefore contributes a one-parameter family, and the honest
comparison optimises J_zz for every point before ranking.  Reporting the
best-fit J_zz alongside the residual also shows whether a point can only fit
by choosing a J_zz far from the value the high-temperature tail allows.

The perturbative ring-exchange scale g6 = 12 |J_pm|^3 / J_zz^2 is printed for
contrast: the "s2" set was constructed to put g6 on the measured T2, and the
point of this scan is that the nonperturbative peak need not sit at g6.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qsi_campaign.material_fit import load_digitized_series  # noqa: E402
from qsi_campaign.protocol import full_hilbert_counterterm_spectrum  # noqa: E402
from qsi_campaign.thermodynamics import thermal_observables  # noqa: E402

GAS_CONSTANT = 8.31446261815324
MEV_TO_K = 1.0 / 0.08617333262
T1, T2 = 0.0635, 0.0244          # measured anomalies, Kelvin
JZZ_BOUNDS = (0.15, 2.5)         # Kelvin; brackets every published set
OUTPUT = ROOT / "campaign" / "outputs" / "ce2hf2o7_scan_ranking.json"


def coupling_tag(value: float) -> str:
    return f"{value:+.6f}".replace("+", "p").replace("-", "m").replace(".", "p")


def model_tag(jpm: float, jpmpm: float) -> str:
    tag = coupling_tag(jpm)
    if jpmpm != 0.0:
        tag += f"_pp{coupling_tag(jpmpm)}"
    return tag


def local_maxima(temperature, heat, floor=0.05):
    interior = np.flatnonzero(
        (heat[1:-1] > heat[:-2]) & (heat[1:-1] >= heat[2:]) & (heat[1:-1] > floor)
    ) + 1
    return [(float(temperature[i]), float(heat[i])) for i in interior]


def evaluate(jpm: float, jpmpm: float):
    tag = model_tag(jpm, jpmpm)
    full_path = ROOT / "campaign" / "cache" / f"full_ed_cubic16_{tag}.npz"
    band_path = ROOT / "campaign" / "outputs" / f"nonperturbative_cubic16_{tag}.npz"
    if not (full_path.exists() and band_path.exists()):
        return None

    full = np.load(full_path)
    spectrum = np.asarray(full["E_full"], dtype=float)
    band = np.load(band_path)
    spliced = full_hilbert_counterterm_spectrum(spectrum, band["M2_spectrum"])

    experiment_t, experiment_c = load_digitized_series(
        ROOT / "campaign" / "data" / "ce2hf2o7_smith2025_digitized.csv"
    )

    def residual(jzz_kelvin: float) -> float:
        reduced = experiment_t / jzz_kelvin
        model = thermal_observables(spliced, reduced, n_sites=16)
        heat = model["heat_capacity_per_site"] * GAS_CONSTANT
        return float(np.sqrt(np.mean((heat - experiment_c) ** 2)))

    best = minimize_scalar(residual, bounds=JZZ_BOUNDS, method="bounded",
                           options={"xatol": 1.0e-4})
    jzz = float(best.x)

    reduced = np.geomspace(1.0e-4, 20.0, 4000)
    winding_free = thermal_observables(spliced, reduced, n_sites=16)
    periodic = thermal_observables(spectrum, reduced, n_sites=16)
    kelvin = reduced * jzz

    peaks = local_maxima(kelvin, winding_free["heat_capacity_per_site"] * GAS_CONSTANT)
    gap = float(spectrum[90] - spectrum[89])
    return {
        "jpm_over_jzz": jpm,
        "jpmpm_over_jzz": jpmpm,
        "best_fit_jzz_K": jzz,
        "best_fit_jzz_meV": jzz / MEV_TO_K,
        "rmse": float(best.fun),
        "band_gap_over_jzz": gap,
        "band_gap_K": gap * jzz,
        "g6_over_jzz": 12.0 * abs(jpm) ** 3,
        "g6_K": 12.0 * abs(jpm) ** 3 * jzz,
        "winding_free_peaks_K": peaks,
        "periodic_peaks_K": local_maxima(
            kelvin, periodic["heat_capacity_per_site"] * GAS_CONSTANT
        ),
        "lowest_peak_K": peaks[0][0] if peaks else None,
        "lowest_peak_over_T2": peaks[0][0] / T2 if peaks else None,
    }


def main() -> None:
    points = [
        (-0.04, 0.11532091), (-0.06, 0.11532091), (-0.08, 0.11532091),
        (-0.10, 0.11532091), (-0.13, 0.11532091), (-0.15555742, 0.11532091),
        (-0.20, 0.11532091), (-0.25, 0.11532091), (-0.10, 0.0),
    ]
    results = [item for point in points if (item := evaluate(*point)) is not None]
    results.sort(key=lambda item: item["rmse"])

    print(f"measured anomalies: T1 = {T1*1e3:.1f} mK, T2 = {T2*1e3:.1f} mK\n")
    header = (f"{'Jpm/Jzz':>9} {'Jpmpm/Jzz':>10} {'Jzz(K)':>7} {'RMSE':>6} "
              f"{'gap(mK)':>8} {'g6(mK)':>7} {'peak(mK)':>9} {'peak/T2':>8}")
    print(header)
    print("-" * len(header))
    for item in results:
        peak = item["lowest_peak_K"]
        print(
            f"{item['jpm_over_jzz']:>9.4f} {item['jpmpm_over_jzz']:>10.4f} "
            f"{item['best_fit_jzz_K']:>7.3f} {item['rmse']:>6.3f} "
            f"{item['band_gap_K']*1e3:>8.1f} {item['g6_K']*1e3:>7.1f} "
            f"{(peak*1e3 if peak else float('nan')):>9.1f} "
            f"{(item['lowest_peak_over_T2'] or float('nan')):>8.2f}"
        )
    OUTPUT.write_text(json.dumps(results, indent=2) + "\n")
    print(f"\nwrote {OUTPUT}")


if __name__ == "__main__":
    main()
