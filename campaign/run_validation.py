#!/usr/bin/env python3
"""Run the active cubic-16/FCC-32/QMC validation campaign."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "notes"))

import recompute_finite_size_artifact as geometry  # noqa: E402
from qsi_campaign.benchmarks import (  # noqa: E402
    load_digitized_thermodynamics,
    log_grid_rmse,
)
from qsi_campaign.protocol import (  # noqa: E402
    centered_relative_error,
    full_hilbert_counterterm_spectrum,
)
from qsi_campaign.thermodynamics import peak_in_window, thermal_observables  # noqa: E402


CAMPAIGN = ROOT / "campaign"
OUTPUT = CAMPAIGN / "outputs"
OUTPUT.mkdir(parents=True, exist_ok=True)


def relative_peak_change(left: tuple[float, float], right: tuple[float, float]) -> float:
    return abs(left[0] - right[0]) / right[0]


def build_order_three(cluster: geometry.Cluster, jpm: float) -> dict[str, np.ndarray]:
    rows = geometry.sw_order23(cluster, verbose=False)
    bare_matrix = geometry.assemble(cluster, rows, jpm, "all")
    clean_matrix = geometry.assemble(cluster, rows, jpm, "delta0")
    return {
        "rows": rows,
        "bare_spectrum": np.linalg.eigvalsh(bare_matrix),
        "clean_spectrum": np.linalg.eigvalsh(clean_matrix),
    }


def main() -> None:
    config = json.loads((CAMPAIGN / "config.json").read_text())
    jpm = float(config["model"]["Jpm_over_Jzz"])
    grid = config["temperature"]
    temperatures = np.geomspace(grid["minimum"], grid["maximum"], grid["points"])

    print("building cubic-16 order-three rows", flush=True)
    cubic = geometry.build_cluster("cubic", (1, 1, 1))
    cubic_data = build_order_three(cubic, jpm)
    cubic_bare_band = thermal_observables(cubic_data["bare_spectrum"], temperatures, n_sites=16)
    cubic_clean_band = thermal_observables(cubic_data["clean_spectrum"], temperatures, n_sites=16)

    print("building FCC-32 order-three rows", flush=True)
    fcc = geometry.build_cluster("fcc", (2, 2, 2))
    fcc_data = build_order_three(fcc, jpm)
    fcc_bare_band = thermal_observables(fcc_data["bare_spectrum"], temperatures, n_sites=32)
    fcc_clean_band = thermal_observables(fcc_data["clean_spectrum"], temperatures, n_sites=32)

    full_cache = CAMPAIGN / "cache" / "full_ed_cubic16_jpm_0p046.npz"
    if not full_cache.exists():
        raise FileNotFoundError(
            f"missing {full_cache}; generate it with notes/plot_full_ed_all_temperature.py infrastructure"
        )
    full_spectrum = np.asarray(np.load(full_cache)["E_full"], dtype=float)
    bare_band_gap = float(full_spectrum[cubic.n_ice] - full_spectrum[cubic.n_ice - 1])
    clean_full_spectrum = full_hilbert_counterterm_spectrum(
        full_spectrum, cubic_data["clean_spectrum"]
    )
    bare_full = thermal_observables(full_spectrum, temperatures, n_sites=16)
    clean_full = thermal_observables(clean_full_spectrum, temperatures, n_sites=16)

    qmc_temperature, qmc_heat, qmc_entropy = load_digitized_thermodynamics(
        CAMPAIGN / config["qmc"]["thermodynamics"]
    )
    qmc_low_peak = peak_in_window(qmc_temperature, qmc_heat, 5.0e-4, 2.0e-2)
    qmc_high_peak = peak_in_window(qmc_temperature, qmc_heat, 5.0e-2, 1.0)
    cubic_bare_peak = peak_in_window(
        temperatures, bare_full["heat_capacity_per_site"], 5.0e-4, 3.0e-2
    )
    cubic_clean_peak = peak_in_window(
        temperatures, clean_full["heat_capacity_per_site"], 5.0e-4, 3.0e-2
    )
    fcc_bare_peak = peak_in_window(
        temperatures, fcc_bare_band["heat_capacity_per_site"], 2.0e-4, 3.0e-2
    )
    fcc_clean_peak = peak_in_window(
        temperatures, fcc_clean_band["heat_capacity_per_site"], 2.0e-4, 3.0e-2
    )
    high_bare_peak = peak_in_window(
        temperatures, bare_full["heat_capacity_per_site"], 5.0e-2, 1.0
    )
    high_clean_peak = peak_in_window(
        temperatures, clean_full["heat_capacity_per_site"], 5.0e-2, 1.0
    )

    qmc_low_mask = qmc_temperature <= 2.0e-2
    qmc_low_rmse_bare = log_grid_rmse(
        qmc_temperature[qmc_low_mask],
        qmc_heat[qmc_low_mask],
        temperatures,
        bare_full["heat_capacity_per_site"],
    )
    qmc_low_rmse_clean = log_grid_rmse(
        qmc_temperature[qmc_low_mask],
        qmc_heat[qmc_low_mask],
        temperatures,
        clean_full["heat_capacity_per_site"],
    )
    qmc_full_heat_rmse = log_grid_rmse(
        qmc_temperature,
        qmc_heat,
        temperatures,
        clean_full["heat_capacity_per_site"],
    )
    qmc_bare_entropy_rmse = log_grid_rmse(
        qmc_temperature,
        qmc_entropy,
        temperatures,
        bare_full["entropy_per_site"],
    )
    qmc_clean_entropy_rmse = log_grid_rmse(
        qmc_temperature,
        qmc_entropy,
        temperatures,
        clean_full["entropy_per_site"],
    )

    m2 = np.load(ROOT / "notes" / "twist_resolved_qed_dipole2_M2_jm0p05.npz")
    m3 = np.load(ROOT / "notes" / "twist_resolved_qed_dipole4_M3_jm0p05.npz")
    m_error = centered_relative_error(m2["H_qed_twist_avg"], m3["H_qed_twist_avg"])

    cubic_channels = geometry.channel_survival(cubic, cubic_data["rows"])
    fcc_channels = geometry.channel_survival(fcc, fcc_data["rows"])
    topology_pass = all(
        channels["H2"]["4_loop_wrapping"]["delta0_max"] < 1.0e-12
        and channels["H3"]["hexagon_contractible"]["delta0_min"] > 1.0 - 1.0e-12
        for channels in (cubic_channels, fcc_channels)
    )

    thresholds = config["gates"]
    high_change = relative_peak_change(high_clean_peak, high_bare_peak)
    high_height_change = abs(high_clean_peak[1] - high_bare_peak[1]) / high_bare_peak[1]
    qmc_low_peak_log_error = abs(np.log10(cubic_clean_peak[0] / qmc_low_peak[0]))
    qmc_high_peak_log_error = abs(np.log10(high_clean_peak[0] / qmc_high_peak[0]))
    report = {
        "method": "canonical order-three gauge block plus zero-transport projection; exact cubic complement retained",
        "coupling": jpm,
        "clusters": {
            "cubic16": {
                "n_ice": cubic.n_ice,
                "loops4": geometry.cycle_summary(cubic.loops4),
                "hexagons": geometry.cycle_summary(cubic.hexes),
                "transport_channels": cubic_channels,
                "bare_low_peak": cubic_bare_peak,
                "clean_low_peak": cubic_clean_peak,
            },
            "fcc32": {
                "n_ice": fcc.n_ice,
                "loops4": geometry.cycle_summary(fcc.loops4),
                "hexagons": geometry.cycle_summary(fcc.hexes),
                "transport_channels": fcc_channels,
                "bare_low_peak_order3_band": fcc_bare_peak,
                "clean_low_peak_order3_band": fcc_clean_peak,
            },
        },
        "qmc": {
            "status": config["qmc"]["status"],
            "source": config["qmc"]["source"],
            "low_peak": qmc_low_peak,
            "high_peak": qmc_high_peak,
            "cubic16_bare_low_temperature_heat_rmse": qmc_low_rmse_bare,
            "cubic16_clean_low_temperature_heat_rmse": qmc_low_rmse_clean,
            "cubic16_clean_full_heat_rmse": qmc_full_heat_rmse,
            "cubic16_bare_entropy_rmse": qmc_bare_entropy_rmse,
            "cubic16_clean_entropy_rmse": qmc_clean_entropy_rmse,
            "clean_low_peak_log10_error": qmc_low_peak_log_error,
            "clean_high_peak_log10_error": qmc_high_peak_log_error,
        },
        "all_temperature": {
            "bare_high_peak": high_bare_peak,
            "clean_high_peak": high_clean_peak,
            "relative_high_peak_temperature_change": high_change,
            "relative_high_peak_height_change": high_height_change,
        },
        "band_separation": {
            "cubic16_gap_above_lowest_90": bare_band_gap,
            "qualification": "spectral gap only; an eigenvector-overlap gate is still required for each exact pullback",
        },
        "character_convergence": {
            "coupling": -0.05,
            "comparison": "existing exact pulled-back cubic-16 M=2 vs M=3 operators",
            "centered_operator_relative_error": m_error,
        },
        "gates": {
            "topology": {"status": "pass" if topology_pass else "fail"},
            "order": {"status": "pending", "reason": "order four not computed"},
            "character": {
                "status": "pass" if m_error < thresholds["m_centered_operator_error_max"] else "fail",
                "threshold": thresholds["m_centered_operator_error_max"],
                "qualification": "precheck only; stored M=2 and M=3 runs use different integer source normalizations",
            },
            "all_temperature": {
                "status": "pass" if high_change < thresholds["high_peak_relative_change_max"] else "fail",
                "threshold": thresholds["high_peak_relative_change_max"],
            },
            "cubic16_spectral_separation": {
                "status": "pass" if bare_band_gap > thresholds["cubic16_band_gap_min"] else "fail",
                "threshold": thresholds["cubic16_band_gap_min"],
            },
            "qmc_heat_capacity": {
                "status": "pass"
                if qmc_low_rmse_clean < qmc_low_rmse_bare
                and qmc_low_rmse_clean < thresholds["qmc_low_temperature_rmse_max"]
                and qmc_low_peak_log_error < thresholds["qmc_peak_log10_error_max"]
                and qmc_high_peak_log_error < thresholds["qmc_high_peak_log10_error_max"]
                else "fail",
                "rmse_threshold": thresholds["qmc_low_temperature_rmse_max"],
                "low_peak_log10_threshold": thresholds["qmc_peak_log10_error_max"],
                "high_peak_log10_threshold": thresholds["qmc_high_peak_log10_error_max"],
            },
            "qmc_entropy": {
                "status": "pass"
                if qmc_clean_entropy_rmse < qmc_bare_entropy_rmse
                and qmc_clean_entropy_rmse < thresholds["qmc_entropy_rmse_max"]
                else "fail",
                "rmse_threshold": thresholds["qmc_entropy_rmse_max"],
                "rmse_reduction_fraction": 1.0
                - qmc_clean_entropy_rmse / qmc_bare_entropy_rmse,
            },
            "qmc_relative_improvement": {
                "status": "pass" if qmc_low_rmse_clean < qmc_low_rmse_bare else "fail",
                "low_temperature_heat_rmse_reduction_fraction": 1.0
                - qmc_low_rmse_clean / qmc_low_rmse_bare,
            },
            "qmc_energy": {
                "status": "pending",
                "reason": "the paper plots C and S but not an energy curve",
            },
            "qmc_sac_dynamics": {
                "status": config["qmc"]["dynamics"]["status"],
                "reason": config["qmc"]["dynamics"]["reason"],
            },
            "fcc32_full_hamiltonian": {
                "status": "pending",
                "reason": "current FCC-32 calculation is order three on Ran(P_ice) only",
            },
        },
    }

    np.savez_compressed(
        OUTPUT / "validation_curves.npz",
        temperature=temperatures,
        cubic_bare_band_c=cubic_bare_band["heat_capacity_per_site"],
        cubic_clean_band_c=cubic_clean_band["heat_capacity_per_site"],
        fcc_bare_band_c=fcc_bare_band["heat_capacity_per_site"],
        fcc_clean_band_c=fcc_clean_band["heat_capacity_per_site"],
        bare_full_c=bare_full["heat_capacity_per_site"],
        clean_full_c=clean_full["heat_capacity_per_site"],
        bare_full_s=bare_full["entropy_per_site"],
        clean_full_s=clean_full["entropy_per_site"],
        qmc_temperature=qmc_temperature,
        qmc_heat_capacity=qmc_heat,
        qmc_entropy=qmc_entropy,
        cubic_bare_band_spectrum=cubic_data["bare_spectrum"],
        cubic_clean_band_spectrum=cubic_data["clean_spectrum"],
        fcc_bare_band_spectrum=fcc_data["bare_spectrum"],
        fcc_clean_band_spectrum=fcc_data["clean_spectrum"],
    )
    (OUTPUT / "validation_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report["gates"], indent=2))
    print(f"wrote {OUTPUT / 'validation_report.json'}")


if __name__ == "__main__":
    main()
