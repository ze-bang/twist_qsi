"""Ce2Hf2O7 curve loading, validation, and discrete parameter fitting."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


MODEL_METHOD = "fcc32_exact_winding_free_xyz"


def abc_to_transverse(ja: float, jb: float, jc: float) -> dict[str, float]:
    """Map the ordered ABC exchanges to the dominant-axis transverse couplings."""
    return {
        "Ja_meV": float(ja),
        "Jpm_meV": float(-(jb + jc) / 4.0),
        "Jpmpm_meV": float((jb - jc) / 4.0),
    }


def load_digitized_series(path: Path, series: str = "experiment") -> tuple[np.ndarray, np.ndarray]:
    rows: list[tuple[float, float]] = []
    with Path(path).open(newline="") as handle:
        reader = csv.DictReader(line for line in handle if not line.startswith("#"))
        for row in reader:
            if row["series"] == series:
                rows.append(
                    (float(row["temperature_K"]), float(row["Cmag_J_molCe_K"]))
                )
    if not rows:
        raise ValueError(f"series {series!r} is absent from {path}")
    rows.sort()
    return np.asarray([row[0] for row in rows]), np.asarray([row[1] for row in rows])


def curve_score(
    experiment_temperature: np.ndarray,
    experiment_heat: np.ndarray,
    model_temperature: np.ndarray,
    model_heat: np.ndarray,
    temperature_window: tuple[float, float],
) -> dict[str, float | int]:
    """Return an unweighted log-temperature interpolation score.

    The raster extraction has no statistical uncertainties, so this is an
    exploratory ranking metric rather than a chi-square statistic.
    """
    experiment_temperature = np.asarray(experiment_temperature, dtype=float)
    experiment_heat = np.asarray(experiment_heat, dtype=float)
    model_temperature = np.asarray(model_temperature, dtype=float)
    model_heat = np.asarray(model_heat, dtype=float)
    if np.any(model_temperature <= 0) or np.any(experiment_temperature <= 0):
        raise ValueError("temperatures must be positive for log interpolation")
    order = np.argsort(model_temperature)
    model_temperature = model_temperature[order]
    model_heat = model_heat[order]
    low = max(float(temperature_window[0]), float(model_temperature[0]))
    high = min(float(temperature_window[1]), float(model_temperature[-1]))
    keep = (experiment_temperature >= low) & (experiment_temperature <= high)
    if np.count_nonzero(keep) < 3:
        raise ValueError("fewer than three experimental points overlap the model curve")
    predicted = np.interp(
        np.log(experiment_temperature[keep]), np.log(model_temperature), model_heat
    )
    residual = predicted - experiment_heat[keep]
    return {
        "n_points": int(np.count_nonzero(keep)),
        "rmse_J_molCe_K": float(np.sqrt(np.mean(residual**2))),
        "mae_J_molCe_K": float(np.mean(np.abs(residual))),
        "maximum_absolute_error_J_molCe_K": float(np.max(np.abs(residual))),
    }


def _scalar(data: np.lib.npyio.NpzFile, key: str):
    if key not in data:
        raise ValueError(f"model curve is missing required field {key!r}")
    value = np.asarray(data[key])
    if value.size != 1:
        raise ValueError(f"model field {key!r} must be scalar")
    return value.reshape(()).item()


def load_fcc32_model_curve(path: Path, require_converged: bool = True) -> dict:
    """Load one production curve and enforce the FCC-32 result contract."""
    with np.load(path, allow_pickle=False) as data:
        method = str(_scalar(data, "method"))
        sites = int(_scalar(data, "n_sites"))
        character_converged = bool(_scalar(data, "character_converged"))
        complement_converged = bool(_scalar(data, "complement_converged"))
        if method != MODEL_METHOD:
            raise ValueError(f"{path} uses {method!r}, expected {MODEL_METHOD!r}")
        if sites != 32:
            raise ValueError(f"{path} has {sites} sites, expected FCC-32")
        if require_converged and not character_converged:
            raise ValueError(f"{path} failed character-grid convergence")
        if require_converged and not complement_converged:
            raise ValueError(f"{path} failed stochastic-complement convergence")
        temperature = np.asarray(data["temperature_K"], dtype=float)
        heat = np.asarray(data["heat_capacity_J_molCe_K"], dtype=float)
        if temperature.ndim != 1 or heat.shape != temperature.shape:
            raise ValueError(f"{path} has incompatible temperature and heat arrays")
        parameters = {
            key: float(_scalar(data, key)) for key in ("Ja_meV", "Jb_meV", "Jc_meV")
        }
        return {
            "path": str(Path(path)),
            "temperature_K": temperature,
            "heat_capacity_J_molCe_K": heat,
            "parameters": parameters,
            "character_M": int(_scalar(data, "character_M")),
            "character_converged": character_converged,
            "complement_converged": complement_converged,
        }


def rank_model_curves(
    paths: list[Path],
    experiment_temperature: np.ndarray,
    experiment_heat: np.ndarray,
    temperature_window: tuple[float, float],
    require_converged: bool = True,
) -> list[dict]:
    ranked = []
    for path in paths:
        curve = load_fcc32_model_curve(path, require_converged=require_converged)
        score = curve_score(
            experiment_temperature,
            experiment_heat,
            curve["temperature_K"],
            curve["heat_capacity_J_molCe_K"],
            temperature_window,
        )
        ranked.append(
            {
                "path": curve["path"],
                "parameters": curve["parameters"],
                "transverse": abc_to_transverse(**{
                    "ja": curve["parameters"]["Ja_meV"],
                    "jb": curve["parameters"]["Jb_meV"],
                    "jc": curve["parameters"]["Jc_meV"],
                }),
                "character_M": curve["character_M"],
                "score": score,
            }
        )
    return sorted(ranked, key=lambda item: item["score"]["rmse_J_molCe_K"])
