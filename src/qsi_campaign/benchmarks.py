"""Metrics used by the validation gates."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


def load_digitized_curve(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    with Path(path).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    return (
        np.asarray([float(row["temperature_over_Jzz"]) for row in rows]),
        np.asarray([float(row["heat_capacity_per_site"]) for row in rows]),
    )


def load_digitized_thermodynamics(
    path: str | Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load temperature, heat capacity, and entropy from one CSV."""
    with Path(path).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    return (
        np.asarray([float(row["temperature_over_Jzz"]) for row in rows]),
        np.asarray([float(row["heat_capacity_per_site"]) for row in rows]),
        np.asarray([float(row["entropy_per_site"]) for row in rows]),
    )


def log_grid_rmse(
    reference_temperature: np.ndarray,
    reference_value: np.ndarray,
    model_temperature: np.ndarray,
    model_value: np.ndarray,
) -> float:
    """RMSE after interpolation in log temperature over the common support."""
    reference_temperature = np.asarray(reference_temperature)
    mask = (
        (reference_temperature >= np.min(model_temperature))
        & (reference_temperature <= np.max(model_temperature))
    )
    interpolated = np.interp(
        np.log(reference_temperature[mask]),
        np.log(model_temperature),
        model_value,
    )
    return float(np.sqrt(np.mean((interpolated - reference_value[mask]) ** 2)))
