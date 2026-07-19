"""Stable canonical thermodynamics from a finite spectrum."""

from __future__ import annotations

import numpy as np


def thermal_observables(
    spectrum: np.ndarray,
    temperatures: np.ndarray,
    *,
    n_sites: int = 1,
    batch_size: int = 128,
) -> dict[str, np.ndarray]:
    """Compute energy, heat capacity, and entropy per site."""
    energies = np.asarray(spectrum, dtype=float)
    temperatures = np.asarray(temperatures, dtype=float)
    if energies.ndim != 1 or temperatures.ndim != 1:
        raise ValueError("spectrum and temperatures must be one-dimensional")
    if len(energies) == 0 or np.any(temperatures <= 0) or n_sites <= 0:
        raise ValueError("nonempty spectrum, positive temperatures and n_sites are required")

    shifted = energies - energies.min()
    energy = np.empty_like(temperatures)
    heat = np.empty_like(temperatures)
    entropy = np.empty_like(temperatures)
    for start in range(0, len(temperatures), batch_size):
        temp = temperatures[start : start + batch_size]
        beta = 1.0 / temp[:, None]
        weights = np.exp(-beta * shifted[None, :])
        partition = weights.sum(axis=1)
        first = (weights @ shifted) / partition
        second = (weights @ shifted**2) / partition
        energy[start : start + batch_size] = (energies.min() + first) / n_sites
        heat[start : start + batch_size] = (second - first**2) / temp**2 / n_sites
        entropy[start : start + batch_size] = (
            np.log(partition) + first / temp
        ) / n_sites
    return {
        "temperature": temperatures,
        "energy_per_site": energy,
        "heat_capacity_per_site": heat,
        "entropy_per_site": entropy,
    }


def peak_in_window(
    temperatures: np.ndarray,
    values: np.ndarray,
    lower: float,
    upper: float,
) -> tuple[float, float]:
    """Locate a sampled maximum in a closed temperature window."""
    mask = (temperatures >= lower) & (temperatures <= upper)
    if not np.any(mask):
        raise ValueError("peak window does not intersect the temperature grid")
    indices = np.flatnonzero(mask)
    index = indices[np.argmax(np.asarray(values)[mask])]
    return float(temperatures[index]), float(np.asarray(values)[index])
