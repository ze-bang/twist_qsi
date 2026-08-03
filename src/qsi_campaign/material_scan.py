"""Winding-free cubic-16 XYZ heat capacity in Ce2Hf2O7 physical units.

The nearest-neighbour dipole-octupole Hamiltonian of Smith et al. is

    H = Ja sum Sa Sa + Jb sum Sb Sb + Jc sum Sc Sc,

which in the dominant-``a`` transverse convention of this repository is

    H = Ja sum Sz Sz - Jpm sum (S+S- + h.c.) + Jpmpm sum (S+S+ + h.c.),
    Jpm = -(Jb + Jc) / 4,   Jpmpm = (Jb - Jc) / 4.

Only the two ratios ``jpm = Jpm/Ja`` and ``jpmpm = Jpmpm/Ja`` enter the
dimensionless spectrum, so ``Ja`` is a pure energy scale: a scan is a scan of
the Smith et al. Fig. 2(b) triangle ``|jpm| + jpmpm <= 1/2``.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from .exact_band import (
    basis_character_phases,
    cubic16_translation_permutations,
    extract_exact_band,
    full_spin_basis,
    microscopic_character_hamiltonian,
    translation_sector_bases,
    uniform_character_grid,
)
from .protocol import (
    character_project,
    full_hilbert_counterterm_spectrum,
    polarization_sector_labels,
    sector_project,
)
from .thermodynamics import thermal_observables

KB_MEV_PER_K = 0.086173332621  # Boltzmann constant in meV/K
R_GAS = 8.31446261815324  # J / (mol K)
N_SITES = 16


def abc_from_ratios(ja: float, jpm: float, jpmpm: float) -> tuple[float, float, float]:
    """Invert the transverse mapping back to the published ``(Ja,Jb,Jc)``."""
    return float(ja), float(2.0 * ja * (jpmpm - jpm)), float(-2.0 * ja * (jpm + jpmpm))


def ratios_from_abc(ja: float, jb: float, jc: float) -> tuple[float, float]:
    return float(-(jb + jc) / (4.0 * ja)), float((jb - jc) / (4.0 * ja))


def in_published_triangle(jpm: float, jpmpm: float, tol: float = 1e-12) -> bool:
    """The domain ``|Jb|,|Jc| <= Ja`` with ``Jb >= Jc`` of Smith et al. Fig. 2(b)."""
    return jpmpm >= -tol and abs(jpm) + jpmpm <= 0.5 + tol


def _sector_blocks(sectors: dict, states: np.ndarray, hamiltonian) -> list[np.ndarray]:
    """Split the microscopic Hamiltonian by Klein translation and S^z parity.

    ``Jpmpm`` breaks total ``S^z`` conservation but preserves its parity, so
    the 65,536-state basis factorises into eight real symmetric blocks of about
    8,192 states each.  This is what makes a full-spectrum parameter scan
    affordable.
    """
    parity = (np.bitwise_count(states.astype(np.uint64)) % 2).astype(np.int64)
    blocks = []
    for basis in sectors.values():
        projected = (basis.conj().T @ (hamiltonian @ basis)).toarray()
        imaginary = float(np.max(np.abs(projected.imag), initial=0.0))
        if imaginary > 1.0e-10:
            raise RuntimeError(f"translation block is not real: {imaginary:.3e}")
        projected = projected.real
        basis = basis.tocsc()
        column_parity = np.array(
            [parity[basis.indices[basis.indptr[column]]] for column in range(basis.shape[1])]
        )
        for value in (0, 1):
            mask = column_parity == value
            blocks.append(projected[np.ix_(mask, mask)])
    if sum(block.shape[0] for block in blocks) != len(states):
        raise RuntimeError("symmetry blocks do not span the microscopic basis")
    return blocks


def solve_point(
    cluster,
    jpm: float,
    jpmpm: float,
    *,
    cache_dir: Path | None = None,
    tolerance: float = 1.0e-10,
) -> dict:
    """Full microscopic spectrum, periodic band, and the M=2 winding-free band.

    The winding-free band is built from the eight exact gauge corners of the
    ``M=2`` character grid, which the manuscript shows already annihilates every
    winding process available at the orders where the artifact lives.  No extra
    sourced eigensolves are therefore needed per parameter point.

    The construction is only defined where the ice-descended band stays
    spectrally isolated and its ice-space Gram matrix stays nonsingular.  Deep
    in the triangle both fail; such a point is recorded with
    ``well_defined = False`` rather than raised, so a scan maps the boundary
    instead of stopping at it.  The band is attempted before the full spectrum
    because it is the cheaper of the two failures.
    """
    tag = f"jpm{jpm:+.6f}_pp{jpmpm:+.6f}".replace(".", "p")
    cache = None if cache_dir is None else Path(cache_dir) / f"cubic16_xyz_{tag}.npz"
    if cache is not None and cache.exists():
        with np.load(cache) as saved:
            return {key: saved[key] for key in saved.files} | {"cached": True}

    started = time.perf_counter()
    states = full_spin_basis(N_SITES)
    state_index = {int(state): index for index, state in enumerate(states)}
    ice_indices = np.asarray(
        [state_index[int(state)] for state in cluster.ice_states], dtype=np.int64
    )
    hamiltonian = microscopic_character_hamiltonian(
        cluster, states, jpm, np.zeros(3), jpmpm=jpmpm
    )
    # The zero-source Hamiltonian still commutes with the four cluster
    # translations even with Jpmpm on, so the band comes from four sparse
    # sector solves rather than one 65,536-dimensional Krylov run.
    sectors = translation_sector_bases(states, cubic16_translation_permutations(cluster))
    try:
        band = extract_exact_band(
            hamiltonian, sectors, ice_indices, cluster.n_ice, tolerance=tolerance
        )
    except (RuntimeError, ValueError) as error:
        failed = {
            "jpm": np.asarray(jpm),
            "jpmpm": np.asarray(jpmpm),
            "well_defined": np.asarray(False),
            "failure": np.asarray(str(error)),
            "solve_seconds": np.asarray(time.perf_counter() - started),
        }
        if cache is not None:
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(cache, **failed)
        return failed | {"cached": False}

    full_spectrum = np.sort(
        np.concatenate([np.linalg.eigvalsh(block) for block in _sector_blocks(sectors, states, hamiltonian)])
    )

    operators = []
    for theta in uniform_character_grid(2):
        phases = basis_character_phases(cluster, states, theta)[ice_indices]
        operators.append(phases[:, None] * band.operator * phases.conjugate()[None, :])
    winding_free = sector_project(
        character_project(np.asarray(operators)), polarization_sector_labels(cluster)
    )

    result = {
        "jpm": np.asarray(jpm),
        "jpmpm": np.asarray(jpmpm),
        "well_defined": np.asarray(True),
        "full_spectrum": full_spectrum,
        "periodic_band": np.linalg.eigvalsh(band.operator),
        "winding_free_band": np.linalg.eigvalsh(winding_free),
        "band_gap": np.asarray(band.fixed_sz_gap_above_band),
        "model_overlap_min": np.asarray(band.diagnostics["model_overlap_min"]),
        "sector_counts": np.asarray(json.dumps(band.diagnostics["sector_counts"])),
        "solve_seconds": np.asarray(time.perf_counter() - started),
    }
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, **result)
    return result | {"cached": False}


def heat_capacity_curves(
    point: dict, ja_meV: float, temperatures_K: np.ndarray
) -> dict[str, np.ndarray]:
    """Molar magnetic heat capacity, J/(K mol_Ce), on a physical temperature grid.

    Both curves replace only the exact ice-descended band and keep the
    microscopic complement, which is what makes them all-temperature results.
    """
    temperatures_K = np.asarray(temperatures_K, dtype=float)
    reduced = temperatures_K * KB_MEV_PER_K / float(ja_meV)
    full = np.asarray(point["full_spectrum"], dtype=float)
    curves = {"temperature_K": temperatures_K}
    for label, band in (
        ("periodic", np.asarray(point["periodic_band"], dtype=float)),
        ("winding_free", np.asarray(point["winding_free_band"], dtype=float)),
    ):
        spectrum = full_hilbert_counterterm_spectrum(full, band)
        thermal = thermal_observables(spectrum, reduced, n_sites=N_SITES)
        curves[label] = thermal["heat_capacity_per_site"] * R_GAS
        curves[label + "_entropy"] = thermal["entropy_per_site"] * R_GAS
    return curves


def local_maxima(temperatures: np.ndarray, values: np.ndarray) -> list[tuple[float, float]]:
    """Interior local maxima, refined by a parabola in ``log T``."""
    temperatures = np.asarray(temperatures, dtype=float)
    values = np.asarray(values, dtype=float)
    logt = np.log(temperatures)
    found = []
    for index in range(1, len(values) - 1):
        if values[index] > values[index - 1] and values[index] >= values[index + 1]:
            left, middle, right = values[index - 1 : index + 2]
            denominator = left - 2.0 * middle + right
            offset = 0.0 if denominator == 0 else 0.5 * (left - right) / denominator
            offset = float(np.clip(offset, -1.0, 1.0))
            step = logt[index + 1] - logt[index]
            found.append(
                (
                    float(np.exp(logt[index] + offset * step)),
                    float(middle - 0.25 * (left - right) * offset),
                )
            )
    return found


def two_peak_summary(temperatures: np.ndarray, values: np.ndarray) -> dict:
    """Report the two most prominent maxima as ``(T2, T1)``, low first."""
    peaks = local_maxima(temperatures, values)
    peaks.sort(key=lambda item: -item[1])
    chosen = sorted(peaks[:2])
    summary: dict[str, float | int] = {"n_maxima": len(peaks)}
    if len(chosen) >= 1:
        summary["T_low_K"], summary["C_low"] = chosen[0]
    if len(chosen) == 2:
        summary["T_high_K"], summary["C_high"] = chosen[1]
        summary["peak_ratio"] = chosen[0][0] / chosen[1][0]
    return summary
