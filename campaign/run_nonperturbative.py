#!/usr/bin/env python3
"""Run nonperturbative cubic-16 exact-band character projections."""

from __future__ import annotations

import argparse
import json
import sys
import time
from itertools import product
from pathlib import Path

import numpy as np
from scipy.sparse import diags

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "notes"))

import recompute_finite_size_artifact as geometry  # noqa: E402
from qsi_campaign.benchmarks import (  # noqa: E402
    load_digitized_thermodynamics,
    log_grid_rmse,
)
from qsi_campaign.exact_band import (  # noqa: E402
    basis_character_phases,
    cubic16_translation_permutations,
    extract_exact_band,
    extract_exact_band_full,
    fixed_magnetization_basis,
    full_spin_basis,
    microscopic_character_hamiltonian,
    translation_sector_bases,
    uniform_character_grid,
)
from qsi_campaign.point_group import (  # noqa: E402
    SpaceGroupOperation,
    character_orbits,
    ice_permutations,
    is_gauge_corner,
    realised_space_group,
    reconstruct_operator,
)
from qsi_campaign.protocol import (  # noqa: E402
    centered_relative_error,
    character_project,
    full_hilbert_counterterm_spectrum,
)
from qsi_campaign.thermodynamics import peak_in_window, thermal_observables  # noqa: E402


OUTPUT = ROOT / "campaign" / "outputs"
POINT_CACHE = ROOT / "campaign" / "cache" / "nonperturbative_points"
OUTPUT.mkdir(parents=True, exist_ok=True)
POINT_CACHE.mkdir(parents=True, exist_ok=True)


def coupling_tag(jpm: float) -> str:
    return f"{jpm:+.6f}".replace("+", "p").replace("-", "m").replace(".", "p")


def model_tag(jpm: float, jpmpm: float) -> str:
    tag = coupling_tag(jpm)
    if jpmpm != 0.0:
        tag += f"_pp{coupling_tag(jpmpm)}"
    return tag


def exact_m2_gauge_orbit(
    cluster, states, ice_indices, microscopic_zero, band_zero, jpm, jpmpm
):
    operators = []
    diagnostics = []
    for index, theta in enumerate(uniform_character_grid(2), start=1):
        full_phases = basis_character_phases(cluster, states, theta)
        full_unitary = diags(full_phases, format="csc")
        microscopic_corner = microscopic_character_hamiltonian(
            cluster, states, jpm, theta, jpmpm=jpmpm
        )
        gauge_corner = full_unitary @ microscopic_zero @ full_unitary.conj().T
        residual = microscopic_corner - gauge_corner
        gauge_residual = float(np.max(np.abs(residual.data), initial=0.0))
        if gauge_residual > 1.0e-11:
            raise RuntimeError(
                f"M=2 gauge identity failed at corner {index}: {gauge_residual:.3e}"
            )
        ice_phases = full_phases[ice_indices]
        operators.append(
            ice_phases[:, None]
            * band_zero
            * ice_phases.conjugate()[None, :]
        )
        diagnostics.append(
            {
                "theta": theta.tolist(),
                "microscopic_gauge_identity_max_residual": gauge_residual,
            }
        )
        print(
            f"M=2 corner {index}/8 theta/2pi="
            f"{tuple(float(value / (2 * np.pi)) for value in theta)} "
            f"gauge_residual={gauge_residual:.3e}",
            flush=True,
        )
    return character_project(np.asarray(operators)), np.asarray(operators), diagnostics


def solve_or_load_point(
    cluster, states, ice_indices, jpm, jpmpm, grid_size, index, tolerance
):
    tag = model_tag(jpm, jpmpm)
    cache = POINT_CACHE / (
        f"cubic16_{tag}_M{grid_size}_{index[0]}{index[1]}{index[2]}.npz"
    )
    if cache.exists():
        saved = np.load(cache)
        return (
            np.asarray(saved["operator"]),
            np.asarray(saved["spectrum"]),
            json.loads(str(saved["diagnostics"])),
            True,
        )
    theta = 2.0 * np.pi * np.asarray(index, dtype=float) / grid_size
    started = time.perf_counter()
    hamiltonian = microscopic_character_hamiltonian(
        cluster, states, jpm, theta, jpmpm=jpmpm
    )
    result = extract_exact_band_full(
        hamiltonian,
        ice_indices,
        cluster.n_ice,
        tolerance=tolerance,
    )
    negative = microscopic_character_hamiltonian(
        cluster, states, jpm, -theta, jpmpm=jpmpm
    )
    conjugation_residual = float(
        np.max(np.abs((negative - hamiltonian.conjugate()).data), initial=0.0)
    )
    diagnostics = {
        **result.diagnostics,
        "index": list(index),
        "theta": theta.tolist(),
        "fixed_sz_gap_above_band": result.fixed_sz_gap_above_band,
        "conjugation_identity_max_residual": conjugation_residual,
        "solve_seconds": time.perf_counter() - started,
    }
    np.savez_compressed(
        cache,
        operator=result.operator,
        spectrum=result.eigenvalues,
        diagnostics=json.dumps(diagnostics),
    )
    return result.operator, result.eigenvalues, diagnostics, False


def exact_character_grid(
    cluster, states, ice_indices, jpm, jpmpm, tolerance, zero_operator, grid_size
):
    """Solve one source point per space-group orbit and rebuild the whole grid.

    Operators are returned in lexicographic ``product(range(M), repeat=3)``
    order, matching :func:`uniform_character_grid`.
    """
    grid = list(product(range(grid_size), repeat=3))
    operators: dict[tuple[int, ...], np.ndarray] = {}
    diagnostics = []

    # Self-conjugate points carry a pure-gauge source: the band there is a
    # diagonal transformation of the zero-source band and costs no solve.
    for index in grid:
        if is_gauge_corner(index, grid_size):
            theta = 2.0 * np.pi * np.asarray(index, dtype=float) / grid_size
            ice_phases = basis_character_phases(cluster, cluster.ice_states, theta)
            operators[index] = (
                ice_phases[:, None] * zero_operator * ice_phases.conjugate()[None, :]
            )
            diagnostics.append({"index": list(index), "kind": "exact_gauge_corner"})

    if jpmpm == 0.0:
        group = realised_space_group(cluster)
        reduction = f"space group ({len(group)} operations) and conjugation"
    else:
        # The pair-flip phase depends on r_i + r_j, which is not covariant under
        # the operations carrying a non-primitive translation.  Fall back to the
        # conjugation-only pairing.
        group = [SpaceGroupOperation(np.eye(3), np.arange(cluster.n_sites))]
        reduction = "conjugation only (Jpmpm != 0 breaks space-group covariance)"
    ice_permutation_by_operation = ice_permutations(cluster, group)
    representatives, recipes = character_orbits(grid_size, group)
    sourced = [index for index in representatives if not is_gauge_corner(index, grid_size)]
    print(
        f"M={grid_size}: {grid_size ** 3} character points reduced by {reduction} "
        f"to {len(operators)} gauge corners and {len(sourced)} solves",
        flush=True,
    )

    for count, index in enumerate(sourced, start=1):
        operator, _, point_diagnostics, cached = solve_or_load_point(
            cluster,
            states,
            ice_indices,
            jpm,
            jpmpm,
            grid_size,
            index,
            tolerance,
        )
        operators[index] = operator
        diagnostics.append(point_diagnostics)
        timing = "cached" if cached else f"{point_diagnostics['solve_seconds']:.2f}s"
        print(
            f"M={grid_size} orbit {count}/{len(sourced)} "
            f"index={index} "
            f"{timing} "
            f"gap={point_diagnostics['fixed_sz_gap_above_band']:.6g} "
            f"overlap_min={point_diagnostics['model_overlap_min']:.6g}",
            flush=True,
        )

    for index in grid:
        if index in operators:
            continue
        representative, _, _ = recipes[index]
        operators[index] = reconstruct_operator(
            operators[representative], recipes[index], ice_permutation_by_operation
        )
    if len(operators) != grid_size**3:
        raise RuntimeError(
            f"M={grid_size} orbit produced {len(operators)} operators"
        )
    ordered = np.asarray([operators[index] for index in grid])
    return character_project(ordered), ordered, diagnostics


def thermodynamic_metrics(
    clean_spectrum,
    full_spectrum,
    temperatures,
    qmc_temperature,
    qmc_heat,
    qmc_entropy,
):
    corrected = full_hilbert_counterterm_spectrum(full_spectrum, clean_spectrum)
    thermal = thermal_observables(corrected, temperatures, n_sites=16)
    low_mask = qmc_temperature <= 2.0e-2
    return corrected, thermal, {
        "low_temperature_heat_rmse": log_grid_rmse(
            qmc_temperature[low_mask],
            qmc_heat[low_mask],
            temperatures,
            thermal["heat_capacity_per_site"],
        ),
        "entropy_rmse": log_grid_rmse(
            qmc_temperature,
            qmc_entropy,
            temperatures,
            thermal["entropy_per_site"],
        ),
        "low_peak": peak_in_window(
            temperatures, thermal["heat_capacity_per_site"], 5.0e-4, 3.0e-2
        ),
        "high_peak": peak_in_window(
            temperatures, thermal["heat_capacity_per_site"], 5.0e-2, 1.0
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jpm", type=float, default=0.046)
    parser.add_argument("--jpmpm", type=float, default=0.0)
    parser.add_argument("--max-grid", type=int, choices=(2, 3, 4), default=2)
    parser.add_argument("--tolerance", type=float, default=1.0e-10)
    args = parser.parse_args()
    cluster = geometry.build_cluster("cubic", (1, 1, 1))
    if args.jpmpm == 0.0:
        states = fixed_magnetization_basis(cluster.n_sites, cluster.n_sites // 2)
        basis_description = "fixed-Sz=0"
    else:
        states = full_spin_basis(cluster.n_sites)
        basis_description = "full Hilbert"
    state_index = {int(state): index for index, state in enumerate(states)}
    ice_indices = np.asarray(
        [state_index[int(state)] for state in cluster.ice_states], dtype=np.int64
    )
    sectors = translation_sector_bases(
        states, cubic16_translation_permutations(cluster)
    )
    sector_dimensions = {name: sector.shape[1] for name, sector in sectors.items()}
    print(
        f"cubic-16 exact campaign: {basis_description} dimension={len(states)}, "
        f"ice dimension={cluster.n_ice}, sectors={sector_dimensions}",
        flush=True,
    )

    started = time.perf_counter()
    microscopic_zero = microscopic_character_hamiltonian(
        cluster, states, args.jpm, np.zeros(3), jpmpm=args.jpmpm
    )
    exact_zero = extract_exact_band(
        microscopic_zero,
        sectors,
        ice_indices,
        cluster.n_ice,
        tolerance=args.tolerance,
    )
    zero_seconds = time.perf_counter() - started
    print(
        f"exact zero-source band: {zero_seconds:.2f}s, "
        f"gap above band={exact_zero.fixed_sz_gap_above_band:.6g}, "
        f"overlap_min={exact_zero.diagnostics['model_overlap_min']:.6g}",
        flush=True,
    )

    m2_operator, m2_corners, m2_diagnostics = exact_m2_gauge_orbit(
        cluster,
        states,
        ice_indices,
        microscopic_zero,
        exact_zero.operator,
        args.jpm,
        args.jpmpm,
    )
    operators = {1: exact_zero.operator, 2: m2_operator}
    operators_by_grid = {2: m2_corners}
    grid_diagnostics = {2: m2_diagnostics}
    for grid_size in range(3, args.max_grid + 1):
        grid_operator, grid_points, diagnostics = exact_character_grid(
            cluster,
            states,
            ice_indices,
            args.jpm,
            args.jpmpm,
            args.tolerance,
            exact_zero.operator,
            grid_size,
        )
        operators[grid_size] = grid_operator
        operators_by_grid[grid_size] = grid_points
        grid_diagnostics[grid_size] = diagnostics

    temperatures = np.geomspace(1.0e-4, 2.0, 1400)
    qmc_matched = args.jpmpm == 0.0 and np.isclose(args.jpm, 0.046)
    full_spectrum = None
    bare_full = None
    qmc_temperature = qmc_heat = qmc_entropy = None
    bare_rmse = None
    if qmc_matched:
        full_cache = ROOT / "campaign" / "cache" / "full_ed_cubic16_jpm_0p046.npz"
        if not full_cache.exists():
            raise FileNotFoundError(
                f"missing exact microscopic complement: {full_cache}"
            )
        full_spectrum = np.asarray(np.load(full_cache)["E_full"], dtype=float)
        bare_full = thermal_observables(full_spectrum, temperatures, n_sites=16)
        qmc_temperature, qmc_heat, qmc_entropy = load_digitized_thermodynamics(
            ROOT / "campaign" / "data" / "huang_2018_qmc_jpm_0p046.csv"
        )
        low_mask = qmc_temperature <= 2.0e-2
        bare_rmse = log_grid_rmse(
            qmc_temperature[low_mask],
            qmc_heat[low_mask],
            temperatures,
            bare_full["heat_capacity_per_site"],
        )

    report = {
        "method": "exact microscopic bands, polar pullback, primitive q=2*delta character",
        "qualification": (
            "nonperturbative finite-character calculation; bulk use requires "
            "M convergence and cluster convergence"
        ),
        "jpm": args.jpm,
        "jpmpm": args.jpmpm,
        "n_sites": cluster.n_sites,
        "basis": basis_description,
        "basis_dimension": len(states),
        "band_dimension": cluster.n_ice,
        "sector_dimensions_at_zero_source": sector_dimensions,
        "zero_source": {
            **exact_zero.diagnostics,
            "gap_above_band": exact_zero.fixed_sz_gap_above_band,
            "solve_seconds": zero_seconds,
        },
        "grids": {
            "M2": {
                "n_characters": 8,
                "maximum_microscopic_gauge_identity_residual": max(
                    item["microscopic_gauge_identity_max_residual"]
                    for item in m2_diagnostics
                ),
            }
        },
        "convergence": {
            "M1_to_M2_centered_operator_relative_error": centered_relative_error(
                operators[1], operators[2]
            )
        },
        "qmc_thermodynamics": (
            {
                "source": "Huang et al., PRL 120, 167202 (2018), Fig. 1(b)",
                "bare_low_temperature_heat_rmse": bare_rmse,
            }
            if qmc_matched
            else {
                "status": "not_applicable",
                "reason": "the available QMC curve has J_pm_pm=0 and J_pm/J_zz=0.046",
            }
        ),
        "pending": {},
    }
    if args.jpmpm == 0.0:
        report["pending"]["full_hilbert_gap"] = (
            "reported gaps are within Sz=0; other conserved magnetization "
            "sectors are required for an unrestricted spectral gap"
        )
    if not qmc_matched:
        report["pending"]["microscopic_complement"] = (
            "only selected-band thermodynamics are emitted; a matching full-spectrum "
            "or stochastic complement is required for all-temperature observables"
        )
    for grid_size in range(3, args.max_grid + 1):
        solved = [
            item
            for item in grid_diagnostics[grid_size]
            if item.get("kind") != "exact_gauge_corner"
        ]
        report["grids"][f"M{grid_size}"] = {
            "n_characters": grid_size**3,
            "n_solved_representatives": len(solved),
            "n_exact_gauge_corners": grid_size**3 - 2 * len(solved),
            "minimum_gap_above_band": min(
                item["fixed_sz_gap_above_band"] for item in solved
            ),
            "minimum_model_overlap": min(
                item["model_overlap_min"] for item in solved
            ),
            "maximum_ritz_residual": max(
                item["maximum_ritz_residual"] for item in solved
            ),
            "maximum_conjugation_identity_residual": max(
                item["conjugation_identity_max_residual"] for item in solved
            ),
        }
        report["convergence"][
            f"M{grid_size - 1}_to_M{grid_size}_centered_operator_relative_error"
        ] = centered_relative_error(
            operators[grid_size - 1], operators[grid_size]
        )

    arrays = {
        "temperature": temperatures,
        "zero_source_exact_band_spectrum": exact_zero.eigenvalues,
    }
    if qmc_matched:
        arrays.update(
            {
                "bare_full_heat_capacity_per_site": bare_full[
                    "heat_capacity_per_site"
                ],
                "bare_full_entropy_per_site": bare_full["entropy_per_site"],
                "qmc_temperature": qmc_temperature,
                "qmc_heat_capacity_per_site": qmc_heat,
                "qmc_entropy_per_site": qmc_entropy,
            }
        )
    for grid, operator in operators.items():
        spectrum = np.linalg.eigvalsh(operator)
        band_thermal = thermal_observables(spectrum, temperatures, n_sites=16)
        arrays[f"M{grid}_operator"] = operator
        arrays[f"M{grid}_spectrum"] = spectrum
        arrays[f"M{grid}_band_heat_capacity_per_site"] = band_thermal[
            "heat_capacity_per_site"
        ]
        if grid >= 2:
            arrays[f"M{grid}_operators_by_character"] = operators_by_grid[grid]
            # Record the grid index of every stored operator.  The array is in
            # lexicographic order, but saying so explicitly keeps any consumer
            # from having to infer it.
            arrays[f"M{grid}_character_indices"] = np.asarray(
                list(product(range(grid), repeat=3)), dtype=np.int64
            )
        if grid >= 2 and qmc_matched:
            replaced, full_thermal, metrics = thermodynamic_metrics(
                spectrum,
                full_spectrum,
                temperatures,
                qmc_temperature,
                qmc_heat,
                qmc_entropy,
            )
            metrics["heat_rmse_reduction_fraction"] = (
                1.0 - metrics["low_temperature_heat_rmse"] / bare_rmse
            )
            report["qmc_thermodynamics"][f"M{grid}"] = metrics
            arrays[f"M{grid}_full_spectrum"] = replaced
            arrays[f"M{grid}_full_heat_capacity_per_site"] = full_thermal[
                "heat_capacity_per_site"
            ]
            arrays[f"M{grid}_full_entropy_per_site"] = full_thermal[
                "entropy_per_site"
            ]
    final_grid = max(operators)
    final_metrics = (
        report["qmc_thermodynamics"][f"M{final_grid}"] if qmc_matched else None
    )
    character_error = None
    if final_grid >= 3:
        character_error = report["convergence"][
            f"M{final_grid - 1}_to_M{final_grid}_centered_operator_relative_error"
        ]
    report["gates"] = {
        "exact_band_overlap": {
            "status": "pass"
            if report["zero_source"]["model_overlap_min"] > 1.0e-8
            else "fail",
            "singularity_cutoff": 1.0e-8,
        },
        "character_convergence": {
            "status": "pass"
            if character_error is not None and character_error < 0.05
            else "pending" if character_error is None else "fail",
            "threshold": 0.05,
            "error": character_error,
        },
        "qmc_heat_capacity": {
            "status": (
                "pass"
                if qmc_matched and final_metrics["low_temperature_heat_rmse"] < 0.02
                else "fail" if qmc_matched else "not_applicable"
            ),
            "threshold": 0.02,
            "rmse": final_metrics["low_temperature_heat_rmse"] if qmc_matched else None,
        },
        "qmc_entropy": {
            "status": (
                "pass"
                if qmc_matched and final_metrics["entropy_rmse"] < 0.02
                else "fail" if qmc_matched else "not_applicable"
            ),
            "threshold": 0.02,
            "rmse": final_metrics["entropy_rmse"] if qmc_matched else None,
        },
        "fcc32_full_hamiltonian": {
            "status": "pending",
            "reason": "current exact-band campaign is cubic-16 only",
        },
        "qmc_sac_dynamics": {
            "status": "pending",
            "reason": "matched finite-cluster momenta and spectral normalization required",
        },
    }
    if args.max_grid < 3:
        report["pending"]["character_convergence"] = (
            "run --max-grid 3 for the first genuine boundary-twist comparison"
        )
    stem = f"nonperturbative_cubic16_{model_tag(args.jpm, args.jpmpm)}"
    np.savez_compressed(OUTPUT / f"{stem}.npz", **arrays)
    (OUTPUT / f"{stem}.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"wrote {OUTPUT / f'{stem}.npz'}")
    print(f"wrote {OUTPUT / f'{stem}.json'}")


if __name__ == "__main__":
    main()
