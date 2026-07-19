#!/usr/bin/env python3
"""Test a local environment-dressed wrapping-four counterterm.

The dressed operator is

    D4 = 1/2 sum_C {N_C, W_C},

where N_C counts flippable wrapping four-loops C' sharing at least two sites
with C (excluding C itself).  This is the smallest local scalar that resolves
the two exact wrapping-four amplitudes in the weak-coupling zero-twist band.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh

import recompute_finite_size_artifact as R
from compare_counterterm_zero_dipole import (
    centered_spectral_error,
    coupling_tag,
    parse_grid,
    target_flux_fraction,
)
from twist_resolved_full_band import sz0_basis
from validate_winding_counterterm_16site import (
    flux_labels,
    local_ice_projected_ring_operator,
    low_peak,
    ring_operator,
    transfer_mixing,
    transverse_operator,
    verify_ice_band,
)

HERE = Path(__file__).resolve().parent
FIGS = HERE / "figs"
FIGS.mkdir(exist_ok=True)


def loop_flippability(
    basis: np.ndarray,
    paths: list[tuple[int, ...]],
    cluster: R.Cluster | None = None,
) -> np.ndarray:
    output = np.empty((len(paths), len(basis)), dtype=bool)
    tetrahedra_by_site = R.site_to_tets(cluster.tets) if cluster is not None else None
    for loop_index, path in enumerate(paths):
        first = (basis >> np.uint64(path[0])) & np.uint64(1)
        alternating = np.ones(len(basis), dtype=bool)
        previous = first
        for site in path[1:]:
            current = (basis >> np.uint64(site)) & np.uint64(1)
            alternating &= current != previous
            previous = current
        alternating &= previous != first
        if cluster is not None:
            touched_tetrahedra = {
                tetrahedron
                for site in path
                for tetrahedron in tetrahedra_by_site[site]
            }
            for tetrahedron in touched_tetrahedra:
                up_count = np.bitwise_count(basis & cluster.tet_masks[tetrahedron])
                alternating &= up_count == 2
        output[loop_index] = alternating
    return output


def dressed_four_operator(
    basis: np.ndarray,
    wrapping_four: list[tuple[tuple[int, ...], tuple[int, int, int]]],
    cluster: R.Cluster | None = None,
) -> sp.csr_matrix:
    paths = [tuple(loop[0]) for loop in wrapping_four]
    flippable = loop_flippability(basis, paths, cluster)
    index = {int(state): position for position, state in enumerate(basis)}
    site_sets = [set(path) for path in paths]
    neighbors = [
        [
            other
            for other in range(len(paths))
            if other != loop and len(site_sets[loop] & site_sets[other]) >= 2
        ]
        for loop in range(len(paths))
    ]
    rows = []
    columns = []
    values = []
    for loop, path in enumerate(paths):
        active_columns = np.flatnonzero(flippable[loop])
        mask = sum(1 << site for site in path)
        active_rows = np.asarray(
            [index[int(basis[column]) ^ mask] for column in active_columns],
            dtype=int,
        )
        environment = np.sum(flippable[neighbors[loop]], axis=0, dtype=float)
        amplitude = 0.5 * (
            environment[active_columns] + environment[active_rows]
        )
        rows.extend(active_rows.tolist())
        columns.extend(active_columns.tolist())
        values.extend(amplitude.tolist())
    operator = sp.coo_matrix(
        (values, (rows, columns)), shape=(len(basis), len(basis))
    ).tocsr()
    asymmetry = operator - operator.T
    if asymmetry.nnz and np.max(np.abs(asymmetry.data)) > 1.0e-12:
        raise RuntimeError("dressed wrapping-four operator is not Hermitian")
    return operator


def solve_candidate(
    h0: np.ndarray,
    transverse: sp.csr_matrix,
    w4: sp.csr_matrix,
    w6w: sp.csr_matrix,
    d4: sp.csr_matrix,
    jpm: float,
    kappa4: float,
    kappa6w: float,
    lambda4: float,
    n_low: int,
    tolerance: float,
    n_ice: int,
    v0: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, float]:
    hamiltonian = (
        sp.diags(h0)
        - jpm * transverse
        + kappa4 * w4
        + kappa6w * w6w
        + lambda4 * d4
    ).tocsr()
    last_error = None
    for attempt, count in enumerate([n_low, max(180, n_low + 40), max(220, n_low + 80)]):
        evals, evecs = eigsh(
            hamiltonian,
            k=count,
            which="SA",
            tol=tolerance,
            ncv=max(2 * count + 20, 320),
            v0=v0 if attempt == 0 else None,
        )
        order = np.argsort(evals)
        evals = np.asarray(evals[order], dtype=float)
        evecs = np.asarray(evecs[:, order], dtype=float)
        try:
            return evals, evecs, verify_ice_band(evals, n_ice)
        except RuntimeError as error:
            last_error = error
            print("    retrying incomplete ARPACK band", flush=True)
    raise RuntimeError("unable to resolve complete ice band") from last_error


def exact_second_moment(
    h0: np.ndarray,
    transverse: sp.csr_matrix,
    w4: sp.csr_matrix,
    w6w: sp.csr_matrix,
    d4: sp.csr_matrix,
    jpm: float,
    kappa4: float,
    kappa6w: float,
    lambda4: float,
) -> float:
    off_diagonal = (
        -jpm * transverse + kappa4 * w4 + kappa6w * w6w + lambda4 * d4
    )
    mean = float(np.mean(h0))
    trace_h2 = float(np.dot(h0, h0) + np.sum(np.abs(off_diagonal.data) ** 2))
    return trace_h2 / len(h0) - mean * mean


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--couplings", default="-0.03,-0.05")
    parser.add_argument("--ratio6-grid", default="-12.0:0.0:5")
    parser.add_argument("--dressed-ratio-grid", default="-24.0:0.0:5")
    parser.add_argument("--n-low", type=int, default=140)
    parser.add_argument("--tol", type=float, default=1.0e-7)
    parser.add_argument(
        "--local-ice-projected",
        action="store_true",
        help="condition W4, W6w, and D4 on the ice rule for touching tetrahedra",
    )
    parser.add_argument(
        "--calibration-json",
        type=Path,
        default=HERE / "counterterm_coupling_sweep.json",
    )
    parser.add_argument(
        "--out", type=Path, default=HERE / "dressed_counterterm_vs_zero_dipole.npz"
    )
    args = parser.parse_args()

    couplings = [float(value) for value in args.couplings.split(",")]
    ratio6_values = np.unique(
        np.concatenate([parse_grid(args.ratio6_grid), np.asarray([0.0])])
    )
    dressed_ratios = np.unique(
        np.concatenate([parse_grid(args.dressed_ratio_grid), np.asarray([0.0])])
    )
    calibration = json.loads(args.calibration_json.read_text())
    calibrated = {
        float(result["jpm"]): result for result in calibration["results"]
    }

    cluster = R.build_cluster("cubic", (1, 1, 1))
    wrapping_four = [
        loop for loop in cluster.loops4 if tuple(loop[1]) != (0, 0, 0)
    ]
    wrapping_hexagons = [
        loop for loop in cluster.hexes if tuple(loop[1]) != (0, 0, 0)
    ]
    basis = sz0_basis(cluster.n_sites)
    basis_index = {int(state): index for index, state in enumerate(basis)}
    ice_indices = np.asarray(
        [basis_index[int(state)] for state in cluster.ice_states], dtype=int
    )
    labels = flux_labels(cluster)
    h0 = R.ising_energy(cluster, basis)
    transverse = transverse_operator(cluster, basis)
    ring_builder = (
        lambda states, loops: local_ice_projected_ring_operator(
            cluster, states, loops
        )
        if args.local_ice_projected
        else ring_operator(states, loops)
    )
    dressed_cluster = cluster if args.local_ice_projected else None
    w4, _ = ring_builder(basis, wrapping_four)
    w6w, _ = ring_builder(basis, wrapping_hexagons)
    d4 = dressed_four_operator(basis, wrapping_four, dressed_cluster)

    if args.local_ice_projected:
        bare_ice_d4 = dressed_four_operator(cluster.ice_states, wrapping_four)
        projected_ice_d4 = dressed_four_operator(
            cluster.ice_states, wrapping_four, cluster
        )
        if (bare_ice_d4 != projected_ice_d4).nnz:
            raise RuntimeError("local projection changed an ice-manifold D4 element")

    full_basis = np.arange(1 << cluster.n_sites, dtype=np.uint64)
    full_h0 = R.ising_energy(cluster, full_basis)
    full_transverse = transverse_operator(cluster, full_basis)
    full_w4, _ = ring_builder(full_basis, wrapping_four)
    full_w6w, _ = ring_builder(full_basis, wrapping_hexagons)
    full_d4 = dressed_four_operator(full_basis, wrapping_four, dressed_cluster)

    results = []
    spectra = {}
    for jpm in couplings:
        target_data = np.load(
            HERE / f"twist_resolved_qed_dipole2_M2_{coupling_tag(jpm)}.npz"
        )
        target_h = np.asarray(target_data["H_qed_twist_avg"], dtype=complex)
        target_evals, target_evecs = np.linalg.eigh(target_h)
        target_evals = target_evals.real
        target_summary = json.loads(str(target_data["summary"]))
        target_peak = float(target_summary["Tpk_qed_twist_operator_avg"])
        betas = np.asarray(calibrated[jpm]["betas"], dtype=float)
        target_mixing = transfer_mixing(
            target_evals,
            target_evecs,
            np.arange(cluster.n_ice),
            labels,
            betas,
        )
        ratio4 = float(calibrated[jpm]["best"]["kappa4_over_jpm2"])
        kappa4 = ratio4 * jpm * jpm
        bare_m2 = exact_second_moment(
            full_h0,
            full_transverse,
            full_w4,
            full_w6w,
            full_d4,
            jpm,
            0.0,
            0.0,
            0.0,
        )
        temperatures = np.geomspace(1.0e-7, 0.08, 1200)
        print(
            f"Jpm={jpm:+.3f}: r4={ratio4:.3f}, "
            f"target transfer={np.mean(target_mixing):.3e}",
            flush=True,
        )
        records = []
        v0 = None
        for ratio6 in ratio6_values:
            for dressed_ratio in dressed_ratios:
                kappa6w = ratio6 * abs(jpm) ** 3
                lambda4 = dressed_ratio * abs(jpm) ** 4
                started = time.time()
                evals, evecs, defect_gap = solve_candidate(
                    h0,
                    transverse,
                    w4,
                    w6w,
                    d4,
                    jpm,
                    kappa4,
                    kappa6w,
                    lambda4,
                    args.n_low,
                    args.tol,
                    cluster.n_ice,
                    v0,
                )
                v0 = evecs[:, 0]
                band = evals[: cluster.n_ice]
                mixing = transfer_mixing(evals, evecs, ice_indices, labels, betas)
                m2 = exact_second_moment(
                    full_h0,
                    full_transverse,
                    full_w4,
                    full_w6w,
                    full_d4,
                    jpm,
                    kappa4,
                    kappa6w,
                    lambda4,
                )
                record = {
                    "ratio4": ratio4,
                    "ratio6": float(ratio6),
                    "dressed_ratio": float(dressed_ratio),
                    "kappa4": float(kappa4),
                    "kappa6w": float(kappa6w),
                    "lambda4": float(lambda4),
                    "spectral_error": centered_spectral_error(band, target_evals),
                    "mixing": [float(value) for value in mixing],
                    "mixing_mean": float(np.mean(mixing)),
                    "low_heat_peak": low_peak(evals, temperatures, cluster.n_ice),
                    "defect_gap": defect_gap,
                    "m2_relative_change": float((m2 - bare_m2) / bare_m2),
                    "elapsed_s": float(time.time() - started),
                }
                records.append(record)
                spectra[(jpm, float(ratio6), float(dressed_ratio))] = band
                print(
                    f"  r6={ratio6:+.1f} rd={dressed_ratio:+.1f}: "
                    f"spec={record['spectral_error']:.4f} "
                    f"mix={record['mixing_mean']:.4f} "
                    f"Tpk/T0={record['low_heat_peak'] / target_peak:.3f}",
                    flush=True,
                )
        best = min(records, key=lambda record: record["spectral_error"])
        baseline = min(
            (
                record
                for record in records
                if record["dressed_ratio"] == 0.0
            ),
            key=lambda record: abs(record["ratio6"] + 6.0),
        )
        results.append(
            {
                "jpm": jpm,
                "target_peak": target_peak,
                "target_mixing": [float(value) for value in target_mixing],
                "target_mixing_mean": float(np.mean(target_mixing)),
                "target_flux_fraction": target_flux_fraction(target_h, labels),
                "baseline_two_term": baseline,
                "individual_best": best,
                "records": records,
            }
        )
        spectra[(jpm, "target")] = target_evals

    record_maps = [
        {
            (round(record["ratio6"], 8), round(record["dressed_ratio"], 8)): record
            for record in result["records"]
        }
        for result in results
    ]
    common = set(record_maps[0])
    for record_map in record_maps[1:]:
        common &= set(record_map)
    shared_key = min(
        common,
        key=lambda key: np.mean(
            [record_map[key]["spectral_error"] for record_map in record_maps]
        ),
    )
    shared_records = [record_map[shared_key] for record_map in record_maps]
    shared = {
        "ratio6": shared_key[0],
        "dressed_ratio": shared_key[1],
        "mean_spectral_error": float(
            np.mean([record["spectral_error"] for record in shared_records])
        ),
        "mean_mixing": float(
            np.mean([record["mixing_mean"] for record in shared_records])
        ),
        "by_coupling": shared_records,
    }
    summary = {
        "method": (
            "full microscopic ED with locally ice-projected, environment-dressed W4"
            if args.local_ice_projected
            else "full microscopic ED with local environment-dressed W4"
        ),
        "local_ice_projected": bool(args.local_ice_projected),
        "dressed_operator": "D4=1/2 sum_C {N_C,W_C}; N_C counts flippable wrapping C' sharing >=2 sites",
        "selection_objective": "mean centered 90-state spectral error across couplings",
        "best_shared": shared,
        "results": results,
    }
    args.out.with_suffix(".json").write_text(json.dumps(summary, indent=2))
    arrays = {"summary": np.asarray(json.dumps(summary))}
    for result, shared_record in zip(results, shared_records):
        jpm = result["jpm"]
        tag = coupling_tag(jpm)
        baseline = result["baseline_two_term"]
        arrays[f"{tag}_E_target"] = spectra[(jpm, "target")]
        arrays[f"{tag}_E_baseline"] = spectra[
            (jpm, baseline["ratio6"], baseline["dressed_ratio"])
        ]
        arrays[f"{tag}_E_dressed"] = spectra[
            (jpm, shared_record["ratio6"], shared_record["dressed_ratio"])
        ]
    np.savez_compressed(args.out, **arrays)

    fig, axes = plt.subplots(1, len(results), figsize=(4.2 * len(results), 3.45))
    if len(results) == 1:
        axes = [axes]
    for axis, result in zip(axes, results):
        tag = coupling_tag(result["jpm"])
        target_peak = result["target_peak"]
        temperatures = np.geomspace(target_peak / 20.0, target_peak * 20.0, 700)
        for key, color, style, label in (
            ("target", "black", "-", "zero dipole"),
            ("baseline", "#d55e00", "--", r"$W_4+W_{6w}$"),
            ("dressed", "#007c83", "-", r"dressed $W_4+W_{6w}$"),
        ):
            axis.plot(
                temperatures / target_peak,
                R.specific_heat(arrays[f"{tag}_E_{key}"], temperatures)
                / cluster.n_sites,
                color=color,
                ls=style,
                lw=1.7,
                label=label,
            )
        axis.set_xscale("log")
        axis.set_title(rf"$J_\pm/J_{{zz}}={result['jpm']:+.2f}$")
        axis.set_xlabel(r"$T/T_{\delta=0}$")
        axis.set_ylabel(r"$C/N$")
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    figure_stem = f"fig_{args.out.stem}"
    for extension in ("pdf", "png"):
        fig.savefig(FIGS / f"{figure_stem}.{extension}", dpi=230)
    plt.close(fig)

    print(f"shared best: r6={shared_key[0]:+.1f}, rd={shared_key[1]:+.1f}")
    print(f"wrote {args.out}")
    print(f"wrote {args.out.with_suffix('.json')}")
    print(f"wrote {FIGS / (figure_stem + '.pdf')}")


if __name__ == "__main__":
    main()
