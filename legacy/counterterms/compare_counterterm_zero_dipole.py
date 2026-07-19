#!/usr/bin/env python3
"""Benchmark microscopic winding counterterms against zero-dipole cleaning.

This is a weak-coupling validation only.  The zero-dipole character-projected
band supplies the target spectrum, while every counterterm candidate is solved
in the full microscopic cubic-16 Sz=0 Hilbert space.
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
from twist_resolved_full_band import sz0_basis
from validate_winding_counterterm_16site import (
    flux_labels,
    global_ice_projected_ring_operator,
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


def parse_grid(text: str) -> np.ndarray:
    start, stop, count = text.split(":")
    return np.linspace(float(start), float(stop), int(count))


def coupling_tag(jpm: float) -> str:
    return f"jm{abs(jpm):.2f}".replace(".", "p")


def centered_spectral_error(candidate: np.ndarray, target: np.ndarray) -> float:
    candidate = np.sort(candidate) - np.mean(candidate)
    target = np.sort(target) - np.mean(target)
    scale = float(np.std(target))
    return float(np.sqrt(np.mean((candidate - target) ** 2)) / scale)


def exact_second_moment(
    diagonal: np.ndarray,
    transverse: sp.csr_matrix,
    w4: sp.csr_matrix,
    w6w: sp.csr_matrix,
    jpm: float,
    kappa4: float,
    kappa6w: float,
) -> float:
    off_diagonal = -jpm * transverse + kappa4 * w4 + kappa6w * w6w
    mean = float(np.mean(diagonal))
    trace_h2 = float(
        np.dot(diagonal, diagonal) + np.sum(np.abs(off_diagonal.data) ** 2)
    )
    return trace_h2 / len(diagonal) - mean * mean


def solve_candidate(
    h0: np.ndarray,
    transverse: sp.csr_matrix,
    w4: sp.csr_matrix,
    w6w: sp.csr_matrix,
    jpm: float,
    kappa4: float,
    kappa6w: float,
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
    ).tocsr()
    attempts = [n_low, max(180, n_low + 40), max(220, n_low + 80)]
    last_error: RuntimeError | None = None
    for attempt, count in enumerate(attempts):
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
            defect_gap = verify_ice_band(evals, n_ice)
            return evals, evecs, defect_gap
        except RuntimeError as error:
            last_error = error
            print(
                f"    retrying incomplete ARPACK band with k={attempts[min(attempt + 1, len(attempts) - 1)]}",
                flush=True,
            )
    raise RuntimeError("unable to resolve complete ice band") from last_error


def target_flux_fraction(operator: np.ndarray, labels: np.ndarray) -> float:
    centered = operator - np.eye(len(operator)) * np.trace(operator) / len(operator)
    different = labels[:, None] != labels[None, :]
    norm2 = float(np.sum(np.abs(centered) ** 2))
    return float(np.sum(np.abs(centered[different]) ** 2) / norm2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--couplings", default="-0.03,-0.05")
    parser.add_argument("--ratio4-offset-grid", default="-0.75:0.75:5")
    parser.add_argument("--ratio6-grid", default="-24.0:12.0:7")
    parser.add_argument("--n-low", type=int, default=140)
    parser.add_argument("--tol", type=float, default=1.0e-7)
    parser.add_argument(
        "--local-ice-projected",
        action="store_true",
        help="condition each counterterm flip on the ice rule for touching tetrahedra",
    )
    parser.add_argument(
        "--global-ice-projected",
        action="store_true",
        help="condition each counterterm flip on the ice rule for every tetrahedron",
    )
    parser.add_argument(
        "--calibration-json",
        type=Path,
        default=HERE / "counterterm_coupling_sweep.json",
    )
    parser.add_argument(
        "--out", type=Path, default=HERE / "counterterm_vs_zero_dipole.npz"
    )
    args = parser.parse_args()
    if args.local_ice_projected and args.global_ice_projected:
        parser.error("choose at most one counterterm projection scope")

    couplings = [float(value) for value in args.couplings.split(",")]
    ratio4_offsets = parse_grid(args.ratio4_offset_grid)
    ratio6_values = np.unique(
        np.concatenate([parse_grid(args.ratio6_grid), np.asarray([0.0])])
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
    if len(wrapping_four) != 36 or len(wrapping_hexagons) != 48:
        raise RuntimeError("unexpected cubic-16 wrapping-loop census")

    basis = sz0_basis(cluster.n_sites)
    basis_index = {int(state): index for index, state in enumerate(basis)}
    ice_indices = np.asarray(
        [basis_index[int(state)] for state in cluster.ice_states], dtype=int
    )
    labels = flux_labels(cluster)
    h0 = R.ising_energy(cluster, basis)
    transverse = transverse_operator(cluster, basis)
    if args.global_ice_projected:
        ring_builder = lambda states, loops: global_ice_projected_ring_operator(
            cluster, states, loops
        )
    elif args.local_ice_projected:
        ring_builder = lambda states, loops: local_ice_projected_ring_operator(
            cluster, states, loops
        )
    else:
        ring_builder = ring_operator
    w4, _ = ring_builder(basis, wrapping_four)
    w6w, _ = ring_builder(basis, wrapping_hexagons)

    if args.local_ice_projected or args.global_ice_projected:
        ice_w4, _ = ring_operator(cluster.ice_states, wrapping_four)
        ice_w6w, _ = ring_operator(cluster.ice_states, wrapping_hexagons)
        projected_ice_w4, _ = ring_builder(cluster.ice_states, wrapping_four)
        projected_ice_w6w, _ = ring_builder(cluster.ice_states, wrapping_hexagons)
        if (ice_w4 != projected_ice_w4).nnz or (ice_w6w != projected_ice_w6w).nnz:
            raise RuntimeError("local projection changed an ice-manifold matrix element")

    full_basis = np.arange(1 << cluster.n_sites, dtype=np.uint64)
    full_h0 = R.ising_energy(cluster, full_basis)
    full_transverse = transverse_operator(cluster, full_basis)
    full_w4, _ = ring_builder(full_basis, wrapping_four)
    full_w6w, _ = ring_builder(full_basis, wrapping_hexagons)

    results = []
    saved_spectra = {}
    for jpm in couplings:
        if jpm not in calibrated:
            raise ValueError(f"no one-term calibration for Jpm={jpm}")
        target_path = HERE / f"twist_resolved_qed_dipole2_M2_{coupling_tag(jpm)}.npz"
        target_data = np.load(target_path)
        target_h = np.asarray(target_data["H_qed_twist_avg"], dtype=complex)
        target_evals, target_evecs = np.linalg.eigh(target_h)
        target_evals = target_evals.real
        target_peak = float(
            json.loads(str(target_data["summary"]))[
                "Tpk_qed_twist_operator_avg"
            ]
        )
        betas = np.asarray(calibrated[jpm]["betas"], dtype=float)
        target_mixing = transfer_mixing(
            target_evals,
            target_evecs,
            np.arange(cluster.n_ice),
            labels,
            betas,
        )
        center_ratio4 = float(calibrated[jpm]["best"]["kappa4_over_jpm2"])
        ratio4_values = center_ratio4 + ratio4_offsets
        bare_m2 = exact_second_moment(
            full_h0,
            full_transverse,
            full_w4,
            full_w6w,
            jpm,
            0.0,
            0.0,
        )
        temperatures = np.geomspace(1.0e-7, 0.08, 1200)

        print(
            f"Jpm={jpm:+.3f}: target peak={target_peak:.7g}, "
            f"target low-T transfer={np.mean(target_mixing):.3e}",
            flush=True,
        )
        records = []
        v0 = None
        for ratio4 in ratio4_values:
            for ratio6 in ratio6_values:
                kappa4 = float(ratio4 * jpm * jpm)
                kappa6w = float(ratio6 * abs(jpm) ** 3)
                started = time.time()
                evals, evecs, defect_gap = solve_candidate(
                    h0,
                    transverse,
                    w4,
                    w6w,
                    jpm,
                    kappa4,
                    kappa6w,
                    args.n_low,
                    args.tol,
                    cluster.n_ice,
                    v0,
                )
                v0 = evecs[:, 0]
                band = evals[: cluster.n_ice]
                mixing = transfer_mixing(
                    evals, evecs, ice_indices, labels, betas
                )
                m2 = exact_second_moment(
                    full_h0,
                    full_transverse,
                    full_w4,
                    full_w6w,
                    jpm,
                    kappa4,
                    kappa6w,
                )
                record = {
                    "ratio4": float(ratio4),
                    "ratio6": float(ratio6),
                    "kappa4": kappa4,
                    "kappa6w": kappa6w,
                    "spectral_error": centered_spectral_error(band, target_evals),
                    "mixing": [float(value) for value in mixing],
                    "mixing_mean": float(np.mean(mixing)),
                    "low_heat_peak": low_peak(evals, temperatures, cluster.n_ice),
                    "defect_gap": defect_gap,
                    "m2_relative_change": float((m2 - bare_m2) / bare_m2),
                    "elapsed_s": float(time.time() - started),
                }
                records.append(record)
                saved_spectra[(jpm, ratio4, ratio6)] = band
                print(
                    f"  r4={ratio4:+.3f} r6={ratio6:+.1f}: "
                    f"spec={record['spectral_error']:.4f} "
                    f"mix={record['mixing_mean']:.4f} "
                    f"Tpk/T0={record['low_heat_peak'] / target_peak:.3f}",
                    flush=True,
                )

        best = min(records, key=lambda record: float(record["spectral_error"]))
        one_term_candidates = [
            record for record in records if abs(float(record["ratio6"])) < 1.0e-12
        ]
        one_term = min(
            one_term_candidates,
            key=lambda record: abs(
                float(record["ratio4"]) - center_ratio4
            ),
        )
        results.append(
            {
                "jpm": jpm,
                "target_path": str(target_path),
                "target_peak": target_peak,
                "target_flux_fraction": target_flux_fraction(target_h, labels),
                "target_mixing": [float(value) for value in target_mixing],
                "target_mixing_mean": float(np.mean(target_mixing)),
                "one_term": one_term,
                "best_two_term": best,
                "records": records,
            }
        )
        saved_spectra[(jpm, "target")] = target_evals
        print(
            f"  BEST: r4={best['ratio4']:.3f}, r6={best['ratio6']:.1f}, "
            f"spec={best['spectral_error']:.4f}, "
            f"Tpk/T0={best['low_heat_peak'] / target_peak:.3f}",
            flush=True,
        )

    record_maps = []
    for result in results:
        record_maps.append(
            {
                (round(float(record["ratio4"]), 8), round(float(record["ratio6"]), 8)): record
                for record in result["records"]
            }
        )
    shared_grid = set(record_maps[0])
    for record_map in record_maps[1:]:
        shared_grid &= set(record_map)
    shared_fit = None
    if shared_grid:
        shared_key = min(
            shared_grid,
            key=lambda key: np.mean(
                [
                    float(record_map[key]["spectral_error"])
                    for record_map in record_maps
                ]
            ),
        )
        shared_fit = {
            "ratio4": shared_key[0],
            "ratio6": shared_key[1],
            "mean_spectral_error": float(
                np.mean(
                    [
                        record_map[shared_key]["spectral_error"]
                        for record_map in record_maps
                    ]
                )
            ),
            "mean_mixing": float(
                np.mean(
                    [record_map[shared_key]["mixing_mean"] for record_map in record_maps]
                )
            ),
            "by_coupling": [record_map[shared_key] for record_map in record_maps],
        }
    shared_ratio6_grid = set(
        round(float(record["ratio6"]), 8) for record in results[0]["records"]
    )
    for result in results[1:]:
        shared_ratio6_grid &= set(
            round(float(record["ratio6"]), 8) for record in result["records"]
        )

    def record_at_calibrated_ratio4(result, ratio6):
        ratio4 = round(float(result["one_term"]["ratio4"]), 8)
        return next(
            record
            for record in result["records"]
            if round(float(record["ratio4"]), 8) == ratio4
            and round(float(record["ratio6"]), 8) == ratio6
        )

    shared_ratio6 = min(
        shared_ratio6_grid,
        key=lambda ratio6: np.mean(
            [
                record_at_calibrated_ratio4(result, ratio6)["spectral_error"]
                for result in results
            ]
        ),
    )
    shared_ratio6_records = [
        record_at_calibrated_ratio4(result, shared_ratio6) for result in results
    ]
    shared_ratio6_fit = {
        "ratio6": shared_ratio6,
        "ratio4_by_coupling": [
            float(result["one_term"]["ratio4"]) for result in results
        ],
        "mean_spectral_error": float(
            np.mean([record["spectral_error"] for record in shared_ratio6_records])
        ),
        "mean_mixing": float(
            np.mean([record["mixing_mean"] for record in shared_ratio6_records])
        ),
        "by_coupling": shared_ratio6_records,
    }
    summary = {
        "method": "full microscopic ED benchmarked against zero-dipole band",
        "local_ice_projected": bool(args.local_ice_projected),
        "global_ice_projected": bool(args.global_ice_projected),
        "projection_scope": (
            "global" if args.global_ice_projected else "local" if args.local_ice_projected else "none"
        ),
        "projection_definition": (
            "a ring flip acts only when the selected tetrahedron set satisfies "
            "the two-in/two-out ice rule"
            if args.local_ice_projected or args.global_ice_projected
            else None
        ),
        "selection_objective": "centered 90-state spectral RMS / target standard deviation",
        "cluster": {
            "n_sites": cluster.n_sites,
            "ice_states": cluster.n_ice,
            "wrapping_four_loops": len(wrapping_four),
            "wrapping_hexagons": len(wrapping_hexagons),
        },
        "best_shared_two_term": shared_fit,
        "best_shared_ratio6_with_calibrated_ratio4": shared_ratio6_fit,
        "results": results,
    }
    args.out.with_suffix(".json").write_text(json.dumps(summary, indent=2))
    arrays = {"summary": np.asarray(json.dumps(summary))}
    for result in results:
        jpm = float(result["jpm"])
        tag = coupling_tag(jpm)
        arrays[f"{tag}_E_target"] = saved_spectra[(jpm, "target")]
        one = result["one_term"]
        best = result["best_two_term"]
        shared = shared_ratio6_records[results.index(result)]
        arrays[f"{tag}_E_one_term"] = saved_spectra[
            (jpm, float(one["ratio4"]), float(one["ratio6"]))
        ]
        arrays[f"{tag}_E_two_term"] = saved_spectra[
            (jpm, float(best["ratio4"]), float(best["ratio6"]))
        ]
        arrays[f"{tag}_E_two_term_shared"] = saved_spectra[
            (jpm, float(shared["ratio4"]), float(shared["ratio6"]))
        ]
    np.savez_compressed(args.out, **arrays)

    fig, axes = plt.subplots(1, len(results), figsize=(4.2 * len(results), 3.45))
    if len(results) == 1:
        axes = [axes]
    for axis, result in zip(axes, results):
        jpm = float(result["jpm"])
        tag = coupling_tag(jpm)
        target_e = arrays[f"{tag}_E_target"]
        one_e = arrays[f"{tag}_E_one_term"]
        two_e = arrays[f"{tag}_E_two_term_shared"]
        target_peak = float(result["target_peak"])
        temperatures = np.geomspace(target_peak / 20.0, target_peak * 20.0, 700)
        axis.plot(
            temperatures / target_peak,
            R.specific_heat(target_e, temperatures) / cluster.n_sites,
            color="black",
            lw=1.8,
            label="zero dipole",
        )
        axis.plot(
            temperatures / target_peak,
            R.specific_heat(one_e, temperatures) / cluster.n_sites,
            color="#d55e00",
            lw=1.4,
            ls="--",
            label=(
                r"$P_{\rm ice}W_4P_{\rm ice}$"
                if args.global_ice_projected
                else r"$P_{\rm loc}W_4P_{\rm loc}$"
                if args.local_ice_projected
                else r"$W_4$"
            ),
        )
        axis.plot(
            temperatures / target_peak,
            R.specific_heat(two_e, temperatures) / cluster.n_sites,
            color="#007c83",
            lw=1.7,
            label=(
                r"globally projected $W_4+W_{6w}$"
                if args.global_ice_projected
                else r"locally projected $W_4+W_{6w}$"
                if args.local_ice_projected
                else r"$W_4+W_{6w}$ (shared $r_6$)"
            ),
        )
        axis.set_xscale("log")
        axis.set_title(rf"$J_\pm/J_{{zz}}={jpm:+.2f}$")
        axis.set_xlabel(r"$T/T_{\delta=0}$")
        axis.set_ylabel(r"$C/N$")
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    figure_stem = f"fig_{args.out.stem}"
    for extension in ("pdf", "png"):
        fig.savefig(FIGS / f"{figure_stem}.{extension}", dpi=230)
    plt.close(fig)

    print(f"wrote {args.out}")
    print(f"wrote {args.out.with_suffix('.json')}")
    print(f"wrote {FIGS / (figure_stem + '.pdf')}")


if __name__ == "__main__":
    main()
