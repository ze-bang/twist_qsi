#!/usr/bin/env python3
"""Coupling sweep for the nonperturbative wrapping-loop counterterm.

The calibration temperature is set by an independently diagonalized embedded
hexagon: six active spins surrounded by six frozen opposite-spin boundary
pairs.  This 64-state microscopic problem contains the physical local hexagon
tunneling but no periodic wrapping process.
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
    centered_second_moment,
    flux_labels,
    low_peak,
    ring_operator,
    transfer_mixing,
    transverse_operator,
    verify_ice_band,
)

HERE = Path(__file__).resolve().parent
FIGS = HERE / "figs"
FIGS.mkdir(exist_ok=True)


def embedded_hexagon_spectrum(jpm: float) -> np.ndarray:
    """Exact microscopic spectrum with a fully constrained central hexagon."""
    hamiltonian = np.zeros((64, 64), dtype=float)
    for state in range(64):
        for site in range(6):
            neighbor = (site + 1) % 6
            bit_i = (state >> site) & 1
            bit_j = (state >> neighbor) & 1
            # Opposite frozen boundary spins cancel their longitudinal field.
            # A parallel active pair is a 3-in/1-out tetrahedron of cost Jzz/2.
            if bit_i == bit_j:
                hamiltonian[state, state] += 0.5
            else:
                flipped = state ^ (1 << site) ^ (1 << neighbor)
                hamiltonian[flipped, state] += -jpm
    return np.linalg.eigvalsh(hamiltonian)


def embedded_hexagon_diagnostics(jpm: float) -> dict[str, float]:
    evals = embedded_hexagon_spectrum(jpm)
    splitting = float(evals[1] - evals[0])
    temperatures = np.geomspace(max(1.0e-7, splitting / 40.0), splitting * 4.0, 700)
    heat = R.specific_heat(evals, temperatures)
    return {
        "ground_energy": float(evals[0]),
        "doublet_splitting": splitting,
        "low_heat_peak": float(R.refined_peak(temperatures, heat)),
        "defect_gap": float(evals[2] - evals[1]),
    }


def evaluate(
    cluster: R.Cluster,
    h0: np.ndarray,
    transverse: sp.csr_matrix,
    w4: sp.csr_matrix,
    ice_indices: np.ndarray,
    labels: np.ndarray,
    full_components: tuple[np.ndarray, sp.csr_matrix, sp.csr_matrix, np.ndarray],
    jpm: float,
    kappa4: float,
    betas: np.ndarray,
    n_low: int,
    tolerance: float,
    v0: np.ndarray | None,
) -> tuple[dict[str, object], np.ndarray]:
    hamiltonian = (sp.diags(h0) - jpm * transverse + kappa4 * w4).tocsr()
    started = time.time()
    evals, evecs = eigsh(
        hamiltonian,
        k=n_low,
        which="SA",
        tol=tolerance,
        ncv=max(2 * n_low + 20, 380),
        v0=v0,
    )
    order = np.argsort(evals)
    evals = np.asarray(evals[order], dtype=float)
    evecs = np.asarray(evecs[:, order], dtype=float)
    defect_gap = verify_ice_band(evals, cluster.n_ice)
    mixing = transfer_mixing(evals, evecs, ice_indices, labels, betas)
    temperatures = np.geomspace(1.0e-7, 0.15, 1200)
    m2_bare = centered_second_moment(cluster, jpm, 0.0, 0.0, full_components)
    m2 = centered_second_moment(cluster, jpm, kappa4, 0.0, full_components)
    record: dict[str, object] = {
        "kappa4": float(kappa4),
        "kappa4_over_jpm2": float(kappa4 / (jpm * jpm)),
        "mixing": [float(value) for value in mixing],
        "mixing_mean": float(np.mean(mixing)),
        "low_heat_peak": low_peak(evals, temperatures, cluster.n_ice),
        "defect_gap": defect_gap,
        "ice_overlap_ground": float(np.sum(evecs[ice_indices, 0] ** 2)),
        "m2_relative_change": float((m2 - m2_bare) / m2_bare),
        "elapsed_s": float(time.time() - started),
    }
    return record, evecs[:, 0]


def unique_sorted(values: list[float], tolerance: float = 1.0e-12) -> list[float]:
    output: list[float] = []
    for value in sorted(values):
        if not output or abs(value - output[-1]) > tolerance:
            output.append(value)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jpm", default="-0.03,-0.04,-0.05,-0.06")
    parser.add_argument("--ratio-min", type=float, default=1.5)
    parser.add_argument("--ratio-max", type=float, default=4.5)
    parser.add_argument("--coarse-count", type=int, default=5)
    parser.add_argument("--refine-count", type=int, default=5)
    parser.add_argument("--n-low", type=int, default=180)
    parser.add_argument("--tol", type=float, default=1.0e-9)
    parser.add_argument(
        "--out", type=Path, default=HERE / "counterterm_coupling_sweep.npz"
    )
    args = parser.parse_args()

    couplings = [float(value) for value in args.jpm.split(",")]
    cluster = R.build_cluster("cubic", (1, 1, 1))
    wrapping_four = [
        loop for loop in cluster.loops4 if tuple(loop[1]) != (0, 0, 0)
    ]
    basis = sz0_basis(cluster.n_sites)
    basis_index = {int(state): index for index, state in enumerate(basis)}
    ice_indices = np.asarray(
        [basis_index[int(state)] for state in cluster.ice_states], dtype=int
    )
    labels = flux_labels(cluster)
    h0 = R.ising_energy(cluster, basis)
    transverse = transverse_operator(cluster, basis)
    w4, _ = ring_operator(basis, wrapping_four)

    full_basis = np.arange(1 << cluster.n_sites, dtype=np.uint64)
    full_w4, full_f4 = ring_operator(full_basis, wrapping_four)
    full_components = (
        R.ising_energy(cluster, full_basis),
        transverse_operator(cluster, full_basis),
        full_w4,
        full_f4,
    )

    all_results = []
    for coupling in couplings:
        local = embedded_hexagon_diagnostics(coupling)
        splitting = local["doublet_splitting"]
        betas = np.asarray([0.06, 0.15, 0.30]) / splitting
        print(
            f"Jpm={coupling:+.4f}: embedded-hex split={splitting:.7g}, "
            f"betas={np.round(betas, 3).tolist()}",
            flush=True,
        )

        bare, v0 = evaluate(
            cluster,
            h0,
            transverse,
            w4,
            ice_indices,
            labels,
            full_components,
            coupling,
            0.0,
            betas,
            args.n_low,
            args.tol,
            None,
        )
        print(
            f"  bare: mix={bare['mixing_mean']:.5f}, "
            f"Tpk={bare['low_heat_peak']:.7g}",
            flush=True,
        )

        coarse_ratios = np.linspace(
            args.ratio_min, args.ratio_max, args.coarse_count
        )
        records = []
        for ratio in coarse_ratios:
            record, v0 = evaluate(
                cluster,
                h0,
                transverse,
                w4,
                ice_indices,
                labels,
                full_components,
                coupling,
                float(ratio * coupling * coupling),
                betas,
                args.n_low,
                args.tol,
                v0,
            )
            records.append(record)
            print(
                f"  ratio={ratio:.4f}: mix={record['mixing_mean']:.5f}, "
                f"Tpk={record['low_heat_peak']:.7g}",
                flush=True,
            )

        coarse_best = min(
            range(len(records)), key=lambda index: records[index]["mixing_mean"]
        )
        lower_index = max(0, coarse_best - 1)
        upper_index = min(len(coarse_ratios) - 1, coarse_best + 1)
        refine_ratios = np.linspace(
            coarse_ratios[lower_index],
            coarse_ratios[upper_index],
            args.refine_count,
        )
        existing = [float(record["kappa4_over_jpm2"]) for record in records]
        new_ratios = [
            ratio
            for ratio in unique_sorted([float(value) for value in refine_ratios])
            if all(abs(ratio - old) > 1.0e-12 for old in existing)
        ]
        for ratio in new_ratios:
            record, v0 = evaluate(
                cluster,
                h0,
                transverse,
                w4,
                ice_indices,
                labels,
                full_components,
                coupling,
                float(ratio * coupling * coupling),
                betas,
                args.n_low,
                args.tol,
                v0,
            )
            records.append(record)
            print(
                f"  refine ratio={ratio:.4f}: "
                f"mix={record['mixing_mean']:.5f}, "
                f"Tpk={record['low_heat_peak']:.7g}",
                flush=True,
            )

        records.sort(key=lambda record: float(record["kappa4"]))
        best = min(records, key=lambda record: float(record["mixing_mean"]))
        print(
            f"  BEST kappa4={best['kappa4']:.7g} "
            f"(ratio={best['kappa4_over_jpm2']:.4f}), "
            f"mix={best['mixing_mean']:.5f}",
            flush=True,
        )
        all_results.append(
            {
                "jpm": coupling,
                "betas": [float(value) for value in betas],
                "embedded_hexagon": local,
                "bare": bare,
                "best": best,
                "records": records,
            }
        )

    summary = {
        "method": "microscopic cubic-16 flux-transfer minimization",
        "temperature_reference": "exact 64-state embedded hexagon",
        "cluster": {
            "n_sites": cluster.n_sites,
            "ice_states": cluster.n_ice,
            "flux_sectors": int(len(np.unique(labels))),
            "wrapping_four_loops": len(wrapping_four),
        },
        "results": all_results,
    }
    args.out.with_suffix(".json").write_text(json.dumps(summary, indent=2))
    np.savez_compressed(args.out, summary=json.dumps(summary))

    absolute_jpm = np.abs([result["jpm"] for result in all_results])
    best_kappa = np.asarray([result["best"]["kappa4"] for result in all_results])
    ratios = np.asarray(
        [result["best"]["kappa4_over_jpm2"] for result in all_results]
    )
    bare_mixing = np.asarray(
        [result["bare"]["mixing_mean"] for result in all_results]
    )
    best_mixing = np.asarray(
        [result["best"]["mixing_mean"] for result in all_results]
    )
    bare_peak = np.asarray(
        [result["bare"]["low_heat_peak"] for result in all_results]
    )
    best_peak = np.asarray(
        [result["best"]["low_heat_peak"] for result in all_results]
    )
    local_peak = np.asarray(
        [result["embedded_hexagon"]["low_heat_peak"] for result in all_results]
    )

    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.35))
    axes[0].plot(absolute_jpm, ratios, "o-", color="#007c83")
    axes[0].set_xlabel(r"$|J_\pm|/J_{zz}$")
    axes[0].set_ylabel(r"$\kappa_4/J_\pm^2$")

    axes[1].semilogy(absolute_jpm, bare_mixing, "o-", color="0.3", label="periodic")
    axes[1].semilogy(
        absolute_jpm, best_mixing, "o-", color="#007c83", label="improved"
    )
    axes[1].set_xlabel(r"$|J_\pm|/J_{zz}$")
    axes[1].set_ylabel("mean flux transfer")
    axes[1].legend(frameon=False, fontsize=8)

    axes[2].plot(absolute_jpm, bare_peak / local_peak, "o-", color="0.3",
                 label="periodic")
    axes[2].plot(absolute_jpm, best_peak / local_peak, "o-", color="#007c83",
                 label="improved")
    axes[2].axhline(1.0, color="0.6", lw=0.8, ls=":")
    axes[2].set_xlabel(r"$|J_\pm|/J_{zz}$")
    axes[2].set_ylabel(r"$T_{\rm peak}/T_{\rm hex}^{\rm exact}$")
    axes[2].legend(frameon=False, fontsize=8)
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
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
