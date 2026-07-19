#!/usr/bin/env python3
"""Dynamical validation of the cubic-16 wrapping-loop counterterm.

The response is evaluated at the cubic star of the extended-zone pinch point
(002), which is compatible with the four primitive FCC translations and has
nonzero longitudinal ice-manifold weight.  Both Hamiltonians are microscopic
and are diagonalized in the full Sz=0 block.
"""
from __future__ import annotations

import argparse
import json
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
    local_ice_projected_ring_operator,
    ring_operator,
    transverse_operator,
    verify_ice_band,
)

HERE = Path(__file__).resolve().parent
FIGS = HERE / "figs"
FIGS.mkdir(exist_ok=True)


def solve(
    h: sp.csr_matrix,
    n_low: int,
    n_ice: int,
    tol: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    evals, evecs = eigsh(
        h,
        k=n_low,
        which="SA",
        tol=tol,
        ncv=max(2 * n_low + 20, 380),
    )
    order = np.argsort(evals)
    evals = np.asarray(evals[order], dtype=float)
    evecs = np.asarray(evecs[:, order], dtype=float)
    return evals, evecs, verify_ice_band(evals, n_ice)


def spectral_lines_diagonal(
    evals: np.ndarray,
    evecs: np.ndarray,
    diagonal_operators: list[np.ndarray],
    ground_tolerance: float,
) -> tuple[np.ndarray, np.ndarray, int]:
    ground_count = int(np.count_nonzero(evals - evals[0] <= ground_tolerance))
    weights = np.zeros(len(evals), dtype=float)
    for diagonal in diagonal_operators:
        for ground in range(ground_count):
            amplitudes = evecs.conj().T @ (diagonal * evecs[:, ground])
            weights += np.abs(amplitudes) ** 2
    weights /= len(diagonal_operators) * ground_count
    omega = evals - evals[0]
    weights[omega <= ground_tolerance] = 0.0
    return omega, weights, ground_count


def spectral_lines_sparse(
    evals: np.ndarray,
    evecs: np.ndarray,
    operator: sp.csr_matrix,
    ground_tolerance: float,
) -> tuple[np.ndarray, np.ndarray, int]:
    ground_count = int(np.count_nonzero(evals - evals[0] <= ground_tolerance))
    weights = np.zeros(len(evals), dtype=float)
    for ground in range(ground_count):
        amplitudes = evecs.conj().T @ (operator @ evecs[:, ground])
        weights += np.abs(amplitudes) ** 2 / ground_count
    omega = evals - evals[0]
    weights[omega <= ground_tolerance] = 0.0
    return omega, weights, ground_count


def broaden(
    line_omega: np.ndarray,
    line_weight: np.ndarray,
    omega: np.ndarray,
    eta: float,
) -> np.ndarray:
    distance = (omega[:, None] - line_omega[None, :]) / eta
    kernel = np.exp(-0.5 * distance * distance) / (np.sqrt(2.0 * np.pi) * eta)
    return kernel @ line_weight


def line_summary(
    line_omega: np.ndarray,
    line_weight: np.ndarray,
    omega: np.ndarray,
    spectrum: np.ndarray,
    total_inelastic_weight: float,
) -> dict[str, float]:
    total = float(np.sum(line_weight))
    nonzero = np.flatnonzero(line_weight > max(1.0e-12, total * 1.0e-7))
    return {
        "integrated_low_spectral_weight": total,
        "total_inelastic_static_weight": total_inelastic_weight,
        "low_band_captured_fraction": total / total_inelastic_weight,
        "lowest_weighted_excitation": (
            float(np.min(line_omega[nonzero])) if len(nonzero) else float("nan")
        ),
        "spectral_centroid": (
            float(np.dot(line_weight, line_omega) / total) if total else float("nan")
        ),
        "broadened_peak": float(omega[np.argmax(spectrum)]),
    }


def static_inelastic_diagonal(
    evecs: np.ndarray,
    diagonal_operators: list[np.ndarray],
    ground_count: int,
) -> float:
    total = 0.0
    for diagonal in diagonal_operators:
        for ground in range(ground_count):
            excited = diagonal * evecs[:, ground]
            elastic = evecs[:, :ground_count].conj().T @ excited
            total += float(np.vdot(excited, excited).real - np.vdot(elastic, elastic).real)
    return total / (len(diagonal_operators) * ground_count)


def static_inelastic_sparse(
    evecs: np.ndarray,
    operator: sp.csr_matrix,
    ground_count: int,
) -> float:
    total = 0.0
    for ground in range(ground_count):
        excited = operator @ evecs[:, ground]
        elastic = evecs[:, :ground_count].conj().T @ excited
        total += float(np.vdot(excited, excited).real - np.vdot(elastic, elastic).real)
    return total / ground_count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jpm", type=float, default=-0.05)
    parser.add_argument("--kappa4", type=float, default=None)
    parser.add_argument("--kappa6w", type=float, default=0.0)
    parser.add_argument("--mu4", type=float, default=0.0)
    parser.add_argument(
        "--local-ice-projected",
        action="store_true",
        help="condition winding counterterms on ice-rule tetrahedra touching each loop",
    )
    parser.add_argument("--n-low", type=int, default=180)
    parser.add_argument("--eta", type=float, default=5.0e-4)
    parser.add_argument("--tol", type=float, default=1.0e-9)
    parser.add_argument(
        "--scan-json",
        type=Path,
        default=HERE / "winding_counterterm_16site_fine.json",
    )
    parser.add_argument(
        "--out", type=Path, default=HERE / "counterterm_szz_validation.npz"
    )
    args = parser.parse_args()

    if args.kappa4 is None:
        kappa4 = float(json.loads(args.scan_json.read_text())["best"]["kappa4"])
    else:
        kappa4 = float(args.kappa4)

    cluster = R.build_cluster("cubic", (1, 1, 1))
    loops4 = [loop for loop in cluster.loops4 if tuple(loop[1]) != (0, 0, 0)]
    loops6w = [loop for loop in cluster.hexes if tuple(loop[1]) != (0, 0, 0)]
    hexagons = [loop for loop in cluster.hexes if tuple(loop[1]) == (0, 0, 0)]
    basis = sz0_basis(cluster.n_sites)
    h0 = R.ising_energy(cluster, basis)
    transverse = transverse_operator(cluster, basis)
    ring_builder = (
        lambda states, loops: local_ice_projected_ring_operator(
            cluster, states, loops
        )
        if args.local_ice_projected
        else ring_operator(states, loops)
    )
    w4, f4 = ring_builder(basis, loops4)
    w6w, _ = ring_builder(basis, loops6w)
    w6_contractible, _ = ring_operator(basis, hexagons)
    f4_center = f4 - np.mean(f4)

    bit_values = (
        (basis[:, None] >> np.arange(cluster.n_sites, dtype=np.uint64)) & 1
    ).astype(float) - 0.5
    pinch_point_momenta = 4.0 * np.pi * np.eye(3)
    szz_diagonals = []
    for momentum in pinch_point_momenta:
        phase = np.exp(1j * (cluster.positions @ momentum)) / np.sqrt(cluster.n_sites)
        szz_diagonals.append(bit_values @ phase)

    frequency = np.linspace(0.0, 0.15, 1500)
    datasets = {}
    for name, kappa, kappa6, mu in (
        ("bare", 0.0, 0.0, 0.0),
        ("improved", kappa4, args.kappa6w, args.mu4),
    ):
        h = (
            sp.diags(h0 + mu * f4_center)
            - args.jpm * transverse
            + kappa * w4
            + kappa6 * w6w
        ).tocsr()
        evals, evecs, defect_gap = solve(h, args.n_low, cluster.n_ice, args.tol)

        line_energy, line_weight, ground_count = spectral_lines_diagonal(
            evals, evecs, szz_diagonals, 1.0e-8
        )
        szz = broaden(line_energy, line_weight, frequency, args.eta)
        szz_static = static_inelastic_diagonal(
            evecs, szz_diagonals, ground_count
        )

        hex_energy, hex_weight, _ = spectral_lines_sparse(
            evals,
            evecs,
            w6_contractible / np.sqrt(len(hexagons)),
            1.0e-8,
        )
        hex_response = broaden(hex_energy, hex_weight, frequency, args.eta)
        hex_static = static_inelastic_sparse(
            evecs, w6_contractible / np.sqrt(len(hexagons)), ground_count
        )
        datasets[name] = {
            "evals": evals,
            "szz": szz,
            "hex_response": hex_response,
            "line_energy": line_energy,
            "line_weight": line_weight,
            "hex_line_energy": hex_energy,
            "hex_line_weight": hex_weight,
            "ground_multiplicity": ground_count,
            "defect_gap": defect_gap,
            "szz_summary": line_summary(
                line_energy, line_weight, frequency, szz, szz_static
            ),
            "hex_summary": line_summary(
                hex_energy, hex_weight, frequency, hex_response, hex_static
            ),
        }
        print(
            f"{name}: ground multiplicity={ground_count}, "
            f"Szz peak={datasets[name]['szz_summary']['broadened_peak']:.6g}, "
            f"hex peak={datasets[name]['hex_summary']['broadened_peak']:.6g}",
            flush=True,
        )

    summary = {
        "method": "full microscopic Sz=0 ED",
        "cluster": "cubic-16",
        "momenta": [list(momentum) for momentum in pinch_point_momenta],
        "momentum_units": "inverse conventional cubic lattice constant",
        "jpm": args.jpm,
        "kappa4": kappa4,
        "kappa6w": args.kappa6w,
        "mu4": args.mu4,
        "local_ice_projected": bool(args.local_ice_projected),
        "eta": args.eta,
        "bare": {
            "ground_multiplicity": datasets["bare"]["ground_multiplicity"],
            "defect_gap": datasets["bare"]["defect_gap"],
            "szz": datasets["bare"]["szz_summary"],
            "contractible_hexagon": datasets["bare"]["hex_summary"],
        },
        "improved": {
            "ground_multiplicity": datasets["improved"]["ground_multiplicity"],
            "defect_gap": datasets["improved"]["defect_gap"],
            "szz": datasets["improved"]["szz_summary"],
            "contractible_hexagon": datasets["improved"]["hex_summary"],
        },
    }
    np.savez_compressed(
        args.out,
        omega=frequency,
        Szz_bare=datasets["bare"]["szz"],
        Szz_improved=datasets["improved"]["szz"],
        hex_bare=datasets["bare"]["hex_response"],
        hex_improved=datasets["improved"]["hex_response"],
        E_bare=datasets["bare"]["evals"],
        E_improved=datasets["improved"]["evals"],
        summary=json.dumps(summary),
    )
    args.out.with_suffix(".json").write_text(json.dumps(summary, indent=2))

    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.35), sharex=True)
    colors = {"bare": "0.25", "improved": "#007c83"}
    labels = {"bare": "periodic", "improved": "winding improved"}
    for name in ("bare", "improved"):
        axes[0].plot(
            frequency,
            datasets[name]["szz"],
            color=colors[name],
            lw=1.7,
            label=labels[name],
        )
        axes[1].plot(
            frequency,
            datasets[name]["hex_response"],
            color=colors[name],
            lw=1.7,
            label=labels[name],
        )
    axes[0].set_title(r"$S^{zz}((002),\omega)$, cubic average")
    axes[1].set_title("contractible-hexagon response")
    for axis in axes:
        axis.set_xlabel(r"$\omega/J_{zz}$")
        axis.set_ylabel("spectral weight")
        axis.set_xlim(0.0, 0.15)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    figure_stem = f"fig_{args.out.stem}"
    for extension in ("pdf", "png"):
        fig.savefig(FIGS / f"{figure_stem}.{extension}", dpi=230)
    plt.close(fig)

    print(json.dumps(summary, indent=2))
    print(f"wrote {args.out}")
    print(f"wrote {FIGS / (figure_stem + '.pdf')}")


if __name__ == "__main__":
    main()
