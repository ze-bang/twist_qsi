#!/usr/bin/env python3
"""All-temperature validation of the microscopic cubic-16 counterterm.

The lowest 90 eigenstates are treated exactly.  The complementary thermal
trace is evaluated by deflated stochastic Lanczos quadrature (SLQ), so the
low-temperature result has no typicality failure while the full 2^16 Hilbert
space is retained at intermediate and high temperature.
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
from scipy.linalg import eigh_tridiagonal
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


def sparse_hamiltonian(
    h0: np.ndarray,
    transverse: sp.csr_matrix,
    w4: sp.csr_matrix,
    w6w: sp.csr_matrix,
    f4: np.ndarray,
    jpm: float,
    kappa4: float,
    kappa6w: float,
    mu4: float,
) -> sp.csr_matrix:
    fcenter = f4 - np.mean(f4)
    return (
        sp.diags(h0 + mu4 * fcenter)
        - jpm * transverse
        + kappa4 * w4
        + kappa6w * w6w
    ).tocsr()


def exact_low_subspace(
    h: sp.csr_matrix,
    n_ice: int,
    n_low: int,
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
    gap = verify_ice_band(evals, n_ice)
    return evals[:n_ice], evecs[:, :n_ice], gap


def lanczos_quadrature(
    h: sp.csr_matrix,
    vector: np.ndarray,
    steps: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    norm2 = float(np.dot(vector, vector))
    q = vector / np.sqrt(norm2)
    q_prev = np.zeros_like(q)
    beta_prev = 0.0
    alphas: list[float] = []
    betas: list[float] = []

    for iteration in range(steps):
        work = h @ q
        if iteration:
            work -= beta_prev * q_prev
        alpha = float(np.dot(q, work))
        work -= alpha * q
        beta = float(np.linalg.norm(work))
        alphas.append(alpha)
        if iteration == steps - 1 or beta < 1.0e-13:
            break
        betas.append(beta)
        q_prev, q = q, work / beta
        beta_prev = beta

    theta, vectors = eigh_tridiagonal(
        np.asarray(alphas), np.asarray(betas[: len(alphas) - 1])
    )
    weights = norm2 * vectors[0, :] ** 2
    return theta, weights, norm2


def deflated_slq(
    h: sp.csr_matrix,
    sz_basis: np.ndarray,
    low_vectors: np.ndarray,
    probes: int,
    steps: int,
    seed: int,
) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray]:
    rng = np.random.default_rng(seed)
    nodes: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    norms = []
    for sample in range(probes):
        vector = rng.integers(0, 2, size=h.shape[0], dtype=np.int8)
        vector = (2 * vector - 1).astype(float)
        sector_part = vector[sz_basis]
        sector_part -= low_vectors @ (low_vectors.T @ sector_part)
        vector[sz_basis] = sector_part
        theta, weight, norm2 = lanczos_quadrature(h, vector, steps)
        nodes.append(theta)
        weights.append(weight)
        norms.append(norm2)
        print(
            f"    SLQ probe {sample + 1:02d}/{probes}: "
            f"m={len(theta)}, norm2={norm2:.3f}",
            flush=True,
        )
    return nodes, weights, np.asarray(norms)


def thermal_curve(
    low_evals: np.ndarray,
    nodes: list[np.ndarray],
    weights: list[np.ndarray],
    temps: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    reference = float(np.min(low_evals))
    beta = 1.0 / temps

    low = low_evals - reference
    low_boltzmann = np.exp(-beta[:, None] * low[None, :])
    exact_z = np.sum(low_boltzmann, axis=1)
    exact_e1 = low_boltzmann @ low
    exact_e2 = low_boltzmann @ (low * low)

    sample_z = []
    sample_e1 = []
    sample_e2 = []
    for theta, weight in zip(nodes, weights):
        energy = theta - reference
        boltzmann = np.exp(-beta[:, None] * energy[None, :])
        sample_z.append(boltzmann @ weight)
        sample_e1.append(boltzmann @ (weight * energy))
        sample_e2.append(boltzmann @ (weight * energy * energy))
    sample_z = np.asarray(sample_z)
    sample_e1 = np.asarray(sample_e1)
    sample_e2 = np.asarray(sample_e2)

    def heat(z_res: np.ndarray, e1_res: np.ndarray, e2_res: np.ndarray) -> np.ndarray:
        z = exact_z + z_res
        e1 = (exact_e1 + e1_res) / z
        e2 = (exact_e2 + e2_res) / z
        return beta * beta * np.maximum(e2 - e1 * e1, 0.0)

    curve = heat(
        np.mean(sample_z, axis=0),
        np.mean(sample_e1, axis=0),
        np.mean(sample_e2, axis=0),
    )
    jackknife = []
    for left_out in range(len(nodes)):
        keep = np.arange(len(nodes)) != left_out
        jackknife.append(
            heat(
                np.mean(sample_z[keep], axis=0),
                np.mean(sample_e1[keep], axis=0),
                np.mean(sample_e2[keep], axis=0),
            )
        )
    jackknife = np.asarray(jackknife)
    mean_jackknife = np.mean(jackknife, axis=0)
    stderr = np.sqrt(
        (len(nodes) - 1)
        * np.mean((jackknife - mean_jackknife[None, :]) ** 2, axis=0)
    )
    trace_estimate = exact_z[-1] + np.mean(sample_z[:, -1])
    return curve, stderr, np.asarray([trace_estimate])


def exact_specific_heat(evals: np.ndarray, temps: np.ndarray) -> np.ndarray:
    energy = np.asarray(evals, dtype=float) - float(np.min(evals))
    out = np.empty_like(temps)
    for index, temp in enumerate(temps):
        weight = np.exp(-energy / temp)
        z = np.sum(weight)
        e1 = np.dot(weight, energy) / z
        e2 = np.dot(weight, energy * energy) / z
        out[index] = (e2 - e1 * e1) / (temp * temp)
    return out


def peak_in_window(
    temps: np.ndarray,
    curve: np.ndarray,
    low: float,
    high: float,
) -> tuple[float, float, int]:
    indices = np.flatnonzero((temps >= low) & (temps <= high))
    index = int(indices[np.argmax(curve[indices])])
    if 0 < index < len(temps) - 1:
        x = np.log(temps[index - 1 : index + 2])
        y = curve[index - 1 : index + 2]
        quadratic = np.polyfit(x, y, 2)
        if quadratic[0] < 0.0:
            vertex = -quadratic[1] / (2.0 * quadratic[0])
            if x[0] <= vertex <= x[-1]:
                return (
                    float(np.exp(vertex)),
                    float(np.polyval(quadratic, vertex)),
                    index,
                )
    return float(temps[index]), float(curve[index]), index


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
    parser.add_argument("--probes", type=int, default=20)
    parser.add_argument("--steps", type=int, default=180)
    parser.add_argument("--n-low", type=int, default=180)
    parser.add_argument("--seed", type=int, default=732451)
    parser.add_argument("--tol", type=float, default=1.0e-9)
    parser.add_argument(
        "--scan-json", type=Path, default=HERE / "winding_counterterm_16site.json"
    )
    parser.add_argument(
        "--out", type=Path, default=HERE / "counterterm_all_temperature.npz"
    )
    args = parser.parse_args()

    if args.kappa4 is None:
        scan = json.loads(args.scan_json.read_text())
        kappa4 = float(scan["best"]["kappa4"])
    else:
        kappa4 = float(args.kappa4)

    cluster = R.build_cluster("cubic", (1, 1, 1))
    loops4 = [loop for loop in cluster.loops4 if tuple(loop[1]) != (0, 0, 0)]
    loops6w = [loop for loop in cluster.hexes if tuple(loop[1]) != (0, 0, 0)]
    full_basis = np.arange(1 << cluster.n_sites, dtype=np.uint64)
    sz_basis = sz0_basis(cluster.n_sites)

    print("building full-space microscopic operators", flush=True)
    h0_full = R.ising_energy(cluster, full_basis)
    transverse_full = transverse_operator(cluster, full_basis)
    ring_builder = (
        lambda states, loops: local_ice_projected_ring_operator(
            cluster, states, loops
        )
        if args.local_ice_projected
        else ring_operator(states, loops)
    )
    w4_full, f4_full = ring_builder(full_basis, loops4)
    w6w_full, _ = ring_builder(full_basis, loops6w)

    h0_sz = R.ising_energy(cluster, sz_basis)
    transverse_sz = transverse_operator(cluster, sz_basis)
    w4_sz, f4_sz = ring_builder(sz_basis, loops4)
    w6w_sz, _ = ring_builder(sz_basis, loops6w)

    temps = np.geomspace(5.0e-4, 5.0, 700)
    results = {}
    for name, kappa, kappa6, mu in (
        ("bare", 0.0, 0.0, 0.0),
        ("improved", kappa4, args.kappa6w, args.mu4),
    ):
        print(
            f"[{name}] kappa4={kappa:+.6f}, kappa6w={kappa6:+.6f}, "
            f"mu4={mu:+.6f}",
            flush=True,
        )
        h_full = sparse_hamiltonian(
            h0_full,
            transverse_full,
            w4_full,
            w6w_full,
            f4_full,
            args.jpm,
            kappa,
            kappa6,
            mu,
        )
        h_sz = sparse_hamiltonian(
            h0_sz,
            transverse_sz,
            w4_sz,
            w6w_sz,
            f4_sz,
            args.jpm,
            kappa,
            kappa6,
            mu,
        )
        started = time.time()
        low_evals, low_vectors, defect_gap = exact_low_subspace(
            h_sz, cluster.n_ice, args.n_low, args.tol
        )
        nodes, weights, norms = deflated_slq(
            h_full,
            sz_basis,
            low_vectors,
            args.probes,
            args.steps,
            args.seed,
        )
        heat, heat_error, trace_estimate = thermal_curve(
            low_evals, nodes, weights, temps
        )
        results[name] = {
            "low_evals": low_evals,
            "heat": heat,
            "heat_error": heat_error,
            "defect_gap": defect_gap,
            "trace_estimate": float(trace_estimate[0]),
            "mean_deflated_probe_norm2": float(np.mean(norms)),
            "elapsed_s": float(time.time() - started),
        }

    exact_path = HERE / "full_ed_16site_spectrum_jm0p05.npz"
    exact_bare = np.asarray(np.load(exact_path)["E_full"], dtype=float)
    exact_bare_heat = exact_specific_heat(exact_bare, temps)

    bare_error = np.abs(results["bare"]["heat"] - exact_bare_heat) / cluster.n_sites
    low_bare = peak_in_window(temps, exact_bare_heat, 5.0e-4, 0.08)
    low_improved = peak_in_window(temps, results["improved"]["heat"], 5.0e-4, 0.08)
    high_bare = peak_in_window(temps, exact_bare_heat, 0.08, 3.0)
    high_improved = peak_in_window(temps, results["improved"]["heat"], 0.08, 3.0)
    high_index = high_improved[2]

    summary = {
        "method": "exact 90-state low band plus deflated full-Hilbert SLQ",
        "n_sites": cluster.n_sites,
        "hilbert_dimension": 1 << cluster.n_sites,
        "jpm": args.jpm,
        "kappa4": kappa4,
        "kappa6w": args.kappa6w,
        "mu4": args.mu4,
        "local_ice_projected": bool(args.local_ice_projected),
        "probes": args.probes,
        "lanczos_steps": args.steps,
        "bare_slq_max_abs_error_per_site": float(np.max(bare_error)),
        "bare_slq_rms_error_per_site": float(np.sqrt(np.mean(bare_error**2))),
        "bare_low_peak": low_bare[:2],
        "improved_low_peak": low_improved[:2],
        "bare_high_peak": high_bare[:2],
        "improved_high_peak": high_improved[:2],
        "high_peak_relative_temperature_shift": float(
            (high_improved[0] - high_bare[0]) / high_bare[0]
        ),
        "improved_high_peak_stderr_per_site": float(
            results["improved"]["heat_error"][high_index] / cluster.n_sites
        ),
        "bare_defect_gap": results["bare"]["defect_gap"],
        "improved_defect_gap": results["improved"]["defect_gap"],
        "bare_trace_estimate_at_Tmax": results["bare"]["trace_estimate"],
        "improved_trace_estimate_at_Tmax": results["improved"]["trace_estimate"],
        "elapsed_s": {
            key: value["elapsed_s"] for key, value in results.items()
        },
    }

    np.savez_compressed(
        args.out,
        T=temps,
        C_bare_slq=results["bare"]["heat"],
        C_bare_slq_error=results["bare"]["heat_error"],
        C_bare_exact=exact_bare_heat,
        C_improved=results["improved"]["heat"],
        C_improved_error=results["improved"]["heat_error"],
        E_low_bare=results["bare"]["low_evals"],
        E_low_improved=results["improved"]["low_evals"],
        summary=json.dumps(summary),
    )
    args.out.with_suffix(".json").write_text(json.dumps(summary, indent=2))

    fig, (ax_low, ax_all) = plt.subplots(1, 2, figsize=(9.0, 3.55))
    for ax in (ax_low, ax_all):
        ax.plot(temps, exact_bare_heat / cluster.n_sites, color="0.25", lw=1.4,
                label="bare, exact full ED")
        ax.plot(temps, results["improved"]["heat"] / cluster.n_sites,
                color="#007c83", lw=1.8, label="improved, deflated SLQ")
        error = results["improved"]["heat_error"] / cluster.n_sites
        ax.fill_between(
            temps,
            results["improved"]["heat"] / cluster.n_sites - error,
            results["improved"]["heat"] / cluster.n_sites + error,
            color="#007c83",
            alpha=0.18,
            linewidth=0,
        )
        ax.set_xscale("log")
        ax.set_xlabel(r"$T/J_{zz}$")
        ax.set_ylabel(r"$C/N$")
        ax.spines[["top", "right"]].set_visible(False)
    ax_low.set_xlim(5.0e-4, 0.08)
    ax_all.set_xlim(5.0e-4, 5.0)
    ax_low.legend(frameon=False, fontsize=8)
    ax_all.text(
        0.04,
        0.94,
        rf"$\kappa_4={kappa4:.4f},\ \kappa_{{6w}}={args.kappa6w:.5f}$",
        transform=ax_all.transAxes,
        va="top",
        fontsize=9,
    )
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
