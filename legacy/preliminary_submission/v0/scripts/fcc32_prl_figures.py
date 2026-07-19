#!/usr/bin/env python3
"""Generate the four FCC-32 figures for the PRL submission."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy.optimize import brentq

HERE = Path(__file__).resolve().parent
SUBMISSION = HERE.parent
PROJECT = SUBMISSION.parent
NOTES = PROJECT / "notes"
sys.path.insert(0, str(NOTES if NOTES.is_dir() else HERE))

import recompute_finite_size_artifact as R  # noqa: E402

OUT_FIG = SUBMISSION / "figures"
OUT_DATA = SUBMISSION / "data"
OUT_FIG.mkdir(exist_ok=True)
OUT_DATA.mkdir(exist_ok=True)

ORANGE = "#d97706"
GREEN = "#087f5b"
BLUE = "#176b87"
RED = "#b83a32"
PURPLE = "#7651a8"
GREY = "#7b8790"
KB_MEV_PER_K = 0.08617333262
R_GAS = 8.314462618
JPM_REFERENCE = -0.05
SMITH_A = np.array([0.050, 0.021, 0.004])
T2_K = 0.025

mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 8.2,
        "axes.labelsize": 8.2,
        "axes.titlesize": 8.5,
        "legend.fontsize": 6.8,
        "xtick.labelsize": 7.2,
        "ytick.labelsize": 7.2,
        "axes.linewidth": 0.7,
        "mathtext.fontset": "cm",
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }
)


def save_figure(fig: plt.Figure, name: str) -> None:
    for ext in ("pdf", "png"):
        fig.savefig(OUT_FIG / f"{name}.{ext}")
    plt.close(fig)


def ice_sz(cl: R.Cluster) -> np.ndarray:
    states = np.asarray(cl.ice_states, dtype=np.uint64)
    bits = (states[:, None] >> np.arange(cl.n_sites, dtype=np.uint64)) & 1
    return bits.astype(float) - 0.5


def parity_labels(cl: R.Cluster) -> np.ndarray:
    x2 = np.rint(2.0 * (ice_sz(cl) @ cl.positions)).astype(int)
    return np.mod(x2, 2) @ np.array([1, 2, 4])


def allowed_momenta(cl: R.Cluster) -> tuple[np.ndarray, list[str]]:
    q = []
    for m0 in (0, 1):
        for m1 in (0, 1):
            for m2 in (0, 1):
                q.append(np.linalg.solve(cl.Lvecs, np.array([m0, m1, m2], float)))
    labels = [
        r"$\Gamma$", r"$L_1$", r"$L_2$", r"$X_z$",
        r"$L_3$", r"$X_y$", r"$X_x$", r"$L_4$",
    ]
    return np.asarray(q), labels


def dssf(cl: R.Cluster, evals: np.ndarray, evecs: np.ndarray,
         qs: np.ndarray, omega_over_g: np.ndarray, ghex: float,
         eta_over_g: float = 0.07) -> np.ndarray:
    """Sublattice-traced longitudinal pseudospin DSSF."""
    energies = (evals - evals[0]) / ghex
    ground = np.flatnonzero(np.abs(energies) < 1e-10)
    sz = ice_sz(cl)
    sublattice = np.arange(cl.n_sites) % 4
    result = np.zeros((len(omega_over_g), len(qs)))
    for g in ground:
        transitions = evecs.conj().T @ (sz * evecs[:, g, None])
        for iq, q in enumerate(qs):
            phase = np.exp(-2j * np.pi * (cl.positions @ q))
            for a in range(4):
                vector = phase * (sublattice == a)
                amp = transitions @ vector / np.sqrt(cl.n_sites)
                weight = np.abs(amp) ** 2
                weight[ground] = 0.0
                delta = (
                    omega_over_g[:, None] - energies[None, :]
                ) / eta_over_g
                result[:, iq] += (
                    np.exp(-0.5 * delta**2) @ weight
                    / (np.sqrt(2.0 * np.pi) * eta_over_g)
                )
    return result / len(ground)


def refined_peak(evals: np.ndarray, scale: float) -> float:
    temperatures = np.geomspace(0.01 * scale, 24.0 * scale, 1600)
    curve = R.specific_heat(evals, temperatures) / 32.0
    return R.refined_peak(temperatures, curve)


def unwrap(cl: R.Cluster, path: tuple[int, ...]) -> np.ndarray:
    points = [cl.positions[path[0]].copy()]
    current = points[0].copy()
    closed = list(path) + [path[0]]
    for a, b in zip(closed, closed[1:]):
        image = next(n for v, n in cl.adj[a] if v == b)
        current += cl.positions[b] - cl.positions[a] - np.asarray(image) @ cl.Lvecs
        points.append(current.copy())
    return np.asarray(points)


def draw_loop(ax, cl: R.Cluster, path: tuple[int, ...], color: str,
              offset: np.ndarray, dashed_boundary: bool) -> None:
    points = unwrap(cl, path) + offset
    for (a, b), (p0, p1) in zip(
        zip(list(path), list(path[1:]) + [path[0]]),
        zip(points[:-1], points[1:]),
    ):
        image = next(n for v, n in cl.adj[a] if v == b)
        crossing = dashed_boundary and np.any(np.asarray(image) != 0)
        ax.plot(*zip(p0, p1), color=color, lw=2.1,
                ls="--" if crossing else "-", zorder=3)
    for k, point in enumerate(points[:-1]):
        ax.scatter(*point, color=RED if k % 2 == 0 else BLUE, s=30,
                   edgecolor="white", linewidth=0.35, depthshade=False, zorder=5)


def draw_cluster(ax, cl: R.Cluster) -> None:
    """Draw the home-cell pyrochlore network behind highlighted loops."""
    for (u, v), image in zip(cl.bonds, cl.bond_wrap):
        if np.any(image):
            continue
        p0, p1 = cl.positions[u], cl.positions[v]
        ax.plot(*zip(p0, p1), color="#aeb6bc", lw=0.55, alpha=0.62, zorder=0)
    ax.scatter(*cl.positions.T, color="#808b93", s=5.5, alpha=0.72,
               depthshade=False, zorder=1)


def draw_unit_cube(ax) -> None:
    corners = np.array(
        [[x, y, z] for x in (0.0, 1.0) for y in (0.0, 1.0)
         for z in (0.0, 1.0)]
    )
    for i, p0 in enumerate(corners):
        for p1 in corners[i + 1:]:
            if np.count_nonzero(np.abs(p1 - p0) > 1.0e-12) == 1:
                ax.plot(*zip(p0, p1), color=GREY, lw=0.55, ls=":",
                        alpha=0.72, zorder=0)


def figure1(cl: R.Cluster, e_bare: np.ndarray, e_clean: np.ndarray) -> None:
    fig = plt.figure(figsize=(7.1, 2.35))
    loop_cl = R.build_cluster("cubic", (1, 1, 1))
    center = np.mean(loop_cl.positions, axis=0)
    four, _ = min(
        ((p, w) for p, w in loop_cl.loops4
         if tuple(sorted(abs(x) for x in w)) == (0, 0, 1)),
        key=lambda item: np.linalg.norm(
            np.mean(unwrap(loop_cl, item[0])[:-1], axis=0) - center
        ),
    )
    six, _ = min(
        ((p, w) for p, w in loop_cl.hexes if w == (0, 0, 0)),
        key=lambda item: np.linalg.norm(
            np.mean(unwrap(loop_cl, item[0])[:-1], axis=0) - center
        ),
    )
    fig.text(0.015, 0.965, "(a) cubic-16 loop geometry", ha="left", va="top",
             fontsize=8.5)
    for bounds, path, color, label, boundary in (
        ([0.005, 0.08, 0.275, 0.80], four, ORANGE, "wrapping 4-loop", True),
        ([0.285, 0.08, 0.275, 0.80], six, GREEN, "bulk hexagon", False),
    ):
        ax_loop = fig.add_axes(bounds, projection="3d")
        draw_cluster(ax_loop, loop_cl)
        draw_unit_cube(ax_loop)
        draw_loop(ax_loop, loop_cl, path, color, np.zeros(3), boundary)
        ax_loop.set_xlim(-0.04, 1.04)
        ax_loop.set_ylim(-0.04, 1.04)
        ax_loop.set_zlim(-0.04, 1.04)
        ax_loop.text2D(0.5, 0.88, label, color=color,
                       transform=ax_loop.transAxes, ha="center", fontsize=7.0)
        ax_loop.set_axis_off()
        ax_loop.set_box_aspect((1.0, 1.0, 1.0))
        ax_loop.set_proj_type("ortho")
        ax_loop.view_init(elev=24, azim=-45)

    ax = fig.add_axes([0.625, 0.20, 0.355, 0.68])
    temperatures = np.geomspace(1.5e-4, 0.04, 900)
    ax.plot(temperatures, R.specific_heat(e_bare, temperatures) / cl.n_sites,
            color=ORANGE, lw=1.45, label="periodic")
    ax.plot(temperatures, R.specific_heat(e_clean, temperatures) / cl.n_sites,
            color=GREEN, lw=1.7, label="winding-free")
    g4 = 4.0 * JPM_REFERENCE**2
    ghex = 12.0 * abs(JPM_REFERENCE) ** 3
    ax.axvline(g4, color=ORANGE, ls=":", lw=0.9)
    ax.axvline(ghex, color=GREEN, ls=":", lw=0.9)
    ax.text(g4 * 1.05, 0.94, r"$g_4$", color=ORANGE,
            transform=ax.get_xaxis_transform(), va="top")
    ax.text(ghex / 1.06, 0.94, r"$g_{\rm hex}$", color=GREEN,
            transform=ax.get_xaxis_transform(), ha="right", va="top")
    ax.set_xscale("log")
    ax.set_xlabel(r"$T/J_{zz}$")
    ax.set_ylabel(r"$C/N$")
    ax.set_title(r"(b) scale correction, $J_\pm/J_{zz}=-0.05$", loc="left")
    ax.legend(frameon=False, loc="center right", bbox_to_anchor=(1.0, 0.56),
              fontsize=5.5, handlelength=1.5)
    ax.spines[["top", "right"]].set_visible(False)
    save_figure(fig, "fig1_fcc32_origin")


def figure2(j_values: np.ndarray, spectra_bare: np.ndarray,
            spectrum_clean: np.ndarray, peak_bare: np.ndarray,
            peak_clean: float, mask_residual: float) -> None:
    fig = plt.figure(figsize=(3.35, 3.75))
    grid = fig.add_gridspec(2, 2, height_ratios=(1.0, 1.15), hspace=0.52,
                            wspace=0.52)
    ax = fig.add_subplot(grid[0, 0])
    x = np.arange(3)
    width = 0.34
    ax.bar(x - width / 2, np.ones(3), width, color=GREY, label="periodic")
    ax.bar(x + width / 2, [0.0, 0.0, 1.0], width, color=GREEN,
           label="projected")
    ax.plot(x[:2] + width / 2, [0.025, 0.025], "x", color=GREEN, ms=4.0,
            mew=1.0)
    ax.set_xticks(x, ["4-wrap", "6-wrap", "bulk hex."])
    ax.tick_params(axis="x", labelsize=6.2)
    ax.set_ylim(0, 1.13)
    ax.set_ylabel("fraction retained")
    ax.set_title("(a) process retention", loc="left")
    ax.legend(frameon=False, fontsize=5.3, ncol=2, loc="upper center",
              columnspacing=0.7, handlelength=1.2)
    ax.spines[["top", "right"]].set_visible(False)

    ax = fig.add_subplot(grid[1, :])
    n_modes = min(24, spectrum_clean.size - 1)
    modes = np.arange(1, n_modes + 1)
    colors = [BLUE, PURPLE, RED]
    selected = [0, len(j_values) // 2, len(j_values) - 1]
    for color, idx in zip(colors, selected):
        ax.plot(modes, spectra_bare[idx, 1:n_modes + 1], "o", ms=2.2,
                mfc="none", mec=color, alpha=0.72,
                label=rf"periodic ${j_values[idx]:.2f}$")
    ax.plot(modes, spectrum_clean[1:n_modes + 1], ".", ms=3.2,
            color="black", label="projected (all)")
    ax.set_xlabel("excitation index")
    ax.set_ylabel(r"$(E_n-E_0)/g_{\rm hex}$")
    ax.set_ylim(0, min(8.0, 1.08 * np.max(spectra_bare[selected, 1:n_modes + 1])))
    ax.set_title(r"(c) levels in units of $g_{\rm hex}$", loc="left")
    ax.legend(frameon=False, ncol=2, loc="upper center", fontsize=5.5,
              columnspacing=0.9, handletextpad=0.35)
    ax.spines[["top", "right"]].set_visible(False)

    ax = fig.add_subplot(grid[0, 1])
    ax.plot(j_values, peak_bare, "o-", color=ORANGE, lw=1.2, ms=3.2,
            label="periodic")
    ax.plot(j_values, np.full_like(j_values, peak_clean), "s-", color=GREEN,
            lw=1.4, ms=3.0, label="projected")
    ax.text(j_values[-1], peak_clean + 0.10, "0.521", color=GREEN,
            fontsize=5.7, ha="right", va="bottom")
    ax.set_xlabel(r"$|J_\pm|/J_{zz}$")
    ax.set_ylabel(r"$T_{\rm peak}/g_{\rm hex}$")
    ax.set_title(r"(b) $T_{\rm peak}/g_{\rm hex}$", loc="left")
    ax.legend(frameon=False, fontsize=5.8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.subplots_adjust(left=0.16, right=0.98, bottom=0.11, top=0.96)
    save_figure(fig, "fig2_fcc32_benchmark")


def figure3(q_labels: list[str], omega: np.ndarray,
            s_bare: np.ndarray, s_clean: np.ndarray) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(3.35, 3.55), sharex=True,
                             sharey=True)
    positive = np.concatenate([s_bare[s_bare > 0], s_clean[s_clean > 0]])
    vmax = np.quantile(positive, 0.995)
    x_edges = np.arange(9) - 0.5
    y_edges = np.concatenate(
        ([omega[0] - 0.5 * (omega[1] - omega[0])],
         0.5 * (omega[:-1] + omega[1:]),
         [omega[-1] + 0.5 * (omega[-1] - omega[-2])])
    )
    for row, (ax, spectrum, title) in enumerate(
        zip(axes, (s_bare, s_clean), ("periodic", "winding-free"))
    ):
        ax.pcolormesh(x_edges, y_edges, spectrum, cmap="magma", shading="flat",
                      vmin=0.0, vmax=vmax, rasterized=True)
        for boundary in x_edges[1:-1]:
            ax.axvline(boundary, color="white", lw=0.35, alpha=0.55)
        ax.axhline(2.0, color="white", ls=":", lw=0.7, alpha=0.8)
        ax.set_xlim(-0.5, 7.5)
        ax.set_ylim(0.0, omega[-1])
        ax.set_xticks(np.arange(8), q_labels, rotation=0)
        ax.set_title(f"({chr(97 + row)}) {title}", loc="left")
        if np.max(spectrum) < 1.0e-12:
            ax.text(0.5, 0.52, "no inelastic weight\nat allowed $\\mathbf{Q}$",
                    color="white", ha="center", va="center",
                    transform=ax.transAxes, fontsize=7.2)
    axes[1].set_xlabel("allowed FCC-32 momentum")
    axes[0].set_yticks([2.0, 4.0, 6.0], [])
    axes[1].set_yticks(
        [2.0, 4.0, 6.0],
        [r"$2g_{\rm hex}$", r"$4g_{\rm hex}$", r"$6g_{\rm hex}$"],
    )
    fig.supylabel(r"$\omega$", x=0.01)
    fig.text(0.5, 0.985,
             r"$S^{zz}(\mathbf{q},\omega)$, $J_\pm/J_{zz}=-0.05$",
             ha="center", va="top", fontsize=8.5)
    fig.tight_layout(rect=(0.05, 0, 1, 0.94), h_pad=0.35)
    save_figure(fig, "fig3_fcc32_dssf")


def read_digitized() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    path = OUT_DATA / "ce2hf2o7_smith2025_digitized.csv"
    rows: dict[str, list[tuple[float, float]]] = {}
    with path.open() as handle:
        data_lines = (line for line in handle if not line.startswith("#"))
        for row in csv.DictReader(data_lines):
            rows.setdefault(row["series"], []).append(
                (float(row["temperature_K"]), float(row["Cmag_J_molCe_K"]))
            )
    return {
        key: (np.array([p[0] for p in values]), np.array([p[1] for p in values]))
        for key, values in rows.items()
    }


def moment_matched_candidate(alpha32: float) -> np.ndarray:
    second_moment = float(np.dot(SMITH_A, SMITH_A))
    transverse_difference = SMITH_A[1] - SMITH_A[2]

    def ja(total: float) -> float:
        return np.sqrt(second_moment - 0.5 * (total**2 + transverse_difference**2))

    def mismatch(total: float) -> float:
        ghex = 12.0 * (total / 4.0) ** 3 / ja(total) ** 2
        return alpha32 * ghex / KB_MEV_PER_K - T2_K

    total = brentq(mismatch, 0.025, 0.060)
    return np.array(
        [ja(total), 0.5 * (total + transverse_difference),
         0.5 * (total - transverse_difference)]
    )


def physical_gauge_curve(spectrum_over_g: np.ndarray, ghex_mev: float,
                         temperatures: np.ndarray) -> np.ndarray:
    return R_GAS * R.specific_heat(
        spectrum_over_g * ghex_mev, KB_MEV_PER_K * temperatures
    ) / 32.0


def figure4(spectrum_clean_over_g: np.ndarray, alpha32: float,
            candidate: np.ndarray) -> None:
    digitized = read_digitized()
    t_exp, c_exp = digitized["experiment"]
    t_nlc, c_nlc = digitized["published_NLC_A"]
    jpm_a = -(SMITH_A[1] + SMITH_A[2]) / 4.0
    ghex_a = 12.0 * abs(jpm_a) ** 3 / SMITH_A[0] ** 2
    jpm_star = -(candidate[1] + candidate[2]) / 4.0
    ghex_star = 12.0 * abs(jpm_star) ** 3 / candidate[0] ** 2
    temperatures = np.geomspace(0.0035, 0.9, 1000)
    c_a = physical_gauge_curve(spectrum_clean_over_g, ghex_a, temperatures)
    c_star = physical_gauge_curve(spectrum_clean_over_g, ghex_star, temperatures)

    fig, axes = plt.subplots(2, 1, figsize=(3.35, 4.05),
                             gridspec_kw={"height_ratios": [1.75, 1.0]})
    ax = axes[0]
    ax.plot(t_exp[::2], c_exp[::2], "D", ms=2.3, color=BLUE,
            label="experiment (digitized)")
    ax.plot(t_nlc, c_nlc, color=PURPLE, lw=1.45, ls="--",
            label="published seventh-order NLC A")
    ax.plot(temperatures, c_a, color=ORANGE, lw=1.15, ls=":",
            label="FCC-32 gauge sector, A")
    ax.plot(temperatures, c_star, color=GREEN, lw=1.75,
            label=r"FCC-32 gauge sector, A$^\star$")
    ax.axvline(T2_K, color="black", lw=0.7, alpha=0.55)
    ax.set_xscale("log")
    ax.set_xlim(0.005, 1.0)
    ax.set_ylim(0.0, 2.35)
    ax.set_xlabel("temperature (K)")
    ax.set_ylabel(r"$C_{\rm mag}$ (J mol$_{\rm Ce}^{-1}$ K$^{-1}$)")
    ax.set_title(r"(a) high-$T$ NLC and low-$T$ gauge scales", loc="left")
    ax.legend(frameon=False, loc="lower left", fontsize=5.8, ncol=2,
              columnspacing=0.8, handlelength=1.8)
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[1]
    x = np.arange(3)
    width = 0.35
    ax.bar(x - width / 2, SMITH_A, width, color=PURPLE, label="NLC A")
    ax.bar(x + width / 2, candidate, width, color=GREEN,
           label=r"moment-matched A$^\star$")
    ax.set_xticks(x, [r"$J_a$", r"$J_b$", r"$J_c$"])
    ax.set_ylabel("exchange (meV)")
    ax.set_title("(b) constrained candidate", loc="left")
    ax.legend(frameon=False, loc="upper right", fontsize=6.0)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(h_pad=0.8)
    save_figure(fig, "fig4_ce2hf2o7")


def main() -> None:
    print("building FCC-32 cluster and order-2/order-3 operator", flush=True)
    cl = R.build_cluster("fcc", (2, 2, 2))
    pt = R.sw_order23(cl, verbose=True)
    h_bare = R.assemble(cl, pt, JPM_REFERENCE, "all")
    h_clean = R.assemble(cl, pt, JPM_REFERENCE, "delta0")

    labels = parity_labels(cl)
    h_mask = h_bare * (labels[:, None] == labels[None, :])
    mask_residual = float(np.max(np.abs(h_mask - h_clean)))
    if mask_residual > 1e-13:
        raise RuntimeError(f"FCC-32 mask identity failed: {mask_residual}")

    print("diagonalizing reference periodic and projected operators", flush=True)
    e_bare, u_bare = np.linalg.eigh(h_bare)
    e_clean, u_clean = np.linalg.eigh(h_clean)
    figure1(cl, e_bare, e_clean)

    j_values = np.array([0.03, 0.04, 0.05, 0.06, 0.08, 0.10])
    spectra_bare = []
    peak_bare = []
    for j in j_values:
        print(f"benchmark diagonalization |Jpm|={j:.2f}", flush=True)
        if np.isclose(j, abs(JPM_REFERENCE)):
            energies = e_bare
        else:
            energies = np.linalg.eigvalsh(R.assemble(cl, pt, -j, "all"))
        ghex = 12.0 * j**3
        spectra_bare.append((energies - energies[0]) / ghex)
        peak_bare.append(refined_peak(energies, ghex) / ghex)
    spectra_bare = np.asarray(spectra_bare)
    ghex_reference = 12.0 * abs(JPM_REFERENCE) ** 3
    spectrum_clean = (e_clean - e_clean[0]) / ghex_reference
    peak_clean = refined_peak(e_clean, ghex_reference) / ghex_reference
    figure2(j_values, spectra_bare, spectrum_clean, np.asarray(peak_bare),
            peak_clean, mask_residual)

    qs, q_labels = allowed_momenta(cl)
    omega = np.linspace(0.0, 8.0, 640)
    print("computing FCC-32 DSSF at eight allowed momenta", flush=True)
    s_bare = dssf(cl, e_bare, u_bare, qs, omega, ghex_reference)
    s_clean = dssf(cl, e_clean, u_clean, qs, omega, ghex_reference)
    figure3(q_labels, omega, s_bare, s_clean)

    alpha32 = peak_clean
    candidate = moment_matched_candidate(alpha32)
    figure4(spectrum_clean, alpha32, candidate)

    np.savez_compressed(
        OUT_DATA / "fcc32_prl_results.npz",
        j_values=j_values,
        spectra_bare_over_ghex=spectra_bare,
        spectrum_clean_over_ghex=spectrum_clean,
        peak_bare_over_ghex=np.asarray(peak_bare),
        peak_clean_over_ghex=peak_clean,
        allowed_momenta=qs,
        omega_over_ghex=omega,
        dssf_bare=s_bare,
        dssf_clean=s_clean,
        reference_eigenvalues_bare=e_bare,
        reference_eigenvalues_clean=e_clean,
    )
    summary = {
        "cluster": {"sites": 32, "ice_states": 2970, "shape": "2x2x2 FCC"},
        "mask_identity_max_residual": mask_residual,
        "nonzero_matrix_elements": {
            "periodic": int(np.count_nonzero(np.abs(h_bare) > 1e-14)),
            "projected": int(np.count_nonzero(np.abs(h_clean) > 1e-14)),
        },
        "reference_Jpm_over_Jzz": JPM_REFERENCE,
        "allowed_momenta": qs.tolist(),
        "alpha32_Tpeak_over_ghex": float(alpha32),
        "smith_NLC_A_meV": {
            "Ja": float(SMITH_A[0]), "Jb": float(SMITH_A[1]),
            "Jc": float(SMITH_A[2]),
        },
        "moment_matched_candidate_meV": {
            "Ja": float(candidate[0]), "Jb": float(candidate[1]),
            "Jc": float(candidate[2]),
            "Jpm": float(-(candidate[1] + candidate[2]) / 4.0),
            "constraint": "matches sum(J_alpha^2) of NLC A and the 25 mK FCC-32 clean ring peak; not a seventh-order NLC refit",
        },
        "experimental_data": "digitized for visual comparison from Smith et al. PRL 135, 086702 (2025), Fig. 2(a)",
        "dssf_observable": "four-sublattice trace of the longitudinal pseudospin correlation; no neutron-polarization factors",
        "dssf_maximum": {
            "periodic": float(np.max(s_bare)),
            "projected": float(np.max(s_clean)),
        },
    }
    (OUT_DATA / "fcc32_prl_results.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
