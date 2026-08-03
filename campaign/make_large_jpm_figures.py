#!/usr/bin/env python3
"""Figures for the large-|J_pm| Feshbach study.

Reads the JSON records written by run_feshbach_anatomy, run_truncated_feshbach,
run_corner_coherence, run_strong_coupling_gs, run_counterterm_observables,
run_winding_residual and run_exact_counterterm; writes four PDFs into
paper/figs.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "campaign" / "outputs"
FIGS = ROOT / "paper" / "figs"
FIGS.mkdir(parents=True, exist_ok=True)

# fixed categorical order, validated for CVD separation on a light surface
BLUE, VERM, GREEN, ORANGE, PINK = ("#0072B2", "#D55E00", "#009E73",
                                   "#E69F00", "#CC79A7")
GREY = "#5c5c5c"

plt.rcParams.update({
    "font.size": 8.5,
    "axes.labelsize": 8.5,
    "axes.titlesize": 9,
    "legend.fontsize": 7.4,
    "xtick.labelsize": 7.6,
    "ytick.labelsize": 7.6,
    "axes.linewidth": 0.7,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "lines.linewidth": 1.5,
    "lines.markersize": 4.0,
    "axes.grid": True,
    "grid.alpha": 0.22,
    "grid.linewidth": 0.5,
    "figure.dpi": 160,
    "savefig.bbox": "tight",
})


def load(pattern: str) -> list[dict]:
    records = []
    for path in sorted(OUT.glob(pattern)):
        with open(path) as handle:
            records.append(json.load(handle))
    records.sort(key=lambda r: r["jpm"])
    return records


def column(records, key, filter_negative=True):
    x, y = [], []
    for record in records:
        if filter_negative and record["jpm"] >= 0:
            continue
        value = record.get(key)
        if value is None or (isinstance(value, float) and np.isnan(value)):
            continue
        x.append(abs(record["jpm"]))
        y.append(value)
    return np.asarray(x), np.asarray(y)


def finish(axis, title, xlabel=r"$|J_\pm|/J_{zz}$", ylabel=None, legend=True):
    axis.set_title(title, loc="left", pad=4)
    axis.set_xlabel(xlabel)
    if ylabel:
        axis.set_ylabel(ylabel)
    if legend:
        axis.legend(frameon=False, handlelength=1.5, borderaxespad=0.3)


# ----------------------------------------------------------------- figure 1
def figure_anatomy():
    anatomy = load("feshbach_anatomy/anatomy_*.json")
    c16 = [r for r in load("truncated_feshbach/trunc_cubic16_L1_*.json")]
    f32 = [r for r in load("truncated_feshbach/trunc_fcc32_L1_*.json")]

    figure, axes = plt.subplots(1, 3, figsize=(10.4, 3.05))

    ax = axes[0]
    classes = [("wind4_axial", VERM, "winding 4-cycle"),
               ("hexagon", BLUE, "hexagon (physical)"),
               ("string8", GREEN, "8-strings"),
               ("string10", ORANGE, "10-strings")]
    for key, colour, label in classes:
        x, y = [], []
        for record in anatomy:
            if record["jpm"] >= 0:
                continue
            entry = record.get(f"class::{key}")
            if entry:
                x.append(abs(record["jpm"]))
                y.append(abs(entry["mean"]))
        ax.loglog(x, y, "o-", color=colour, label=label)
    grid = np.asarray([abs(r["jpm"]) for r in anatomy if r["jpm"] < 0])
    ax.loglog(grid, 4 * grid**2, ":", color=GREY, lw=1.1,
              label=r"$4J_\pm^2$ (2nd order)")
    ax.loglog(grid, 12 * grid**3, "--", color=GREY, lw=1.1,
              label=r"$12|J_\pm|^3$ ($g_6$)")
    finish(ax, "(a) dressed ice-block amplitudes",
           ylabel=r"$|F(z_0)_{ab}|/J_{zz}$")

    ax = axes[1]
    x, y = column(anatomy, "g_eff")
    ax.plot(x, np.abs(y) / (12 * x**3), "o-", color=BLUE,
            label="hexagon, cubic-16 (exact)")
    x, y = column(c16, "g_over_g6")
    ax.plot(x, np.abs(y), "s--", color=BLUE, alpha=0.55,
            label="cubic-16 (1 pair)")
    x, y = column(f32, "g_over_g6")
    ax.plot(x, np.abs(y), "^-", color=GREEN, label="FCC-32 (1 pair)")
    x, y = column(c16, "wind_over_bare")
    ax.plot(x, y, "s--", color=VERM, alpha=0.55, label="4-cycle, cubic-16")
    x, y = column(f32, "wind_over_bare")
    ax.plot(x, y, "^-", color=ORANGE, label="4-cycle, FCC-32")
    ax.set_ylim(0, 1.05)
    finish(ax, "(b) renormalisation, and its size dependence",
           ylabel=r"amplitude / perturbative value")

    ax = axes[2]
    x, y = column(anatomy, "long_over_hex")
    ax.plot(x, y, "o-", color=BLUE, label=r"$\|$long strings$\|/\|$hexagon$\|$")
    x, y = column(anatomy, "rk_ratio")
    ax.plot(x, np.abs(y), "s-", color=VERM, label=r"$|V/g|$ (RK ratio)")
    x, y = column(anatomy, "z_ground")
    ax.plot(x, y, "^-", color=GREEN, label=r"residue $Z$")
    ax.set_ylim(0, 1.05)
    finish(ax, "(c) what changes in the effective theory", ylabel="ratio")

    figure.tight_layout()
    figure.savefig(FIGS / "fig_feshbach_anatomy.pdf")
    plt.close(figure)


# ----------------------------------------------------------------- figure 2
def figure_coherence():
    coherence = load("corner_coherence/coherence_*.json")
    figure, axes = plt.subplots(1, 3, figsize=(10.4, 3.05))

    ax = axes[0]
    x, y = column(coherence, "band_coherence_min")
    ax.plot(x, y, "o-", color=BLUE, label=r"corner coherence $\min\,\mathrm{eig}\,K$")
    x, y = column(coherence, "manifold_return_mean")
    ax.plot(x, y, "s-", color=VERM, label="mean corner return")
    x, y = column(coherence, "band_ice_weight")
    ax.plot(x, y, "^--", color=GREEN, label="band ice weight")
    ax.set_ylim(0, 1.05)
    finish(ax, "(a) coherence is the ice weight in disguise")

    ax = axes[1]
    x, y = column(coherence, "manifold_return_mean")
    ax.plot(x, y, "o-", color=BLUE, label=r"manifold: $\langle v|U_c^\dagger P U_c|v\rangle$")
    x, y = column(coherence, "vector_return_mean")
    ax.plot(x, y, "s-", color=VERM, label=r"vector: $|\langle v|U_c|v\rangle|^2$")
    ax.set_ylim(0, 1.05)
    finish(ax, "(b) the corner action reshuffles the band")

    ax = axes[2]
    wanted = [-0.005, -0.05, -0.11]
    chosen = [min(coherence, key=lambda r: abs(r["jpm"] - w)) for w in wanted]
    for record, colour, marker in zip(chosen, (BLUE, VERM, GREEN),
                                      ("o", "s", "^")):
        greedy = record["greedy"]
        ax.semilogy([g["dropped"] for g in greedy],
                    [max(g["sigma_min"], 1e-6) for g in greedy], marker + "-",
                    color=colour, label=rf"$J_\pm={record['jpm']:+.3f}$")
    ax.set_ylim(1e-6, 2.0)
    finish(ax, "(c) per-vector selection destroys closure",
           xlabel="band vectors removed (largest transport variance first)",
           ylabel=r"$\sigma_{\min}(A_c)$ of the remainder")

    figure.tight_layout()
    figure.savefig(FIGS / "fig_corner_coherence.pdf")
    plt.close(figure)


# ----------------------------------------------------------------- figure 3
def figure_protocol():
    anatomy = load("feshbach_anatomy/anatomy_*.json")
    classct = load("counterterm_observables/ctobs_*.json")
    exact = load("exact_counterterm/exact_ct_*.json")
    residual = load("winding_residual/residual_*.json")
    raw = load("strong_coupling/gs_*.json")

    figure, axes = plt.subplots(1, 3, figsize=(10.4, 3.05))

    ax = axes[0]
    x, y = column(anatomy, "ring_peak_over_g6")
    ax.plot(x, y, "o-", color=BLUE, label="masked Feshbach band")
    if exact:
        x, y = column(exact, "ring_peak_over_g6")
        ax.plot(x, y, "d-", color=GREEN, label=r"$H+C$ (exact counterterm)")
    x, y = column(classct, "ring_peak_over_g6")
    ax.plot(x, y, "s--", color=ORANGE, label="loop-class counterterms")
    finish(ax, "(a) the ring scale does not collapse",
           ylabel=r"$T_{\rm peak}/g_6$")

    ax = axes[1]
    x, y = column(raw, "hexagon_flux")
    ax.plot(x, y, "o-", color=VERM, label="raw periodic ED")
    x, y = column(classct, "hexagon_flux")
    ax.plot(x, y, "s--", color=ORANGE, label="loop-class counterterms")
    if exact:
        x, y = column(exact, "hexagon_flux")
        ax.plot(x, y, "d-", color=GREEN, label=r"$H+C$")
    ax.axhline(0.0, color=GREY, lw=0.8)
    ax.set_xlim(0.0, 0.37)
    finish(ax, r"(b) the $\pi$-flux law, lost and restored",
           ylabel=r"$\langle O_{\rm hex}\rangle$")

    ax = axes[2]
    x, y = column(residual, "winding_fraction_bare")
    ax.plot(x, y, "o-", color=VERM, label="bare $H$")
    x, y = column(residual, "winding_fraction_corrected")
    ax.plot(x, y, "s--", color=ORANGE, label="loop-class counterterms")
    ax.plot(x, np.zeros_like(x), "d-", color=GREEN,
            label=r"$H+C$ (zero by construction)")
    ax.set_ylim(-0.05, 1.05)
    finish(ax, "(c) transporting fraction of the ice-block map",
           ylabel=r"$\|F_{\rm cross}\|/\|F_{\rm offdiag}\|$")

    figure.tight_layout()
    figure.savefig(FIGS / "fig_winding_free_protocol.pdf")
    plt.close(figure)


# ----------------------------------------------------------------- figure 4
def figure_physics():
    raw = load("strong_coupling/gs_*.json")
    anatomy = load("feshbach_anatomy/anatomy_*.json")

    figure, axes = plt.subplots(1, 3, figsize=(10.4, 3.05))

    ax = axes[0]
    x, y = column(raw, "ice_weight")
    ax.plot(x, y, "o-", color=BLUE, label="ground-state ice weight")
    x, y = column(anatomy, "z_ground")
    ax.plot(x, y, "s-", color=VERM, label=r"Feshbach residue $Z$")
    x, y = column(raw, "defect_number")
    ax.plot(x, y / 16.0, "^-", color=GREEN,
            label="charged tetrahedra / site")
    finish(ax, "(a) dressing is smooth and never stops")

    ax = axes[1]
    positive = [r for r in raw if r["jpm"] > 0]
    negative = [r for r in raw if r["jpm"] < 0]
    x, y = column(negative, "condensate_fraction")
    ax.plot(x, y, "o-", color=BLUE, label=r"$\pi$ flux ($J_\pm<0$)")
    x = np.asarray([r["jpm"] for r in positive])
    y = np.asarray([r["condensate_fraction"] for r in positive])
    ax.plot(x, y, "s-", color=VERM, label=r"zero flux ($J_\pm>0$)")
    ax.axhline(1.0 / 32.0, color=GREY, lw=0.9, ls=":")
    ax.annotate("uncondensed, $1/2N$", (0.55, 1.0 / 32.0), xytext=(0, 4),
                textcoords="offset points", color=GREY, fontsize=7)
    finish(ax, "(b) no transverse condensate at $\\pi$ flux",
           ylabel=r"$n_0/N$")

    ax = axes[2]
    x, y = column(raw, "winding4_amplitude")
    ax.plot(x, y, "o-", color=VERM, label=r"$\langle O_4\rangle$ (winding)")
    x, y = column(raw, "hexagon_flux")
    ax.plot(x, y, "s-", color=BLUE, label=r"$\langle O_{\rm hex}\rangle$")
    ax.axhline(0.0, color=GREY, lw=0.8)
    finish(ax, "(c) the raw torus ground state is artifact-dominated",
           ylabel="loop amplitude")

    figure.tight_layout()
    figure.savefig(FIGS / "fig_large_jpm_physics.pdf")
    plt.close(figure)


if __name__ == "__main__":
    figure_anatomy()
    figure_coherence()
    figure_protocol()
    figure_physics()
    print("figures written to", FIGS)
