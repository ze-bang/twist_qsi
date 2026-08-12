#!/usr/bin/env python3
"""Two figures for the single-spine masked-Feshbach manuscript.

Figure 1 contains the method and its two benchmarks: published sign-free QMC
at Jpm/Jzz=+0.046, and the weak-coupling peak exponent.  The only corrected
spectrum used is obtained by replacing the lowest 90 periodic levels with the
self-consistent roots of the transport-masked Feshbach map.  No counterterm or
polar-pullback data enter.

Figure 2 shows the measured simple-loop amplitudes directly against coupling,
together with their leading degenerate-perturbation-theory values.  L is the
number of flipped spins (and the bond perimeter) of a single alternating loop.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyArrowPatch, Rectangle  # noqa: E402
from matplotlib.ticker import NullFormatter  # noqa: E402
import numpy as np  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "campaign" / "outputs"
METHOD = OUT / "method_comparison"
TOMO = OUT / "wp1_tomography"
FIGS = ROOT / "paper" / "figs"

INK = "#17191d"
MUTED = "#626872"
GRID = "#e2e5e9"
PERIODIC = "#a95a45"
MASKED = "#356399"
QMC = "#17191d"
LOOP = {8: "#356399", 10: "#a95a45", 12: "#8b6f2f"}
MARKER = {8: "o", 10: "s", 12: "^"}

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "mathtext.fontset": "cm",
    "font.size": 8.6,
    "axes.labelsize": 8.8,
    "axes.titlesize": 9.0,
    "legend.fontsize": 7.3,
    "xtick.labelsize": 7.6,
    "ytick.labelsize": 7.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.7,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "figure.dpi": 180,
    "savefig.bbox": "tight",
})


def method_cartoon(axis: plt.Axes) -> None:
    """Draw the fold and the sector mask without introducing extra methods."""
    axis.set_axis_off()
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)

    # Microscopic P/Q block decomposition.
    x0, y0, w, h = 0.03, 0.57, 0.25, 0.29
    axis.add_patch(Rectangle((x0, y0), w, h, fill=False, ec=INK, lw=0.9))
    axis.plot([x0 + 0.43 * w] * 2, [y0, y0 + h], color=INK, lw=0.7)
    axis.plot([x0, x0 + w], [y0 + 0.43 * h] * 2, color=INK, lw=0.7)
    axis.text(x0 + 0.21 * w, y0 + 0.70 * h, r"$PHP$", ha="center", va="center")
    axis.text(x0 + 0.71 * w, y0 + 0.70 * h, r"$PHQ$", ha="center", va="center")
    axis.text(x0 + 0.21 * w, y0 + 0.21 * h, r"$QHP$", ha="center", va="center")
    axis.text(x0 + 0.71 * w, y0 + 0.21 * h, r"$QHQ$", ha="center", va="center")
    axis.text(x0 + 0.5 * w, y0 + h + 0.065, "microscopic Hamiltonian",
              ha="center", color=MUTED, fontsize=7.2)
    axis.text(x0 + 0.10 * w, y0 - 0.055, r"ice $P$", ha="center", color=MUTED)
    axis.text(x0 + 0.73 * w, y0 - 0.055, r"spinons $Q$", ha="center", color=MUTED)

    axis.add_patch(FancyArrowPatch((0.30, 0.715), (0.43, 0.715),
                                  arrowstyle="-|>", mutation_scale=9,
                                  color=INK, lw=0.9))
    axis.text(0.365, 0.76, "fold out $Q$", ha="center", fontsize=7.2)
    axis.text(0.365, 0.65, r"$F(z)$ on $P$",
              ha="center", fontsize=7.0)

    # Ice block sorted by covering-lattice transport sector.
    mx, my, mw, mh = 0.46, 0.52, 0.25, 0.37
    axis.add_patch(Rectangle((mx, my), mw, mh, fill=False, ec=INK, lw=0.9))
    cuts = (0.31, 0.66)
    for cut in cuts:
        axis.plot([mx + cut * mw] * 2, [my, my + mh], color=INK, lw=0.6)
        axis.plot([mx, mx + mw], [my + cut * mh] * 2, color=INK, lw=0.6)
    for i, left in enumerate((0.0, cuts[0], cuts[1])):
        right = (cuts + (1.0,))[i]
        axis.add_patch(Rectangle((mx + left * mw, my + left * mh),
                                 (right-left) * mw, (right-left) * mh,
                                 fc=MASKED, ec="none", alpha=0.30))
    for px, py in ((0.56, 0.80), (0.64, 0.60), (0.50, 0.69), (0.66, 0.75)):
        axis.plot(px, py, "x", color=PERIODIC, ms=4.0, mew=1.0)
    axis.text(mx + 0.5 * mw, my + mh + 0.065,
              r"ice states sorted by transport $\mathcal{T}$",
              ha="center", color=MUTED, fontsize=7.2)
    axis.text(mx + 0.5 * mw, my - 0.065,
              "red: torus-changing matrix elements", ha="center",
              color=PERIODIC, fontsize=7.0)

    axis.add_patch(FancyArrowPatch((0.73, 0.715), (0.83, 0.715),
                                  arrowstyle="-|>", mutation_scale=9,
                                  color=INK, lw=0.9))
    axis.text(0.78, 0.76, r"keep $\mathcal{T}_a=\mathcal{T}_b$",
              ha="center", fontsize=7.0)

    fx, fy, fw, fh = 0.85, 0.58, 0.13, 0.27
    axis.add_patch(Rectangle((fx, fy), fw, fh, fill=False, ec=INK, lw=0.9))
    for cut in cuts:
        axis.plot([fx + cut * fw] * 2, [fy, fy + fh], color=INK, lw=0.55)
        axis.plot([fx, fx + fw], [fy + cut * fh] * 2, color=INK, lw=0.55)
    for i, left in enumerate((0.0, cuts[0], cuts[1])):
        right = (cuts + (1.0,))[i]
        axis.add_patch(Rectangle((fx + left * fw, fy + left * fh),
                                 (right-left) * fw, (right-left) * fh,
                                 fc=MASKED, ec="none", alpha=0.45))
    axis.text(fx + 0.5 * fw, fy - 0.065, r"$\Delta[F(z)]$",
              ha="center", color=MASKED, fontsize=8.2)

    axis.text(0.03, 0.38, "What the mask means", weight="bold", fontsize=8.0)
    axis.text(0.03, 0.29,
              r"$\mathcal{T}_a=\mathcal{T}_b$: a local loop closes on the covering lattice $\rightarrow$ keep",
              fontsize=7.25, color=MASKED)
    axis.text(0.03, 0.20,
              r"$\mathcal{T}_a\ne\mathcal{T}_b$: a path closes only through a periodic image $\rightarrow$ remove",
              fontsize=7.25, color=PERIODIC)
    axis.text(0.03, 0.07,
              "Self-consistent roots of the remaining blocks define the finite-volume theory.",
              fontsize=7.15, color=INK)
    axis.set_title("(a) transport-masked Feshbach map", loc="left", pad=2)


def qmc_benchmark(axis: plt.Axes) -> None:
    data = np.load(METHOD / "compare_+0p046.npz")
    qmc = np.genfromtxt(
        ROOT / "campaign" / "data" / "huang_2018_qmc_jpm_0p046.csv",
        delimiter=",", names=True)
    grid = np.asarray(data["grid"])
    axis.plot(grid, data["periodic_curve"], color=PERIODIC, lw=1.35,
              label="periodic ED")
    axis.plot(grid, data["feshbach_curve"], color=MASKED, lw=1.55,
              label="masked Feshbach")
    axis.plot(qmc["temperature_over_Jzz"], qmc["heat_capacity_per_site"],
              "o", color=QMC, mfc="white", mew=0.75, ms=2.4,
              label="QMC (Huang et al.)", zorder=4)
    qmc_peak = qmc["temperature_over_Jzz"][
        np.argmax(qmc["heat_capacity_per_site"])]
    periodic_peak = 0.025519193373413688
    masked_peak = 0.0026505703858804787
    for value, color in ((qmc_peak, QMC), (periodic_peak, PERIODIC),
                         (masked_peak, MASKED)):
        axis.axvline(value, color=color, lw=0.7, ls=":")
    axis.set_xscale("log")
    axis.set_xlim(6.5e-4, 0.55)
    axis.set_ylim(0, 0.31)
    axis.set_xlabel(r"$T/J_{zz}$")
    axis.set_ylabel(r"$C/N$")
    axis.grid(True, color=GRID, lw=0.55)
    axis.legend(frameon=False, loc="upper center", ncol=1)
    axis.text(0.04, 0.08, r"low-$T$ peak: $23.2\times\to2.41\times$ QMC",
              transform=axis.transAxes, fontsize=7.0)
    axis.set_title(r"(b) sign-free benchmark, $J_\pm/J_{zz}=+0.046$",
                   loc="left")


def peak_scaling(axis: plt.Axes) -> None:
    records = []
    for coupling in (-0.01, -0.02, -0.03, -0.04, -0.05):
        tag = f"{coupling:+.3f}".replace(".", "p")
        records.append(json.loads((METHOD / f"compare_{tag}.json").read_text()))
    x = np.abs([r["jpm"] for r in records])
    periodic = np.array([r["periodic_gauge"] for r in records])
    masked = np.array([r["feshbach_gauge"] for r in records])
    dense = np.geomspace(0.008, 0.06, 200)
    axis.loglog(x, periodic, "s-", color=PERIODIC, ms=3.5, lw=1.2,
                label="periodic ED")
    axis.loglog(x, masked, "o-", color=MASKED, ms=3.7, lw=1.3,
                label="masked Feshbach")
    axis.loglog(dense, 4 * dense**2, ":", color=PERIODIC, lw=1.0,
                label=r"$4J_\pm^2/J_{zz}$")
    axis.loglog(dense, 12 * dense**3, "--", color=MASKED, lw=1.0,
                label=r"$12|J_\pm|^3/J_{zz}^2$")
    axis.set_xlim(0.008, 0.06)
    axis.set_ylim(6e-6, 4e-2)
    axis.set_xticks([0.01, 0.02, 0.04, 0.06])
    axis.set_xticklabels([
        r"$10^{-2}$", r"$2\times10^{-2}$",
        r"$4\times10^{-2}$", r"$6\times10^{-2}$",
    ])
    axis.xaxis.set_minor_formatter(NullFormatter())
    axis.set_xlabel(r"$|J_\pm|/J_{zz}$")
    axis.set_ylabel(r"$T_{\rm peak}/J_{zz}$")
    axis.grid(True, which="both", color=GRID, lw=0.5)
    axis.legend(frameon=False, loc="upper left", fontsize=6.7)
    axis.set_title(r"(c) false $J_\pm^2$ scale removed", loc="left")


def method_figure() -> None:
    figure = plt.figure(figsize=(7.05, 5.0))
    grid = figure.add_gridspec(2, 2, height_ratios=(1.0, 0.92),
                               left=0.055, right=0.985, bottom=0.09,
                               top=0.965, hspace=0.27, wspace=0.29)
    cartoon = figure.add_subplot(grid[0, :])
    qmc = figure.add_subplot(grid[1, 0])
    scaling = figure.add_subplot(grid[1, 1])
    method_cartoon(cartoon)
    qmc_benchmark(qmc)
    peak_scaling(scaling)
    FIGS.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGS / "fig_method_benchmark.pdf")
    figure.savefig(FIGS / "fig_method_benchmark.png", dpi=220)
    plt.close(figure)


def perturbative_ratio(length: int, x: np.ndarray) -> np.ndarray:
    """Leading K_L/K_6 for one alternating simple loop."""
    if length == 8:
        return (10.0 / 3.0) * x
    if length == 10:
        return (35.0 / 3.0) * x**2
    if length == 12:
        return 42.0 * x**3
    raise ValueError(length)


def loop_figure() -> None:
    records = []
    for path in sorted(TOMO.glob("tomo_fcc32_L2_-0p*.json")):
        if "_n" in path.stem:
            continue
        record = json.loads(path.read_text())
        if "k8_over_k6" in record:
            records.append(record)
    records.sort(key=lambda r: abs(r["jpm"]))
    x = np.array([abs(r["jpm"]) for r in records])

    figure, axis = plt.subplots(figsize=(6.7, 3.65))
    dense = np.linspace(0.02, 0.31, 300)
    for length in (8, 10, 12):
        y = np.array([r[f"k{length}_over_k6"] for r in records])
        axis.plot(x, y, marker=MARKER[length], color=LOOP[length],
                  ms=4.2, lw=1.35, label=rf"$K_{{{length}}}/K_6$",
                  zorder=3)
        axis.plot(dense, perturbative_ratio(length, dense), "--",
                  color=LOOP[length], lw=1.05, alpha=0.72, zorder=2)

    # Matched deeper-complement checks at the two strongest couplings.
    for jpm in (-0.20, -0.30):
        tag = f"{jpm:+.3f}".replace(".", "p")
        path = TOMO / f"tomo_fcc32_L3_{tag}_n512.json"
        record = json.loads(path.read_text())
        for length in (8, 10, 12):
            axis.plot(abs(jpm), record[f"k{length}_over_k6"],
                      marker=MARKER[length], ms=7.0, mfc="white",
                      mec=LOOP[length], mew=1.2, ls="none", zorder=4)

    axis.set_yscale("log")
    axis.set_xlim(0.02, 0.315)
    axis.set_ylim(4e-4, 1.35)
    axis.set_xlabel(r"$|J_\pm|/J_{zz}$")
    axis.set_ylabel(r"simple-loop amplitude relative to $K_6$")
    axis.grid(True, which="both", color=GRID, lw=0.55)
    solid = [Line2D([], [], color=LOOP[L], marker=MARKER[L], lw=1.3,
                    label=rf"$K_{{{L}}}/K_6$") for L in (8, 10, 12)]
    guides = [Line2D([], [], color=MUTED, ls="--", lw=1.0,
                     label="leading perturbation theory"),
              Line2D([], [], color=MUTED, marker="o", mfc="white", ls="none",
                     label=r"deeper virtual space ($d=3$)")]
    axis.legend(handles=solid + guides, frameon=False, ncol=2,
                loc="lower right")
    axis.text(0.03, 0.94,
              r"$K_L^{(0)}=2\binom{L-2}{L/2-1}|J_\pm|^{L/2}/J_{zz}^{L/2-1}$",
              transform=axis.transAxes, va="top", fontsize=8.0)
    axis.text(0.03, 0.83,
              r"At $-0.30$ ($d=3$): $K_8/K_6=0.262$, $K_{10}/K_6=0.0617$, $K_{12}/K_6=0.0204$",
              transform=axis.transAxes, va="top", fontsize=7.5)
    axis.set_title("Nonperturbative suppression of longer simple loops",
                   loc="left")
    figure.tight_layout()
    figure.savefig(FIGS / "fig_loop_ratios.pdf")
    figure.savefig(FIGS / "fig_loop_ratios.png", dpi=220)
    plt.close(figure)


def main() -> None:
    method_figure()
    loop_figure()
    print("wrote", FIGS / "fig_method_benchmark.pdf")
    print("wrote", FIGS / "fig_loop_ratios.pdf")


if __name__ == "__main__":
    main()
