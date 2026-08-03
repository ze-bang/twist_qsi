#!/usr/bin/env python3
"""Twist-averaged heat capacity versus the winding-free construction.

Panel (a): lower-peak position vs Jpm for bare periodic ED, the observable
twist average, and the winding-free band, against the perturbative scales g6
and g4.  Panel (b): C(T) at a representative coupling, with the individual
twist curves' envelope showing what the average is made of.

Reads ``campaign/outputs/twist_average_scan_m2.json`` and the matching curves
archive; writes ``campaign/outputs/fig_twist_average.{pdf,png}``.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "campaign" / "outputs"

BARE = "#a95a45"
AVG = "#7a5b9e"
SINGLE = "#3f8f5f"
CLEAN = "#356399"
G6C = "0.45"
G4C = "#c47a5f"

plt.rcParams.update({
    "font.family": "serif", "mathtext.fontset": "cm",
    "font.size": 8.5, "axes.labelsize": 8.5, "axes.titlesize": 8.5,
    "legend.fontsize": 7.0, "xtick.labelsize": 7.6, "ytick.labelsize": 7.6,
    "axes.linewidth": 0.6, "legend.frameon": False,
    "figure.facecolor": "white", "savefig.facecolor": "white", "savefig.dpi": 400,
})

RING_CEILING = 0.1  # a "lower" maximum above this is the ice peak, not the ring
SHOWCASE_JPM = -0.05


def peak(record: dict, label: str) -> float:
    value = record.get(label, {}).get("T_low_Jzz", np.nan)
    return value if value < RING_CEILING else np.nan


def main() -> None:
    data = json.loads((OUTPUT / "twist_average_scan_m2.json").read_text())
    points = sorted(data["points"], key=lambda p: p["jpm"])
    curves = np.load(OUTPUT / "twist_average_curves_m2.npz")
    single_path = OUTPUT / "twist_average_scan_m1.json"
    single_points = {}
    if single_path.exists():
        single_points = {
            round(p["jpm"], 6): p
            for p in json.loads(single_path.read_text())["points"]
        }

    figure, (ax, bx) = plt.subplots(
        1, 2, figsize=(6.9, 2.6), gridspec_kw={"width_ratios": (1.15, 1.0)}
    )

    x = np.array([p["jpm"] for p in points])
    bare = np.array([peak(p, "bare") for p in points])
    average = np.array([peak(p, "average") for p in points])
    clean = np.array([peak(p, "winding_free") for p in points])

    dense = np.linspace(-0.32, 0.06, 400)
    ax.plot(dense, 12.0 * np.abs(dense) ** 3, color=G6C, lw=1.1, ls=(0, (5, 1.6)),
            label=r"$g_6=12|J_\pm|^3/J_{zz}^2$")
    ax.plot(dense, 4.0 * dense ** 2, color=G4C, lw=1.0, ls=(0, (1.5, 1.5)),
            label=r"$g_4=4J_\pm^2/J_{zz}$")
    single = np.array([
        peak(single_points.get(round(p["jpm"], 6), {}), "average") for p in points
    ])
    for sign in (-1, 1):
        m = np.sign(x) == sign
        first = sign < 0
        ax.plot(x[m], bare[m], color=BARE, lw=1.2, marker="s", ms=2.6,
                label="periodic ED" if first else None)
        ax.plot(x[m], average[m], color=AVG, lw=1.3, marker="D", ms=2.6,
                label="twist average" if first else None)
        if np.any(np.isfinite(single[m])):
            ax.plot(x[m], single[m], color=SINGLE, lw=1.3, marker="^", ms=2.8,
                    label=r"single twist $(\pi/2)^3$" if first else None)
        ax.plot(x[m], clean[m], color=CLEAN, lw=1.4, marker="o", ms=3.0,
                label="winding-free" if first else None)

    ax.axvline(0.0, color="0.8", lw=0.6)
    ax.set_yscale("log")
    ax.set_xlim(-0.32, 0.06)
    ax.set_ylim(6e-6, 0.6)
    ax.set_xlabel(r"$J_\pm/J_{zz}$")
    ax.set_ylabel(r"lower-peak position $T/J_{zz}$")
    ax.legend(loc="lower left", handlelength=1.7, borderaxespad=0.4,
              labelspacing=0.3, bbox_to_anchor=(0.03, 0.03))
    ax.set_title("(a) ring-peak position", loc="left", fontsize=8.0)

    tag = f"jpm{SHOWCASE_JPM:+.6f}".replace(".", "p")
    temperatures = curves["temperatures"]
    bare_curve = curves[f"{tag}_bare_curve"]
    average_curve = curves[f"{tag}_average_curve"]
    spread = curves[f"{tag}_curve_spread"]
    bx.plot(temperatures, bare_curve, color=BARE, lw=1.1, label="periodic ED")
    bx.fill_between(temperatures, average_curve - 0.5 * spread,
                    average_curve + 0.5 * spread, color=AVG, alpha=0.22, lw=0,
                    label="twist envelope")
    bx.plot(temperatures, average_curve, color=AVG, lw=1.3, label="twist average")
    single_curves = OUTPUT / "twist_average_curves_m1.npz"
    if single_curves.exists():
        with np.load(single_curves) as saved:
            key = f"{tag}_average_curve"
            if key in saved:
                bx.plot(temperatures, saved[key], color=SINGLE, lw=1.2,
                        label=r"single twist $(\pi/2)^3$")
    key = f"{tag}_winding_free_curve"
    if key in curves:
        bx.plot(temperatures, curves[key], color=CLEAN, lw=1.3, label="winding-free")
    g6 = 12.0 * abs(SHOWCASE_JPM) ** 3
    bx.axvline(g6, color=G6C, lw=0.8, ls=(0, (5, 1.6)))
    bx.text(g6 * 1.15, 0.62, r"$g_6$", fontsize=7.2, color="0.35")
    g4 = 4.0 * SHOWCASE_JPM ** 2
    bx.axvline(g4, color=G4C, lw=0.8, ls=(0, (1.5, 1.5)))
    bx.text(g4 * 1.15, 0.62, r"$g_4$", fontsize=7.2, color=G4C)
    bx.set_xscale("log")
    bx.set_xlim(2e-4, 2.0)
    bx.set_ylim(0.0, 0.7)
    bx.set_xlabel(r"$T/J_{zz}$")
    bx.set_ylabel(r"$C/N k_B$")
    bx.legend(loc="upper left", handlelength=1.7, labelspacing=0.3)
    bx.set_title(f"(b) $C(T)$ at $J_\\pm/J_{{zz}}={SHOWCASE_JPM}$",
                 loc="left", fontsize=8.0)

    figure.tight_layout(pad=0.5)
    for suffix in ("pdf", "png"):
        figure.savefig(OUTPUT / f"fig_twist_average.{suffix}", bbox_inches="tight")
    plt.close(figure)
    print(f"wrote {OUTPUT / 'fig_twist_average.pdf'}")


if __name__ == "__main__":
    main()
