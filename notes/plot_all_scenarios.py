#!/usr/bin/env python3
"""Plot one summary figure containing all recomputed finite-size scenarios."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
FIGS = HERE / "figs"
FIGS.mkdir(exist_ok=True)

C_BARE = "#e67e22"
C_CLEAN = "#27ae60"
C_HEX = "#8e44ad"
C_GREY = "#7f8c8d"
C_BLUE = "#2471a3"
C_RED = "#c0392b"

plt.rcParams.update({
    "font.size": 9.5,
    "axes.titlesize": 10.5,
    "axes.labelsize": 9.5,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "mathtext.fontset": "cm",
    "font.family": "serif",
    "axes.linewidth": 0.85,
    "savefig.bbox": "tight",
    "savefig.dpi": 220,
})


def cluster_label(case):
    c = case["cluster"]
    return "16 cubic" if c["basis"] == "cubic" else "32 FCC"


def main():
    data = json.loads((HERE / "recomputed_finite_size_artifact.json").read_text())
    cases = data["cases"]

    # Flatten peak records in a stable order: cluster, Jpm, mode.
    peak_rows = []
    for case in cases:
        label = cluster_label(case)
        for jpm in (-0.10, -0.05, 0.05):
            for mode in ("all", "delta0"):
                rec = next(
                    r for r in case["curves"]
                    if abs(r["Jpm"] - jpm) < 1e-12 and r["mode"] == mode
                )
                peak_rows.append((label, jpm, mode, rec))

    fig = plt.figure(figsize=(12.0, 8.6))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.15], height_ratios=[0.95, 1.05],
                          hspace=0.36, wspace=0.28)
    ax_geom = fig.add_subplot(gs[0, 0])
    ax_surv = fig.add_subplot(gs[0, 1])
    ax_abs = fig.add_subplot(gs[1, 0])
    ax_norm = fig.add_subplot(gs[1, 1])

    # ------------------------------------------------------------------
    # (a) Geometry census: what exists on each cluster.
    x = np.arange(len(cases))
    width = 0.22
    labels = [cluster_label(c) for c in cases]
    loop4 = [c["cluster"]["loops4"]["total"] for c in cases]
    wrap_hex = [c["cluster"]["hexagons"]["total"] - c["cluster"]["hexagons"]["contractible"] for c in cases]
    phys_hex = [c["cluster"]["hexagons"]["contractible"] for c in cases]
    b1 = ax_geom.bar(x - width, loop4, width, color=C_BARE, label="winding 4-loop")
    b2 = ax_geom.bar(x, wrap_hex, width, color=C_GREY, label="wrapping hexagon")
    b3 = ax_geom.bar(x + width, phys_hex, width, color=C_CLEAN, label="contractible hexagon")
    for bars in (b1, b2, b3):
        ax_geom.bar_label(bars, padding=2, fontsize=8)
    ax_geom.set_xticks(x)
    ax_geom.set_xticklabels(labels)
    ax_geom.set_ylabel("number of loops")
    ax_geom.set_title("(a) loop census on the two ED clusters", loc="left")
    ax_geom.legend(frameon=False, loc="upper left")

    # ------------------------------------------------------------------
    # (b) Projector survival: mean fraction kept by channel and cluster.
    channel_names = ["4-loop", "wrap hex", "phys hex"]
    xpos = np.arange(len(channel_names))
    gap = 0.08
    barw = 0.18
    offsets = [-1.5 * barw - gap, -0.5 * barw - gap / 3, 0.5 * barw + gap / 3, 1.5 * barw + gap]
    colors = [C_GREY, C_BLUE, "#9b9b9b", C_CLEAN]
    hatch = ["", "", "//", "//"]
    legend_entries = []
    for ci, case in enumerate(cases):
        surv = case["channel_survival"]
        vals_corner = [
            surv["H2"]["4_loop_wrapping"]["corner_mean"],
            surv["H3"]["hexagon_wrapping"]["corner_mean"],
            surv["H3"]["hexagon_contractible"]["corner_mean"],
        ]
        vals_delta = [
            surv["H2"]["4_loop_wrapping"]["delta0_mean"],
            surv["H3"]["hexagon_wrapping"]["delta0_mean"],
            surv["H3"]["hexagon_contractible"]["delta0_mean"],
        ]
        lab = cluster_label(case)
        bars = ax_surv.bar(xpos + offsets[2 * ci], vals_corner, barw,
                           color=colors[2 * ci], hatch=hatch[2 * ci],
                           label=f"{lab}: boundary twist")
        legend_entries.append(bars)
        bars = ax_surv.bar(xpos + offsets[2 * ci + 1], vals_delta, barw,
                           color=colors[2 * ci + 1], hatch=hatch[2 * ci + 1],
                           label=f"{lab}: $\\delta=0$")
        legend_entries.append(bars)
    ax_surv.set_xticks(xpos)
    ax_surv.set_xticklabels(channel_names)
    ax_surv.set_ylim(0, 1.16)
    ax_surv.set_ylabel("fraction of coupling kept")
    ax_surv.axhline(1.0, color="black", lw=0.6, ls=":")
    ax_surv.set_title("(b) what the averaging/projection keeps", loc="left")
    ax_surv.legend(frameon=False, ncol=2, loc="upper left", columnspacing=0.9)

    # ------------------------------------------------------------------
    # (c,d) All peak scenarios.
    categories = []
    xcat = []
    bare_abs, clean_abs, bare_norm, clean_norm = [], [], [], []
    g4_vals, ghex_vals = [], []
    for case in cases:
        lab = cluster_label(case).replace(" ", "\n")
        for jpm in (-0.10, -0.05, 0.05):
            categories.append(f"{lab}\n$J_\\pm={jpm:+.2f}$")
            rb = next(r for r in case["curves"] if abs(r["Jpm"] - jpm) < 1e-12 and r["mode"] == "all")
            rc = next(r for r in case["curves"] if abs(r["Jpm"] - jpm) < 1e-12 and r["mode"] == "delta0")
            bare_abs.append(rb["Tpk"])
            clean_abs.append(rc["Tpk"])
            bare_norm.append(rb["Tpk_over_ghex"])
            clean_norm.append(rc["Tpk_over_ghex"])
            g4_vals.append(rb["g4"])
            ghex_vals.append(rb["ghex"])
    xcat = np.arange(len(categories))
    bw = 0.34
    ax_abs.bar(xcat - bw / 2, bare_abs, bw, color=C_BARE, label="bare")
    ax_abs.bar(xcat + bw / 2, clean_abs, bw, color=C_CLEAN, label=r"clean ($\delta=0$)")
    ax_abs.scatter(xcat, g4_vals, marker="_", s=150, color="black", linewidths=1.4,
                   label=r"$g_4$")
    ax_abs.scatter(xcat, ghex_vals, marker="x", s=35, color=C_HEX,
                   label=r"$g_{\rm hex}$")
    ax_abs.set_yscale("log")
    ax_abs.set_xticks(xcat)
    ax_abs.set_xticklabels(categories, rotation=0)
    ax_abs.set_ylabel(r"$T_{\rm peak}/J_{zz}$")
    ax_abs.set_title("(c) all low-temperature peak positions", loc="left")
    ax_abs.legend(frameon=False, ncol=2, loc="upper right")

    ax_norm.bar(xcat - bw / 2, bare_norm, bw, color=C_BARE, label="bare")
    ax_norm.bar(xcat + bw / 2, clean_norm, bw, color=C_CLEAN, label=r"clean ($\delta=0$)")
    ax_norm.axhline(1.0, color=C_HEX, ls="--", lw=1.0, label=r"photon scale $g_{\rm hex}$")
    # Artifact expectation g4/ghex = 1/(3|Jpm|), category by category.
    artifact_ratio = [1.0 / (3.0 * abs(float(cat.split("=")[-1].strip("$")))) for cat in categories]
    ax_norm.scatter(xcat, artifact_ratio, marker="_", s=150, color="black", linewidths=1.4,
                    label=r"$g_4/g_{\rm hex}$")
    ax_norm.set_yscale("log")
    ax_norm.set_xticks(xcat)
    ax_norm.set_xticklabels(categories, rotation=0)
    ax_norm.set_ylabel(r"$T_{\rm peak}/g_{\rm hex}$")
    ax_norm.set_title("(d) same data in photon units", loc="left")
    ax_norm.legend(frameon=False, ncol=2, loc="upper right")

    for ax in (ax_geom, ax_surv, ax_abs, ax_norm):
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.18, lw=0.6)

    fig.suptitle("All independently recomputed finite-size scenarios", y=0.995, fontsize=13)
    for ext in ("pdf", "png"):
        out = FIGS / f"fig_all_scenarios.{ext}"
        fig.savefig(out)
        print("wrote", out)
    plt.close(fig)


if __name__ == "__main__":
    main()
