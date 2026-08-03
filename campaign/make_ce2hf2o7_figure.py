#!/usr/bin/env python3
"""Ce2Hf2O7 winding-free segment figure.

Two panels, all from cached cubic-16 XYZ points and the digitized Fig. 2(a) of
Smith et al.:

  (a) the measured two-peak heat capacity (points) with the two anomalies
      marked, overlaid with three cubic-16 curves at J_a = 0.05 meV: periodic
      ED and winding-free ED at the published NLC-A parameters, and winding-free
      ED at the g_6-pinned fine-tune (J_pm/J_zz = -0.1515).  The periodic lower
      peak is the winding four-loop artifact; the winding-free ring-exchange
      peak lies far below the measured T_2 and only edges up under the tune.

  (b) why: the winding-free ring-exchange peak against the third-order estimate
      g_6 = 12|J_pm|^3/J_zz^2 along the XXZ line.  They agree at weak coupling
      and diverge toward NLC-A strength -- at the pinned point the true peak is
      a third of g_6 -- so pinning the formula to T_2 does not put the exact
      peak there on 16 sites.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "notes"))

import recompute_finite_size_artifact as geometry  # noqa: E402
from qsi_campaign.material_fit import load_digitized_series  # noqa: E402
from qsi_campaign.material_scan import (  # noqa: E402
    KB_MEV_PER_K,
    heat_capacity_curves,
    local_maxima,
    solve_point,
)

CACHE = ROOT / "campaign" / "cache" / "ce2hf2o7_points"
FIGS = ROOT / "paper" / "figs"
FIGS.mkdir(parents=True, exist_ok=True)

JA_MEV = 0.050            # NLC-A high-temperature scale
NLC_A = (-0.125, 0.085)   # (J_pm, J_pmpm)/J_zz
TUNED = (-0.1515, 0.085)  # g_6 pinned to the measured T_2

BARE = "#a95a45"     # periodic ED, muted iron red
CLEAN = "#356399"    # winding-free ED at NLC-A, muted cobalt
TUNE = "#2e7d74"     # winding-free ED, g_6-pinned tune, muted teal
DATA = "#16181d"
EDGE = "#c9a53a"
INK = "#16181d"

plt.rcParams.update({
    "font.family": "serif", "mathtext.fontset": "cm",
    "font.size": 8.5, "axes.labelsize": 8.5, "axes.titlesize": 8.5,
    "legend.fontsize": 6.9, "xtick.labelsize": 7.6, "ytick.labelsize": 7.6,
    "axes.linewidth": 0.6, "legend.frameon": False,
    "figure.facecolor": "white", "savefig.facecolor": "white", "savefig.dpi": 400,
})


def measured(smoothing: int = 5):
    temperature, heat = load_digitized_series(
        ROOT / "campaign" / "data" / "ce2hf2o7_smith2025_digitized.csv"
    )
    kernel = np.ones(smoothing) / smoothing
    smoothed = np.convolve(heat, kernel, "same")
    peaks = [p for p in local_maxima(temperature[smoothing:-smoothing], smoothed[smoothing:-smoothing]) if p[0] < 0.2]
    peaks.sort(key=lambda item: -item[1])
    low, high = sorted(p[0] for p in peaks[:2])
    return temperature, heat, low, high


def main() -> None:
    cluster = geometry.build_cluster("cubic", (1, 1, 1))
    temperature, heat, t2, t1 = measured()
    grid_K = np.geomspace(1.0e-4, 10.0, 2000)

    periodic_nlc = heat_capacity_curves(
        solve_point(cluster, *NLC_A, cache_dir=CACHE), JA_MEV, grid_K)["periodic"]
    windfree_nlc = heat_capacity_curves(
        solve_point(cluster, *NLC_A, cache_dir=CACHE), JA_MEV, grid_K)["winding_free"]
    windfree_tuned = heat_capacity_curves(
        solve_point(cluster, *TUNED, cache_dir=CACHE), JA_MEV, grid_K)["winding_free"]

    figure, axes = plt.subplots(1, 2, figsize=(7.0, 2.55))

    # (a) data + three cubic-16 curves --------------------------------------
    ax = axes[0]
    ax.plot(temperature, heat, lw=0, marker="o", ms=1.7, mfc=DATA, mec="none",
            label=r"Ce$_2$Hf$_2$O$_7$ (Smith $et\ al.$)", zorder=5)
    ax.plot(grid_K, periodic_nlc, color=BARE, lw=1.3, ls=(0, (5, 1.6)),
            label="periodic ED, NLC-A")
    ax.plot(grid_K, windfree_nlc, color=CLEAN, lw=1.4, label="winding-free, NLC-A")
    ax.plot(grid_K, windfree_tuned, color=TUNE, lw=1.4,
            label=r"winding-free, tuned $J_\pm/J_{zz}{=}{-}0.151$")
    for temperature_peak, label in ((t2, r"$T_2$"), (t1, r"$T_1$")):
        ax.axvline(temperature_peak, color=EDGE, lw=0.7, ls=(0, (4, 2)), zorder=0)
        ax.text(temperature_peak, 2.34, label, ha="center", va="bottom", fontsize=8, color=INK)
    periodic_low = min(local_maxima(grid_K, periodic_nlc), key=lambda p: p[0])
    ax.annotate("winding\nartifact", xy=periodic_low, xytext=(0.052, 0.62),
                fontsize=6.6, color=BARE, ha="center", va="top",
                arrowprops=dict(arrowstyle="-", color=BARE, lw=0.6))
    ax.set_xscale("log")
    ax.set_xlim(8e-4, 8)
    ax.set_ylim(0, 2.5)
    ax.set_xlabel(r"$T$ (K)")
    ax.set_ylabel(r"$C_{\mathrm{mag}}$ (J K$^{-1}$ mol$_{\mathrm{Ce}}^{-1}$)")
    ax.set_title(r"(a) heat capacity, $J_a=0.05$ meV", fontsize=8)
    ax.legend(loc="upper right", handlelength=1.7, borderaxespad=0.3, labelspacing=0.3)

    # (b) exact ring peak vs the third-order estimate -----------------------
    ax = axes[1]
    scan = json.loads((ROOT / "campaign" / "outputs" / "ce2hf2o7_peak_scan.json").read_text())
    line = []
    for point in scan["points"]:
        if not point.get("well_defined") or "T_low_K" not in point["winding_free"]:
            continue
        if abs(point["jpmpm"]) > 1e-9:
            continue
        x = abs(point["jpm"])
        line.append((x, 12.0 * x**3 * JA_MEV / KB_MEV_PER_K * 1e3,
                     point["winding_free"]["T_low_K"] * 1e3))
    line.sort()
    xs = np.array([row[0] for row in line])
    g6 = np.array([row[1] for row in line])
    ed = np.array([row[2] for row in line])
    ax.plot(xs, g6, color="0.45", lw=1.2, ls=(0, (5, 1.6)),
            label=r"$g_6=12|J_\pm|^3/J_{zz}^2$")
    ax.plot(xs, ed, color=CLEAN, lw=1.4, marker="o", ms=3,
            label="winding-free ED peak")
    ax.axhline(t2 * 1e3, color=DATA, lw=1.0, ls="--")
    ax.text(0.052, t2 * 1e3 + 1.2, r"measured $T_2$", fontsize=6.9, color=DATA)
    for x0, name, off in ((0.125, "NLC-A", (-0.004, 7, "right")),
                          (0.1515, "tuned", (0.004, 7, "left"))):
        peak = float(np.interp(x0, xs, ed))
        ax.scatter([x0], [peak], s=20, color=TUNE if name == "tuned" else BARE,
                   zorder=6, edgecolors="white", linewidths=0.5)
        ax.annotate(name, xy=(x0, peak), xytext=(x0 + off[0], peak + off[1]),
                    fontsize=6.6, ha=off[2], color=TUNE if name == "tuned" else BARE)
    ax.set_xlim(0.045, 0.185)
    ax.set_ylim(0, 44)
    ax.set_xlabel(r"$|J_\pm|/J_{zz}$   (XXZ line)")
    ax.set_ylabel(r"lower-peak temperature (mK)")
    ax.set_title("(b) exact ring peak vs third order", fontsize=8)
    ax.legend(loc="upper left", handlelength=1.7, borderaxespad=0.3)

    figure.tight_layout(pad=0.5)
    for suffix in ("pdf", "png"):
        figure.savefig(FIGS / f"fig_ce2hf2o7.{suffix}", bbox_inches="tight")
    plt.close(figure)

    windfree_low_nlc = min(local_maxima(grid_K, windfree_nlc), key=lambda p: p[0])
    windfree_low_tuned = min(local_maxima(grid_K, windfree_tuned), key=lambda p: p[0])
    print(f"measured T2={t2*1e3:.1f} mK  T1={t1*1e3:.1f} mK")
    print(f"periodic NLC-A low peak {periodic_low[0]*1e3:.2f} mK")
    print(f"winding-free NLC-A low peak {windfree_low_nlc[0]*1e3:.2f} mK")
    print(f"winding-free tuned low peak {windfree_low_tuned[0]*1e3:.2f} mK "
          f"(g6 formula {12*0.1515**3*JA_MEV/KB_MEV_PER_K*1e3:.1f} mK)")
    print(f"wrote {FIGS/'fig_ce2hf2o7.pdf'}")


if __name__ == "__main__":
    main()
