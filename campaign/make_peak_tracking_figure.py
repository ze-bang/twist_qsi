#!/usr/bin/env python3
"""Peak tracking, one panel per method: every anomaly followed across coupling.

Each panel shows all maxima of C(T) found for that method against |J_pm|, with
the two candidate scales g_4 = 4 J_pm^2 and g_6 = 12 |J_pm|^3 drawn in.  Filled
symbols joined by lines are the pi-flux side (J_pm < 0); open symbols are the
zero-flux side (J_pm > 0).  Marker shape is the label:

    circle    ring anomaly (gauge anomaly for the raw periodic cluster)
    triangle  transport-sector anomaly
    square    spinon anomaly
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "campaign" / "outputs" / "method_comparison"
FIGS = ROOT / "paper" / "figs"

BLUE, VERM, GREEN, ORANGE, PINK = ("#0072B2", "#D55E00", "#009E73",
                                   "#E69F00", "#CC79A7")
GREY = "#5c5c5c"
METHODS = ("periodic", "polar", "feshbach", "hc")
TITLE = {"periodic": "(a) periodic ED (baseline)",
         "polar": "(b) polar pullback (original method)",
         "feshbach": "(c) masked Feshbach", "hc": r"(d) counterterm $H+C$"}
COLOUR = {"ring": BLUE, "gauge": BLUE, "sector": PINK, "spinon": VERM}
MARKER = {"ring": "o", "gauge": "o", "sector": "^", "spinon": "s"}
SHOWN = ("ring", "gauge", "sector", "spinon")

plt.rcParams.update({
    "font.size": 8.5, "axes.labelsize": 8.5, "axes.titlesize": 9,
    "legend.fontsize": 7.0, "xtick.labelsize": 7.6, "ytick.labelsize": 7.6,
    "axes.linewidth": 0.7, "xtick.direction": "in", "ytick.direction": "in",
    "lines.linewidth": 1.4, "lines.markersize": 4.6,
    "axes.grid": True, "grid.alpha": 0.22, "grid.linewidth": 0.5,
    "figure.dpi": 160, "savefig.bbox": "tight",
})


def main():
    with open(OUT / "peak_labels.json") as handle:
        records = json.load(handle)
    records.sort(key=lambda r: r["jpm"])

    figure, axes = plt.subplots(1, 4, figsize=(13.8, 3.3), sharey=True)
    grid = np.geomspace(0.008, 0.33, 200)

    for axis, method in zip(axes, METHODS):
        axis.loglog(grid, 4 * grid**2, ":", color=GREY, lw=1.3,
                    label=r"$g_4=4J_\pm^2$", zorder=1)
        axis.loglog(grid, 12 * grid**3, "--", color=GREY, lw=1.3,
                    label=r"$g_6=12|J_\pm|^3$", zorder=1)
        for label in SHOWN:
            for negative, fill, style in ((True, True, "-"),
                                          (False, False, "")):
                x, y = [], []
                for record in records:
                    if (record["jpm"] < 0) != negative:
                        continue
                    for peak in record.get(method, []):
                        if peak["label"] != label or peak["at_grid_edge"]:
                            continue
                        x.append(abs(record["jpm"]))
                        y.append(peak["T"])
                if not x:
                    continue
                order = np.argsort(x)
                x = np.asarray(x)[order]
                y = np.asarray(y)[order]
                axis.loglog(x, y, MARKER[label] + style, color=COLOUR[label],
                            mfc=COLOUR[label] if fill else "none",
                            label=(label if negative else None), zorder=3)
        axis.set_title(TITLE[method], loc="left", pad=4)
        axis.set_xlabel(r"$|J_\pm|/J_{zz}$")
        axis.set_xlim(0.008, 0.35)
        axis.set_ylim(1e-6, 2.0)
    axes[0].set_ylabel(r"$T_{\rm peak}/J_{zz}$")
    axes[0].legend(frameon=False, loc="upper left", ncol=1)
    figure.suptitle("filled: $J_\\pm<0$ ($\\pi$ flux)   open: $J_\\pm>0$ "
                    "(zero flux)", y=1.03, fontsize=8.5)
    figure.tight_layout()
    figure.savefig(FIGS / "fig_peak_tracking.pdf")
    plt.close(figure)
    print("figure written to", FIGS / "fig_peak_tracking.pdf")

    for method in METHODS:
        counts = {}
        for record in records:
            for peak in record.get(method, []):
                counts[peak["label"]] = counts.get(peak["label"], 0) + 1
        print(f"  {method:9s} peaks tracked: {counts}")


if __name__ == "__main__":
    main()
