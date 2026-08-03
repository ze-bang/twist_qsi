#!/usr/bin/env python3
"""Low spectrum vs Jpm at theta = 0, colored by spinon density.

The flat charge-free gauge band sits at the bottom while the two-spinon
spectrum descends linearly with |Jpm|; hybridization appears as intermediate
colors where the two meet, and the merger is where any construction built on
a separated gauge manifold must end.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIGS = ROOT / "paper" / "figs"
DATA = ROOT / "campaign" / "outputs" / "jpm_levels"

plt.rcParams.update({
    "font.family": "serif", "mathtext.fontset": "cm",
    "font.size": 8.5, "axes.labelsize": 8.5,
    "xtick.labelsize": 7.6, "ytick.labelsize": 7.6,
    "axes.linewidth": 0.6, "figure.facecolor": "white",
    "savefig.facecolor": "white", "savefig.dpi": 400,
})

CMAP = mpl.colors.LinearSegmentedColormap.from_list(
    "pairs", ["#1145a8", "#ffc61a", "#d1261f"])
PAIR_MAX = 1.5


def main() -> None:
    files = sorted(DATA.glob("levels_*.npz"))
    figure, ax = plt.subplots(figsize=(4.6, 3.0))
    for path in files:
        blob = np.load(path)
        jpm = float(blob["jpm"])
        energy = blob["levels"] - blob["levels"][0]
        keep = energy <= 1.25
        ax.scatter(np.full(keep.sum(), jpm), energy[keep],
                   c=np.clip(blob["spinon_pairs"][keep], 0, PAIR_MAX),
                   cmap=CMAP, vmin=0.0, vmax=PAIR_MAX,
                   s=7.0, marker="_", linewidths=1.3, rasterized=True)
    ax.set_xlabel(r"$J_\pm/J_{zz}$")
    ax.set_ylabel(r"$E-E_0\;[J_{zz}]$")
    ax.set_xlim(-0.25, 0.0)
    ax.set_ylim(-0.03, 1.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    mappable = mpl.cm.ScalarMappable(
        norm=mpl.colors.Normalize(0, PAIR_MAX), cmap=CMAP)
    colorbar = figure.colorbar(mappable, ax=ax, fraction=0.04, pad=0.02)
    colorbar.set_label("spinon density", fontsize=7.8)
    colorbar.set_ticks([0.0, 0.5, 1.0, 1.5])
    colorbar.set_ticklabels(["0", "0.5", "1", r"$\geq1.5$"])
    figure.tight_layout(pad=0.4)
    for suffix in ("pdf", "png"):
        figure.savefig(FIGS / f"fig_jpm_levels.{suffix}", bbox_inches="tight")
    print(f"wrote {FIGS / 'fig_jpm_levels.pdf'}")


if __name__ == "__main__":
    main()
