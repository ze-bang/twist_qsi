#!/usr/bin/env python3
"""Sourced spectrum vs theta, colored by ice character.

One panel per coupling: the lowest levels of H(0,0,t) relative to the
instantaneous ground state, each point colored by the eigenstate's ice
weight <psi|P_ice|psi>.  A spectrally isolated, ice-saturated 90-state
manifold appears as a solid block of ice-colored levels separated from
spinon-colored levels by a visible gap at every t; hybridization appears as
intermediate colors, and loss of adiabatic continuation as the two colors
interpenetrating.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIGS = ROOT / "paper" / "figs"
OUTPUT = ROOT / "campaign" / "outputs"

plt.rcParams.update({
    "font.family": "serif", "mathtext.fontset": "cm",
    "font.size": 8.5, "axes.labelsize": 8.5, "legend.fontsize": 7.2,
    "xtick.labelsize": 7.6, "ytick.labelsize": 7.6,
    "axes.linewidth": 0.6, "figure.facecolor": "white",
    "savefig.facecolor": "white", "savefig.dpi": 400,
})

CMAP = mpl.colors.LinearSegmentedColormap.from_list(
    "pairs", ["#1145a8", "#ffc61a", "#d1261f"])  # ice -> mixed -> spinon
PAIR_MAX = 1.5


def main() -> None:
    couplings = [float(v) for v in sys.argv[1:]] or [-0.05, -0.18, -0.22]
    figure, axes = plt.subplots(1, len(couplings), figsize=(7.0, 2.6))
    for ax, jpm in zip(np.atleast_1d(axes), couplings):
        tag = f"{jpm:+.3f}".replace(".", "p")
        data = np.load(OUTPUT / f"theta_bands_jpm{tag}.npz")
        t, levels, pairs = data["t"], data["levels"], data["spinon_pairs"]
        energy = levels - levels[:, :1]
        top = 1.35 * float(energy[:, 110].max())
        for it in range(len(t)):
            keep = energy[it] <= top
            ax.scatter(np.full(keep.sum(), t[it] / np.pi), energy[it][keep],
                       c=np.clip(pairs[it][keep], 0.0, PAIR_MAX), cmap=CMAP,
                       vmin=0.0, vmax=PAIR_MAX,
                       s=6.0, marker="_", linewidths=1.4, rasterized=True)
        ax.set_xlim(-0.04, 1.04)
        ax.set_ylim(-0.02 * top, top)
        ax.set_xlabel(r"$\theta_z/\pi$")
        ax.set_title(rf"$J_\pm/J_{{zz}}={jpm:+.2f}$", fontsize=8.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    np.atleast_1d(axes)[0].set_ylabel(r"$E-E_0(\theta)\;[J_{zz}]$")
    mappable = mpl.cm.ScalarMappable(
        norm=mpl.colors.Normalize(0, PAIR_MAX), cmap=CMAP)
    colorbar = figure.colorbar(mappable, ax=list(np.atleast_1d(axes)),
                               fraction=0.025, pad=0.015)
    colorbar.set_label(r"spinon density", fontsize=7.6)
    colorbar.set_ticks([0.0, 0.5, 1.0, 1.5])
    colorbar.set_ticklabels(["0", "0.5", "1", r"$\geq1.5$"])
    for suffix in ("pdf", "png"):
        figure.savefig(FIGS / f"fig_theta_bands.{suffix}", bbox_inches="tight")
    print(f"wrote {FIGS / 'fig_theta_bands.pdf'}")


if __name__ == "__main__":
    main()
