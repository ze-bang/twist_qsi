#!/usr/bin/env python3
"""Low spectrum vs Jpm resolved by total magnetization S^z_tot.

Note on conventions: the emergent gauge charge is the STAGGERED divergence
Q_t = eta_t sum_i S^z_i (eta = +1/-1 on up/down tetrahedra); its total
vanishes identically on the bipartite torus, so every excitation is a
gauge-neutral spinon-antispinon pair.  The sectors here are magnetization
sectors: S^z = +-1 pairs carry net spin and cannot annihilate back into
the ice manifold (protected flat floor), S^z = 0 pairs are spin singlets
that resonate with the ice manifold (descending continuum).

S^z = 0 levels in grey; S^z = +-1 and +-2 sectors overlaid in color, all
referenced to the S^z = 0 ground state.  Spectra at +-S^z are exactly
degenerate (global spin flip), so each colored level is twofold.
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
    "font.size": 8.5, "axes.labelsize": 8.5, "legend.fontsize": 7.4,
    "xtick.labelsize": 7.6, "ytick.labelsize": 7.6,
    "axes.linewidth": 0.6, "legend.frameon": False,
    "figure.facecolor": "white", "savefig.facecolor": "white",
    "savefig.dpi": 400,
})

STYLE = {0: ("0.72", 5.5, 1.0), 1: ("#d1261f", 8.0, 1.5),
         2: ("#5b0a8c", 8.0, 1.5)}


def main() -> None:
    neutral = {}
    for path in sorted(DATA.glob("levels_-*.npz")):
        blob = np.load(path)
        neutral[round(float(blob["jpm"]), 4)] = blob["levels"]

    figure, ax = plt.subplots(figsize=(4.6, 3.0))
    for jpm, levels in neutral.items():
        energy = levels - levels[0]
        keep = energy <= 2.1
        ax.scatter(np.full(keep.sum(), jpm), energy[keep], color=STYLE[0][0],
                   s=STYLE[0][1], marker="_", linewidths=STYLE[0][2],
                   rasterized=True)
    for sz in (1, 2):
        for path in sorted(DATA.glob(f"levels_sz{sz}_*.npz")):
            blob = np.load(path)
            jpm = round(float(blob["jpm"]), 4)
            if jpm not in neutral:
                continue
            energy = blob["levels"] - neutral[jpm][0]
            keep = energy <= 2.1
            color, size, width = STYLE[sz]
            ax.scatter(np.full(keep.sum(), jpm), energy[keep], color=color,
                       s=size, marker="_", linewidths=width, rasterized=True)

    handles = [
        mpl.lines.Line2D([], [], color=STYLE[0][0], lw=1.6, label=r"$S^z_{\rm tot}=0$"),
        mpl.lines.Line2D([], [], color=STYLE[1][0], lw=1.6, label=r"$S^z_{\rm tot}=\pm1$"),
        mpl.lines.Line2D([], [], color=STYLE[2][0], lw=1.6, label=r"$S^z_{\rm tot}=\pm2$"),
    ]
    ax.legend(handles=handles, loc="upper right", handlelength=1.4,
              labelspacing=0.35)
    ax.set_xlabel(r"$J_\pm/J_{zz}$")
    ax.set_ylabel(r"$E-E_0^{S^z=0}\;[J_{zz}]$")
    ax.set_xlim(-0.25, 0.0)
    ax.set_ylim(-0.05, 2.1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    figure.tight_layout(pad=0.4)
    for suffix in ("pdf", "png"):
        figure.savefig(FIGS / f"fig_jpm_levels_charge.{suffix}",
                       bbox_inches="tight")
    print(f"wrote {FIGS / 'fig_jpm_levels_charge.pdf'}")


if __name__ == "__main__":
    main()
