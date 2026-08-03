#!/usr/bin/env python3
"""fig_anchor_convergence.pdf: the splitting is the residual, over ten decades.

The central evidence that the counterterm's ground-multiplet splitting is a
convergence artifact rather than an intrinsic "anchoring residue": plotted
against the residual of the Brillouin-Wigner condition it is a straight line of
slope one.

Usage: make_anchor_figure.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "campaign" / "outputs" / "anchor_convergence"
FIGURES = ROOT / "paper" / "figs"
FIGURES.mkdir(parents=True, exist_ok=True)

# One hue per coupling, ordered; identity follows the coupling, not the rank.
TONES = ("#0072B2", "#D55E00", "#009E73", "#CC79A7")


def main() -> None:
    plt.rcParams.update({
        "font.size": 8.5, "axes.labelsize": 9, "legend.fontsize": 8,
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "axes.linewidth": 0.6, "savefig.dpi": 400,
        "savefig.facecolor": "white",
    })
    paths = sorted(DATA.glob("anchor_*.json"),
                   key=lambda p: -float(json.loads(p.read_text())["jpm"]))
    if not paths:
        raise SystemExit(f"no anchor_*.json in {DATA}")

    figure, ax = plt.subplots(figsize=(4.4, 3.2))
    for tone, path in zip(TONES, paths):
        record = json.loads(path.read_text())
        steps = record["trajectory"]
        residual = np.array([s["residual"] for s in steps])
        splitting = np.array([s["splitting"] for s in steps])
        keep = (residual > 0) & (splitting > 0)
        ax.plot(residual[keep], splitting[keep], marker="o", ms=3.0, lw=1.0,
                color=tone, label=rf"$J_\pm/J_{{zz}}={record['jpm']:+.2f}$")

    span = np.array([1e-16, 1e-1])
    ax.plot(span, 1.5 * span, color="#5C5C5C", lw=0.7, ls=(0, (3, 2)),
            zorder=0, label=r"$\delta=1.5\,|g|$")

    # the splitting at which the spurious anomaly would reach g6 at -0.20
    g6 = 12.0 * 0.20**3
    ax.axhline(2.4 * g6, color="#5C5C5C", lw=0.6, ls=":", zorder=0)
    ax.text(1e-15, 2.4 * g6 * 1.4,
            r"spurious anomaly would reach $g_6$", fontsize=7.2,
            color="#5C5C5C", va="bottom")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1e-16, 3e-1)
    ax.set_ylim(1e-15, 3e-1)
    ax.set_xlabel(r"anchor residual $|g|=|E_0(H_0+C)-z_0|$  $[J_{zz}]$")
    ax.set_ylabel(r"ground-multiplet splitting $\delta$  $[J_{zz}]$")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(color="#E6E6E6", lw=0.45, zorder=0)
    ax.legend(frameon=False, loc="lower right")
    figure.tight_layout()
    figure.savefig(FIGURES / "fig_anchor_convergence.pdf", bbox_inches="tight")
    print(f"wrote {FIGURES / 'fig_anchor_convergence.pdf'}")


if __name__ == "__main__":
    main()
