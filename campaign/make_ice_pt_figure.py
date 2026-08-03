#!/usr/bin/env python3
"""Fourth-order perturbation theory against the exact winding-free peak.

Draws the lower-peak position in units of g_6 across both flux branches:
the exact winding-free construction (cubic-16), the fourth-order series
with the cluster's own coefficients (E4 hexagon -68, its apples-to-apples
partner), and the fourth-order series with the thermodynamic-limit
coefficients from the covering-lattice enumeration (E4 hexagon -120,
8-ring -40, bulk diagonal).  Orders <= 3 are cluster-independent, so the
two series share the same g_6 limit and differ only through the fourth
order.  The shaded band marks where the cluster's high-order Bloch series
stops converging (its complex-plane radius); the exact construction is
unaffected there.
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
FIGS = ROOT / "paper" / "figs"
OUTPUT = ROOT / "campaign" / "outputs"

CLEAN = "#356399"
CLUSTER4 = "#c49a6a"
BULK4 = "#703630"

plt.rcParams.update({
    "font.family": "serif", "mathtext.fontset": "cm",
    "font.size": 8.5, "axes.labelsize": 8.5, "legend.fontsize": 7.0,
    "xtick.labelsize": 7.6, "ytick.labelsize": 7.6,
    "axes.linewidth": 0.6, "legend.frameon": False,
    "figure.facecolor": "white", "savefig.facecolor": "white", "savefig.dpi": 400,
})


def series_peak(e3, e4, coupling):
    g6 = 12.0 * abs(coupling) ** 3
    levels = np.linalg.eigvalsh(coupling**3 * e3 + coupling**4 * e4)
    shifted = levels - levels.min()
    grid = np.geomspace(1e-2, 1e2, 900) * g6
    beta = 1.0 / grid[:, None]
    weights = np.exp(-beta * shifted[None, :])
    z = weights.sum(axis=1)
    mean = (weights * shifted[None, :]).sum(axis=1) / z
    second = (weights * shifted[None, :] ** 2).sum(axis=1) / z
    heat = (second - mean**2) / grid**2
    return grid[int(np.argmax(heat))] / g6


def main() -> None:
    data = json.loads((OUTPUT / "ice_pt_convergence.json").read_text())
    coeff = np.load(OUTPUT / "ice_pt_coefficients.npz")
    e3, e4_cluster = coeff["E3"], coeff["E4"]
    e4_bulk = np.load(OUTPUT / "bulk_pt_e4.npz")["E4_bulk"]

    points = data["points"]
    x = np.array([p["jpm"] for p in points])
    exact = np.array([p["exact_peak_over_g6"] for p in points])
    order = np.argsort(x)
    x, exact = x[order], exact[order]

    couplings = np.linspace(-0.185, 0.055, 121)
    couplings = couplings[np.abs(couplings) > 4e-3]
    cluster_curve = [series_peak(e3, e4_cluster, v) for v in couplings]
    bulk_curve = [series_peak(e3, e4_bulk, v) for v in couplings]

    figure, ax = plt.subplots(figsize=(3.4, 2.5))
    ax.axvspan(-0.185, -0.095, color="0.93", lw=0, zorder=0)
    ax.text(-0.14, 0.06, "cluster series\ndiverges", fontsize=6.6,
            color="0.45", ha="center", va="bottom")
    ax.axhline(1.0, color="0.85", lw=0.6, zorder=0)

    negative = couplings < 0
    for mask in (negative, ~negative):
        ax.plot(couplings[mask], np.array(cluster_curve)[mask], color=CLUSTER4,
                lw=1.1, ls=(0, (4, 1.6)),
                label="4th order, cubic-16 coefficients" if mask is negative else None)
        ax.plot(couplings[mask], np.array(bulk_curve)[mask], color=BULK4,
                lw=1.1,
                label="4th order, thermodynamic limit" if mask is negative else None)
    for sign in (-1, 1):
        m = np.sign(x) == sign
        ax.plot(x[m], exact[m], color=CLEAN, lw=1.6, marker="o", ms=2.8,
                zorder=5, label="exact winding-free (cubic-16)" if sign < 0 else None)

    ax.set_xlim(-0.19, 0.06)
    ax.set_ylim(0.0, 1.85)
    ax.set_xlabel(r"$J_\pm/J_{zz}$")
    ax.set_ylabel(r"$T_{\mathrm{pk}}\,/\,g_6$")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper left", handlelength=1.8, labelspacing=0.35,
              borderaxespad=0.25)

    figure.tight_layout(pad=0.4)
    for suffix in ("pdf", "png"):
        figure.savefig(FIGS / f"fig_ice_pt.{suffix}", bbox_inches="tight")
    print(f"wrote {FIGS / 'fig_ice_pt.pdf'}")


if __name__ == "__main__":
    main()
