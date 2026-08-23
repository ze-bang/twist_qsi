"""fig_mechanism: the flippability census as a microscopic fingerprint.

One panel, no fitted scale anywhere: the Hellmann-Feynman derivative
d(omega)/dV of the dominant levels of every L-type momentum class on
every diagonalized cluster.  Red: the reconstructed soft level.
Indigo: the upper transverse level of the same class.  Open gray: the
dominant levels of the NON-reconstructing L classes (64- and 96-site
anisotropic boxes) -- the within-cluster control at identical |q|.
The reconstructing levels sit at positive census on every cluster;
the controls sit at negative census.  Values from the exact
diagonalizations (48: dense; 64/72/96: Lanczos Ritz vectors, gated
against dense at 48 to four decimals).

Data provenance: rk_slopes/exact_census (48), ring_allq_fcc224 (64),
ring_allq_fcc233 (72), ring96_fcc234 (96).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif": ["cmr10", "Computer Modern Roman", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "axes.unicode_minus": False,
})

ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "paper_multiphoton" / "figs"
INK, INK2, EDGE = "#0f172a", "#475569", "#cbd5e1"
PHOTON, ROTON = "#4f46e5", "#e11d48"

# (x-label, [(census, share, kind)]) kind: r soft, p upper, c control
GROUPS = [
    ("48\n$L$", [(+0.339, 0.508, "r"), (-0.064, 0.323, "p")]),
    ("64\n$L$", [(+0.338, 0.500, "r"), (-0.062, 0.313, "p")]),
    ("64\n$L'$", [(-0.166, 0.771, "c")]),
    ("72\n$L$", [(+0.134, 0.540, "r"), (+0.137, 0.178, "p")]),
    ("72\n$(\\frac{1}{6},\\frac{1}{2},\\frac{1}{2})$",
     [(+0.152, 0.550, "r"), (+0.166, 0.171, "p")]),
    ("96\n$L$", [(+0.259, 0.544, "r"), (+0.135, 0.225, "p")]),
    ("96\n$L'$", [(-0.015, 0.387, "c"), (-0.088, 0.301, "c")]),
]


def main():
    fig, ax = plt.subplots(figsize=(3.4, 2.7))
    fig.patch.set_facecolor("white")
    for x, (lab, levels) in enumerate(GROUPS):
        for cen, sh, kind in levels:
            if kind == "c":
                ax.scatter(x, cen, s=430 * sh, facecolor="white",
                           edgecolor=INK2, linewidth=1.1, zorder=3)
            else:
                col = ROTON if kind == "r" else PHOTON
                ax.scatter(x, cen, s=430 * sh, color=col,
                           edgecolor="white", linewidth=0.6, zorder=4)
    ax.axhline(0.0, color=EDGE, lw=0.9, zorder=1)
    ax.scatter([], [], s=90, color=ROTON, label="lower branch (reconstructing class)")
    ax.scatter([], [], s=90, color=PHOTON, label="upper branch")
    ax.scatter([], [], s=90, facecolor="white", edgecolor=INK2,
               label="non-reconstructing class")
    ax.legend(loc="upper right", fontsize=7.2, frameon=False,
              handletextpad=0.2, borderaxespad=0.2)
    ax.set_xticks(range(len(GROUPS)))
    ax.set_xticklabels([g[0] for g in GROUPS], fontsize=7.6)
    ax.set_ylabel(r"$d\omega_n/dV$   (flippability census)",
                  fontsize=9.2, color=INK)
    ax.set_ylim(-0.30, 0.47)
    ax.set_xlim(-0.6, len(GROUPS) - 0.4)
    ax.tick_params(labelsize=8, colors=INK2, length=3)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(EDGE)
    fig.subplots_adjust(left=0.16, right=0.975, top=0.97, bottom=0.16)
    for ext in ("pdf", "png"):
        fig.savefig(OUTDIR / f"fig_mechanism.{ext}", dpi=300)
    print("wrote", OUTDIR / "fig_mechanism.pdf")


if __name__ == "__main__":
    main()
