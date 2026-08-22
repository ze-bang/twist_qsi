"""THE figure: the dynamical structure factor near L.

(a) Exact S^zz(q, omega) of the 72-site cluster at a sequence of
    momenta approaching L, each curve the full set of
    continued-fraction poles broadened for display and normalized by
    its own S(q).  Photon-like momenta peak at 1.5--2.2K; at L -- and,
    equally, at the soft-valley momentum (1/6,1/2,1/2) -- the spectrum
    collapses onto a single dominant pole near 1K carrying over half
    the weight.  The roton is not drawn or asserted; it is simply what
    the spectrum looks like.

(b) The L-point spectrum at all four diagonalized sizes, overlaid on
    one axis.  The soft pole sits at 0.90--1.07K with about half the
    weight at every size; the upper transverse branch at 1.65--1.81K.

All poles are exact diagonalization (dense at 48; Lanczos with the
sum-rule gate reading 1.0000 at 64/72/96); the broadening
eta = 0.05K is display-only and stated in the caption.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif": ["cmr10", "Computer Modern Roman", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "axes.unicode_minus": False,
})

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "campaign" / "outputs" / "ring_model_dssf"
OUTDIR = ROOT / "paper_multiphoton" / "figs"
INK, INK2, EDGE = "#0f172a", "#475569", "#cbd5e1"
PHOTON, ROTON = "#4f46e5", "#e11d48"
ETA = 0.05


def broaden(lines, om):
    y = np.zeros_like(om)
    for e, w, *_ in lines:
        y += w * (ETA / np.pi) / ((om - e) ** 2 + ETA ** 2)
    return y


def entry(fname, star, S=None):
    d = json.load(open(D / fname))
    for m in d["momenta"]:
        if m["star"] == star and (S is None or abs(m["S"] - S) < 1e-3):
            return m
    raise KeyError(f"{star} not in {fname}")


def panel_approach(ax):
    om = np.linspace(0, 3.4, 1200)
    seq = [("?[0.333, 0.333, 0.333]", r"$(\frac{1}{3},\frac{1}{3},\frac{1}{3})$",
            PHOTON),
           ("?[0.0, 0.667, 0.667]", r"$(0,\frac{2}{3},\frac{2}{3})$", PHOTON),
           ("?[0.167, 0.167, 0.833]", r"$(\frac{1}{6},\frac{1}{6},\frac{5}{6})$",
            PHOTON),
           ("?[0.167, 0.5, 0.5]", r"$(\frac{1}{6},\frac{1}{2},\frac{1}{2})$",
            ROTON),
           ("L", "$L$", ROTON)]
    off = 0.0
    step = 2.6
    for star, lab, col in seq:
        m = entry("ring_allq_fcc233.json", star)
        y = broaden(m["lines"], om) / m["S"]
        ax.fill_between(om, off, off + y, color=col, alpha=0.16, lw=0)
        ax.plot(om, off + y, color=col, lw=1.4)
        ax.annotate(lab, xy=(3.32, off + 0.32), fontsize=9, color=col,
                    ha="right")
        off += step
    ax.axvspan(0.88, 1.12, color=ROTON, alpha=0.06, zorder=0)
    ax.set_xlim(0, 3.4)
    ax.set_ylim(-0.15, off + 1.4)
    ax.set_yticks([])
    ax.set_xlabel(r"$\omega/K$", fontsize=9.5, color=INK)
    ax.set_ylabel(r"$S^{zz}(\bf q,\omega)\;/\;S(\bf q)$   (offset)",
                  fontsize=9.3, color=INK)
    ax.set_title("(a) approaching $L$ on the 72-site cluster",
                 loc="left", fontsize=9.4, color=INK)


def panel_sizes(ax):
    om = np.linspace(0, 3.4, 1200)
    srcs = [("ring96_fcc223.json", "L", None, "48", 0.30),
            ("ring_allq_fcc224.json", "L", 0.3691, "64", 0.50),
            ("ring_allq_fcc233.json", "L", None, "72", 0.70),
            ("ring96_fcc234.json", "L", 0.3548, "96", 1.00)]
    for fn, star, S, lab, alpha in srcs:
        m = entry(fn, star, S)
        y = broaden(m["lines"], om) / m["S"]
        ax.plot(om, y, color=ROTON, lw=1.5, alpha=alpha, label=lab)
    ax.legend(loc="upper right", fontsize=8, frameon=False,
              title="spins", title_fontsize=8, handlelength=1.4)
    ax.annotate("upper transverse\nbranch", xy=(2.35, 1.7), fontsize=8.6,
                color=INK2, ha="center")
    ax.set_xlim(0, 3.4)
    ax.set_ylim(0, 4.6)
    ax.set_xlabel(r"$\omega/K$", fontsize=9.5, color=INK)
    ax.set_ylabel(r"$S^{zz}(L,\omega)\;/\;S(L)$", fontsize=9.3,
                  color=INK)
    ax.set_title("(b) the $L$ spectrum at 48, 64, 72, 96 spins",
                 loc="left", fontsize=9.4, color=INK)


def main():
    fig, ax = plt.subplots(2, 1, figsize=(3.4, 5.3))
    fig.patch.set_facecolor("white")
    panel_approach(ax[0])
    panel_sizes(ax[1])
    for a in ax:
        a.tick_params(labelsize=8.5, colors=INK2, length=3)
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            a.spines[s].set_color(EDGE)
    fig.subplots_adjust(left=0.13, right=0.975, top=0.955, bottom=0.085,
                        hspace=0.34)
    for ext in ("pdf", "png"):
        fig.savefig(OUTDIR / f"fig_dssf.{ext}", dpi=300)
    print("wrote", OUTDIR / "fig_dssf.pdf")


if __name__ == "__main__":
    main()
