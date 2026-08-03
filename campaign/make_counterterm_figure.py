#!/usr/bin/env python3
"""Counterterm winding-free ED: summary across the coupling ladder.

Panel (a): band ring peak in units of g6 for naive periodic ED, the
single-counterterm construction (winding four-cycles removed), the
two-family construction (plus wrapping hexagons), and the full band
protocol where it exists.
Panel (b): the tuned counterterm coefficients against their perturbative
seeds.
Panel (c): ground-multiplet hexagon flux -- the flux-law restoration test.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIGS = ROOT / "paper" / "figs"
OUT = ROOT / "campaign" / "outputs"

plt.rcParams.update({
    "font.family": "serif", "mathtext.fontset": "cm",
    "font.size": 8.5, "axes.labelsize": 8.5, "legend.fontsize": 7.0,
    "xtick.labelsize": 7.6, "ytick.labelsize": 7.6,
    "axes.linewidth": 0.6, "legend.frameon": False,
    "figure.facecolor": "white", "savefig.facecolor": "white",
    "savefig.dpi": 400,
})


def load(directory, key_peak):
    rows = []
    for path in sorted((OUT / directory).glob("counterterm_*.npz")):
        d = np.load(path)
        rows.append((float(d["jpm"]), float(d[key_peak]),
                     float(d["gs_flux"]),
                     d["coeffs"] if "coeffs" in d.files else
                     np.array([float(d["c"]), np.nan, np.nan])
                     if "c" in d.files else
                     np.array([float(d["c4"]), np.nan, float(d["c6"])])))
    rows.sort()
    return rows


def main() -> None:
    one = load("counterterm", "band_peak")
    two = load("counterterm3", "band_peak")
    frames = json.load(open(OUT / "three_frames.json"))["points"]
    frames = {round(r["jpm"], 4): r for r in frames}

    fig, (a, b, c) = plt.subplots(3, 1, figsize=(4.8, 6.2), sharex=True,
                                  height_ratios=(1.5, 1.0, 1.0))
    x1 = np.array([r[0] for r in one]); g61 = 12 * np.abs(x1) ** 3
    x2 = np.array([r[0] for r in two]); g62 = 12 * np.abs(x2) ** 3

    xf = np.array(sorted(k for k in frames if k < 0))
    naive = np.array([frames[k].get("periodic_low") or np.nan for k in xf])
    wf = np.array([frames[k].get("ice_low")
                   or frames[k].get("dressed_low") or np.nan for k in xf])
    a.plot(xf, naive / (12 * np.abs(xf) ** 3), color="#a95a45", marker="s",
           ms=2.6, lw=1.0, label="naive periodic")
    a.plot(x1, [r[1] for r in one] / g61, color="#c9a53a", marker="^",
           ms=3.0, lw=1.1, label=r"counterterm $c_4$")
    a.plot(x2, [r[1] for r in two] / g62, color="#7a3f9d", marker="D",
           ms=2.8, lw=1.1, label=r"counterterms $c_4+c_6$")
    a.plot(xf, wf / (12 * np.abs(xf) ** 3), color="#356399", marker="o",
           ms=3.0, lw=1.4, label="full band protocol")
    a.set_yscale("log")
    a.set_ylabel(r"ring peak $T_{\rm pk}/g_6$")
    a.legend(ncol=2, loc="upper left")

    coeffs2 = np.array([r[3] for r in two])
    b.plot(x2, coeffs2[:, 0] / (4 * x2**2), color="#c9a53a", marker="^",
           ms=3.0, lw=1.1, label=r"$c_4 / 4\lambda^2$")
    b.plot(x2, -coeffs2[:, 2] / (12 * np.abs(x2) ** 3), color="#7a3f9d",
           marker="D", ms=2.8, lw=1.1, label=r"$-c_6 / 12|\lambda|^3$")
    b.axhline(1.0, color="0.8", lw=0.7, ls=":")
    b.set_ylabel("coefficient / seed")
    b.legend(loc="lower left")

    c.plot(x2, [r[2] for r in two], color="#356399", marker="o", ms=3.0,
           lw=1.2, label="counterterm ED")
    c.axhline(0.0, color="0.8", lw=0.7)
    c.plot(x2, np.full_like(x2, 0.107), color="#a95a45", lw=1.0,
           ls=(0, (3, 2)), label=r"naive ED ($\approx+0.11$, sign blind)")
    c.set_ylabel(r"ground flux $\overline{\langle O_h+O_h^\dagger\rangle}$")
    c.set_xlabel(r"$J_\pm/J_{zz}$")
    c.legend(loc="lower right")

    for ax in (a, b, c):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.tight_layout(pad=0.4)
    for suffix in ("pdf", "png"):
        fig.savefig(FIGS / f"fig_counterterm.{suffix}", bbox_inches="tight")
    print(f"wrote {FIGS / 'fig_counterterm.pdf'}")


if __name__ == "__main__":
    main()
