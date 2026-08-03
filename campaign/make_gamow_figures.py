#!/usr/bin/env python3
"""Step 4: thermodynamics of the winding-free band of poles.

Reads campaign/outputs/gamow/gamow_*.npz and produces:

  fig_gamow_widths  -- mean and maximal width of the winding-free poles vs
                       coupling, with the golden-rule eta-band as a
                       robustness envelope;
  fig_gamow_peak    -- ring-peak position and height from the sharp band
                       (Gamma = 0) against the broadened band (each pole
                       replaced by a Gaussian of sigma = Gamma/2, sampled
                       by 21-node Gauss-Hermite quadrature): the death of
                       the ring peak as a lifetime effect.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "campaign" / "outputs" / "gamow"
FIGS = ROOT / "paper" / "figs"

plt.rcParams.update({
    "font.family": "serif", "mathtext.fontset": "cm",
    "font.size": 8.5, "axes.labelsize": 8.5, "legend.fontsize": 7.2,
    "xtick.labelsize": 7.6, "ytick.labelsize": 7.6,
    "axes.linewidth": 0.6, "legend.frameon": False,
    "figure.facecolor": "white", "savefig.facecolor": "white",
    "savefig.dpi": 400,
})

NODES, WEIGHTS = np.polynomial.hermite.hermgauss(21)


def heat_curve(levels, weights, grid):
    shifted = levels - levels.min()
    beta = 1.0 / grid[:, None]
    boltz = weights[None, :] * np.exp(-beta * shifted[None, :])
    z = boltz.sum(axis=1)
    mean = (boltz * shifted[None, :]).sum(axis=1) / z
    second = (boltz * shifted[None, :] ** 2).sum(axis=1) / z
    return (second - mean * mean) / grid**2


def broadened_levels(poles):
    energies, weights = [], []
    for pole in poles:
        sigma = max(-pole.imag, 0.0)      # sigma = Gamma / 2
        if sigma < 1e-12:
            energies.append(pole.real)
            weights.append(1.0)
            continue
        energies.extend(pole.real + np.sqrt(2.0) * sigma * NODES)
        weights.extend(WEIGHTS / np.sqrt(np.pi))
    return np.asarray(energies), np.asarray(weights)


def main() -> None:
    rows = []
    for path in sorted(DATA.glob("gamow_*.npz")):
        d = np.load(path)
        poles = np.asarray(d["wf_poles"])
        poles = poles[np.isfinite(poles)]
        if len(poles) == 0:
            continue
        jpm = float(d["jpm"])
        g6 = 12.0 * abs(jpm) ** 3
        grid = np.geomspace(1e-7, 0.3, 2600)
        sharp = heat_curve(poles.real, np.ones(len(poles)), grid)
        elevels, eweights = broadened_levels(poles)
        broad = heat_curve(elevels, eweights, grid)
        window = grid < 0.1
        rows.append({
            "jpm": jpm, "g6": g6,
            "gamma_mean": float(-2.0 * np.mean(poles.imag)),
            "gamma_max": float(-2.0 * np.min(poles.imag)),
            "g1_lo": float(np.nanmean(d["gamma1_eta2"])),
            "g1_mid": float(np.nanmean(d["gamma1_eta4"])),
            "g1_hi": float(np.nanmean(d["gamma1_eta8"])),
            "sharp_peak": float(grid[window][np.argmax(sharp[window])]),
            "broad_peak": float(grid[window][np.argmax(broad[window])]),
            "height_ratio": float(np.max(broad[window])
                                  / np.max(sharp[window])),
        })
    rows.sort(key=lambda r: r["jpm"])
    x = np.array([r["jpm"] for r in rows])
    g6 = np.array([r["g6"] for r in rows])

    fig, ax = plt.subplots(figsize=(4.4, 2.7))
    ax.fill_between(x, [r["g1_lo"] for r in rows], [r["g1_hi"] for r in rows],
                    color="0.85", label=r"golden rule ($\eta$ band)")
    ax.plot(x, [r["g1_mid"] for r in rows], color="0.35", lw=1.0)
    ax.plot(x, [r["gamma_mean"] for r in rows], color="#d1261f", lw=1.4,
            marker="o", ms=3, label=r"$\bar\Gamma$ (winding-free poles)")
    ax.plot(x, [r["gamma_max"] for r in rows], color="#d1261f", lw=0.9,
            ls=(0, (3, 1.5)), marker="s", ms=2.4, label=r"$\Gamma_{\max}$")
    ax.plot(x, g6, color="#5182bd", lw=0.9, ls=(0, (1.2, 1.8)),
            label=r"$g_6$")
    ax.set_yscale("log")
    ax.set_xlabel(r"$J_\pm/J_{zz}$")
    ax.set_ylabel(r"width $[J_{zz}]$")
    ax.legend(loc="upper left", fontsize=6.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout(pad=0.4)
    for suffix in ("pdf", "png"):
        fig.savefig(FIGS / f"fig_gamow_widths.{suffix}", bbox_inches="tight")

    fig2, (a1, a2) = plt.subplots(2, 1, figsize=(4.4, 3.6), sharex=True,
                                  height_ratios=(1.4, 1.0))
    a1.plot(x, np.array([r["sharp_peak"] for r in rows]) / g6, color="#356399",
            lw=1.3, marker="o", ms=3, label=r"sharp band ($\Gamma=0$)")
    a1.plot(x, np.array([r["broad_peak"] for r in rows]) / g6, color="#d1261f",
            lw=1.3, marker="^", ms=3, label="broadened band")
    a1.set_yscale("log")
    a1.set_ylabel(r"$T_{\rm pk}/g_6$")
    a1.legend(fontsize=7.0)
    a2.plot(x, [r["height_ratio"] for r in rows], color="0.25", lw=1.3,
            marker="o", ms=3)
    a2.axhline(1.0, color="0.85", lw=0.7)
    a2.set_ylabel(r"$C_{\rm pk}^{\rm broad}/C_{\rm pk}^{\rm sharp}$")
    a2.set_xlabel(r"$J_\pm/J_{zz}$")
    for a in (a1, a2):
        a.spines["top"].set_visible(False)
        a.spines["right"].set_visible(False)
    fig2.tight_layout(pad=0.4)
    for suffix in ("pdf", "png"):
        fig2.savefig(FIGS / f"fig_gamow_peak.{suffix}", bbox_inches="tight")

    for r in rows:
        print({k: (round(v, 6) if isinstance(v, float) else v)
               for k, v in r.items()})
    print("wrote fig_gamow_widths + fig_gamow_peak")


if __name__ == "__main__":
    main()
