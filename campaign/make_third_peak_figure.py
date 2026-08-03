#!/usr/bin/env python3
"""The third-peak taxonomy, demonstrated from the cached spectra.

Three panels, one per bump:
(a) The fake bump (Jpm = -0.20).  H+C shows a low-T bump that fold does
    not.  Surgical test, computed fresh here: collapse the six lowest H+C
    levels (the image of the exactly-degenerate masked ground sextet) to
    their mean and recompute C(T) -- the bump vanishes, proving it is
    pure multiplet splitting, i.e. the counterterm's residue.
(b) The real gap (Jpm = -0.30).  The fold curve peaks where the exact
    masked spectrum's internal gap Delta = 0.020638 predicts a two-level
    Schottky, T ~ Delta/2.4 = 0.0086.
(c) The crossing.  Lowest masked levels vs coupling: the ground sextet is
    exactly degenerate through -0.22, and at -0.25 the 12-dimensional
    sector's doublet dives below it (6 -> 2), leaving the tiny splitting
    delta = 0.001206 that radiates the ultra-low bump at delta/2.4.

Outputs: outputs/two_sided_fold/fig_third_peak.{pdf,png}
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FOLD = ROOT / "campaign" / "outputs" / "two_sided_fold"
BAND = ROOT / "campaign" / "outputs" / "masked_band_complete"
THERMO = ROOT / "campaign" / "outputs" / "wf_thermodynamics"

HC, FOLD_C, NEUTRAL = "#e69f00", "#d55e00", "0.35"
SEXTET = 6


def heat(levels: np.ndarray, grid: np.ndarray) -> np.ndarray:
    shifted = np.sort(levels) - np.min(levels)
    beta = 1.0 / grid[:, None]
    boltzmann = np.exp(-beta * shifted[None, :])
    partition = boltzmann.sum(axis=1)
    first = (boltzmann @ shifted) / partition
    second = (boltzmann @ shifted**2) / partition
    return (second - first**2) / grid**2 / 16.0


def masked_levels(tag: str) -> np.ndarray:
    record = json.loads((BAND / f"band_{tag}.json").read_text())
    values = np.asarray(
        [x for x in record["levels"] if x is not None], float)
    return np.sort(values[np.isfinite(values)])


def main() -> None:
    fig, (a, b, c) = plt.subplots(
        1, 3, figsize=(12.6, 3.9), constrained_layout=True)

    # ---- (a) the fake bump, and its surgical removal (-0.20) ----------
    data = np.load(FOLD / "fold_-0p200.npz")
    grid = data["grid"]
    window = (grid > 8e-4) & (grid < 0.06)
    hc_levels = np.sort(np.load(THERMO / "thermo_-0p200.npz")["wf_levels"])
    collapsed = hc_levels.copy()
    collapsed[:SEXTET] = collapsed[:SEXTET].mean()
    a.plot(grid[window], data["curve_hc"][window], color=HC, lw=1.8,
           label="counterterm $H+C$")
    a.plot(grid[window], heat(collapsed, grid)[window], color=HC, lw=1.6,
           ls="--", label="$H+C$, sextet collapsed")
    a.plot(grid[window], data["curve_fold"][window], color=FOLD_C, lw=1.8,
           label="two-sided fold")
    a.set_xscale("log")
    a.set_xlabel(r"$T/J_{zz}$")
    a.set_ylabel(r"$C/N$")
    a.set_title("(a) bump 1 is splitting: collapse it, it dies\n"
                r"$J_{\pm}=-0.20$", fontsize=9.5)
    a.legend(frameon=False, fontsize=7.5, loc="upper left")

    # ---- (b) the real gap (-0.30) -------------------------------------
    data30 = np.load(FOLD / "fold_-0p300.npz")
    grid30 = data30["grid"]
    win30 = (grid30 > 8e-4) & (grid30 < 0.06)
    levels30 = masked_levels("-0p300")
    gap = levels30[2] - levels30[0]
    schottky = gap / 2.4
    b.plot(grid30[win30], data30["curve_hc"][win30], color=HC, lw=1.8,
           label="counterterm $H+C$")
    b.plot(grid30[win30], data30["curve_fold"][win30], color=FOLD_C, lw=1.8,
           label="two-sided fold")
    b.axvline(schottky, color=NEUTRAL, ls=":", lw=1.4)
    b.text(schottky * 1.13, 0.056,
           rf"$\Delta_{{\rm masked}}/2.4 = {schottky:.4f}$",
           fontsize=8, color=NEUTRAL)
    b.set_xscale("log")
    b.set_xlabel(r"$T/J_{zz}$")
    b.set_title("(b) bump 2 sits where the exact gap says\n"
                r"$J_{\pm}=-0.30$, $\Delta_{\rm masked}=%.4f$" % gap,
                fontsize=9.5)
    b.legend(frameon=False, fontsize=7.5, loc="upper left")

    # ---- (c) the level crossing ---------------------------------------
    tags = ["-0p200", "-0p220", "-0p250", "-0p280", "-0p300"]
    xs = [0.20, 0.22, 0.25, 0.28, 0.30]
    for x, tag in zip(xs, tags):
        rel = masked_levels(tag)
        rel = rel[:9] - rel[0]
        distinct, counts = np.unique(np.round(rel, 6), return_counts=True)
        for value, count in zip(distinct, counts):
            c.plot([x - 0.008, x + 0.008], [value, value],
                   color=NEUTRAL, lw=2.2)
            c.annotate(rf"$\times{count}$", (x + 0.009, value),
                       fontsize=7, color=NEUTRAL, va="center")
    c.annotate("ground sextet\nexactly degenerate", (0.207, 0.0012),
               fontsize=8, color=NEUTRAL, ha="center")
    c.annotate("12-dim sector's doublet\ntakes over: $6\\to2$",
               (0.253, 0.0045), fontsize=8, color=FOLD_C, ha="center")
    c.annotate(r"$\delta=0.0012$" + "\n" + r"$\to$ bump 3 at $\delta/2.4$",
               (0.276, 0.0008), fontsize=7.5, color=FOLD_C, ha="center")
    c.annotate(r"$\Delta=0.0206\ \to$ bump 2", (0.288, 0.0225),
               fontsize=8, color=NEUTRAL, ha="center")
    c.set_xlim(0.19, 0.315)
    c.set_ylim(-0.0012, 0.026)
    c.set_xlabel(r"$|J_{\pm}|/J_{zz}$")
    c.set_ylabel(r"$E - E_0$  (masked band)")
    c.set_title("(c) bump 3 is a level crossing\nlowest exact masked levels",
                fontsize=9.5)

    for axis in (a, b, c):
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.tick_params(labelsize=8)

    fig.savefig(FOLD / "fig_third_peak.pdf")
    fig.savefig(FOLD / "fig_third_peak.png", dpi=180)
    print(f"gap(-0.30) = {gap:.6f}, Schottky prediction {schottky:.6f}")
    print(f"wrote {FOLD / 'fig_third_peak.pdf'}")


if __name__ == "__main__":
    main()
