"""What the ring exchange actually costs, and where that puts the photon.

(a) The dressed denominator D_eff is INTENSIVE: 16 and 32 sites agree.
(b) D_eff pulls away from the spinon gap it is often assumed to track.
(c) In mK for Ce2Hf2O7: the photon has never been inside the measured window.

Sources: outputs/map_normalization/mu_denominator.json (D_eff, both clusters)
         outputs/fcc32_thermodynamics/thermo_*.json (continuum edge, band bottom)
"""

from __future__ import annotations

import glob
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "campaign" / "outputs" / "map_normalization"
THERMO = ROOT / "campaign" / "outputs" / "fcc32_thermodynamics"

# dataviz slots 1-3 (these three validate on the all-pairs list, both modes)
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#1a1a19", "#55554e", "#dedcd5"
JZZ_MK = 0.050 * 11604.5 / 1000.0 * 1000.0 / 1000.0   # 0.050 meV in kelvin
JZZ_MK = 0.050 * 11.6045 * 1000.0                     # -> mK


def style(ax, xlabel, ylabel, title):
    ax.set_xlabel(xlabel, fontsize=9.5, color=INK)
    ax.set_ylabel(ylabel, fontsize=9.5, color=INK)
    ax.set_title(title, fontsize=10.5, color=INK, loc="left", pad=8)
    ax.grid(True, which="major", lw=0.6, color=GRID, alpha=0.9)
    ax.grid(True, which="minor", lw=0.4, color=GRID, alpha=0.45)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(labelsize=8.5, colors=INK2, length=3)


def load_denominator():
    recs = json.load(open(OUT / "mu_denominator.json"))
    out = {}
    for r in recs:
        out.setdefault((r["cluster"], r["levels"]), []).append(
            (abs(r["jpm"]), r["D_eff"]))
    return {k: np.array(sorted(v)) for k, v in out.items()}


def load_gap():
    rows = []
    for path in glob.glob(str(THERMO / "thermo_*.json")):
        d = json.load(open(path))
        if d["jpm"] > 0:
            continue
        rows.append((abs(d["jpm"]),
                     d["continuum_edge"] - d["masked_band_bottom"]))
    return np.array(sorted(rows))


def main():
    dd = load_denominator()
    gap = load_gap()
    cub = dd[("cubic16", 4)]
    f1, f2 = dd[("fcc32", 1)], dd[("fcc32", 2)]

    fig, axes = plt.subplots(1, 3, figsize=(15.4, 4.9))
    fig.patch.set_facecolor("#fcfcfb")
    for ax in axes:
        ax.set_facecolor("#fcfcfb")

    # ------------------------------------------------------------- (a)
    ax = axes[0]
    ax.plot(cub[:, 0], cub[:, 1], "-o", lw=2.2, ms=7, color=C1,
            mec="#fcfcfb", mew=1.2, label="cubic-16 (exact fold)")
    ax.plot(f1[:, 0], f1[:, 1], "--s", lw=2.0, ms=6.5, color=C2,
            mec="#fcfcfb", mew=1.2, label="FCC-32, virtual depth 1")
    ax.plot(f2[:, 0], f2[:, 1], "-^", lw=2.2, ms=7.5, color=C3,
            mec="#fcfcfb", mew=1.2, label="FCC-32, virtual depth 2")
    ax.axhline(1.0, lw=1.2, color=INK2, alpha=0.6)
    ax.text(0.032, 1.06, "perturbation theory assumes this",
            fontsize=8.2, color=INK2)
    style(ax, r"$|J_\pm| / J_{zz}$",
          r"mediating energy  $D_{\rm eff}\ /\ J_{zz}$",
          "(a)  Doubling the crystal barely moves it")
    ax.legend(frameon=False, fontsize=8.6, loc="upper left",
              labelcolor=INK, bbox_to_anchor=(0.02, 0.98))

    # ------------------------------------------------------------- (b)
    ax = axes[1]
    ax.plot(f1[:, 0], f1[:, 1], "-o", lw=2.2, ms=7, color=C2,
            mec="#fcfcfb", mew=1.2)
    ax.plot(gap[:, 0], gap[:, 1], "-o", lw=2.2, ms=7, color=C1,
            mec="#fcfcfb", mew=1.2)
    ax.axhline(1.0, lw=1.2, color=INK2, alpha=0.6)
    ax.annotate("$D_{\\rm eff}$: weighted mean over ALL\nintermediate states",
                (f1[3, 0], f1[3, 1]), textcoords="offset points",
                xytext=(10, -30), ha="left", fontsize=8.6, color=INK)
    ax.annotate("lowest intermediate state\n(continuum edge above the band)",
                (gap[-1, 0], gap[-1, 1]), textcoords="offset points",
                xytext=(-8, -22), ha="right", fontsize=8.6, color=INK)
    ax.set_xlim(0, 0.52)
    ax.set_ylim(0.35, 4.35)
    style(ax, r"$|J_\pm| / J_{zz}$", r"energy $/\ J_{zz}$",
          "(b)  Both scales grow; the denominator outruns the gap")
    ax.text(0.02, 0.02,
            "a weighted mean necessarily exceeds the minimum;\n"
            "the content is that the RATIO grows, 1.29 -> 2.90",
            transform=ax.transAxes, fontsize=8.2, color=INK2, va="bottom")

    # ------------------------------------------------------------- (c)
    ax = axes[2]
    x = cub[:, 0]
    g6 = 12.0 * x ** 3 * JZZ_MK
    k6 = g6 / cub[:, 1] ** 2
    ax.plot(x, g6, "--s", lw=2.0, ms=6.5, color=C2, mec="#fcfcfb", mew=1.2,
            label="perturbation theory  $g_6$")
    ax.plot(x, k6, "-o", lw=2.2, ms=7, color=C1, mec="#fcfcfb", mew=1.2,
            label="measured  $K_6=g_6/D_{\\rm eff}^2$")
    ax.axhspan(0, 20.6, color=INK2, alpha=0.10, zorder=0)
    ax.axhline(20.6, lw=1.4, color=INK2, alpha=0.8, ls=(0, (5, 3)))
    ax.axhline(24.2, lw=1.6, color="#e34948", alpha=0.9)
    ax.text(0.031, 26, "the 24 mK anomaly", fontsize=8.6, color="#e34948")
    ax.text(0.031, 14.0, "never measured: no data below 20.6 mK",
            fontsize=8.4, color=INK2)
    ax.set_yscale("log")
    ax.set_ylim(0.08, 400)
    style(ax, r"$|J_\pm| / J_{zz}$", "temperature (mK)",
          r"(c)  Ce$_2$Hf$_2$O$_7$: where the photon actually is")
    ax.legend(frameon=False, fontsize=8.6, loc="lower right",
              labelcolor=INK)

    fig.suptitle("What one ring-exchange move costs, and why it matters",
                 fontsize=12.5, color=INK, y=0.995, x=0.5)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    for ext in ("png", "pdf"):
        fig.savefig(OUT / ("fig_denominator." + ext), dpi=200,
                    facecolor=fig.get_facecolor())
    print("wrote", OUT / "fig_denominator.png")
    print("\n  x     D_eff(cub16)  gap   D/gap    g6(mK)   K6(mK)")
    gi = np.interp(x, gap[:, 0], gap[:, 1])
    for i, xv in enumerate(x):
        print("  %.2f     %6.3f    %6.3f  %5.2f   %8.2f %8.2f"
              % (xv, cub[i, 1], gi[i], cub[i, 1] / gi[i], g6[i], k6[i]))


if __name__ == "__main__":
    main()
