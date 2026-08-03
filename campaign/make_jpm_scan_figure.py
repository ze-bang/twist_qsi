#!/usr/bin/env python3
"""Lower-peak position vs Jpm across both flux sectors.

Reads ``campaign/outputs/jpm_scan_lower_peak.json`` and draws, on the pi-flux
(Jpm<0) and zero-flux (Jpm>0) branches: the winding-free lower-peak position,
the periodic-ED lower peak (the winding artifact), and the perturbative scales
g_6 and g_4.  The region where the ice-descended band is no longer isolated is
shaded; there the construction is undefined and no point is drawn.
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
FIGS.mkdir(parents=True, exist_ok=True)

BARE = "#a95a45"    # periodic ED
CLEAN = "#356399"   # winding-free
G6C = "0.45"
G4C = "#c47a5f"

plt.rcParams.update({
    "font.family": "serif", "mathtext.fontset": "cm",
    "font.size": 8.5, "axes.labelsize": 8.5, "axes.titlesize": 8.5,
    "legend.fontsize": 7.0, "xtick.labelsize": 7.6, "ytick.labelsize": 7.6,
    "axes.linewidth": 0.6, "legend.frameon": False,
    "figure.facecolor": "white", "savefig.facecolor": "white", "savefig.dpi": 400,
})


def main() -> None:
    data = json.loads((ROOT / "campaign" / "outputs" / "jpm_scan_lower_peak.json").read_text())
    points = sorted(data["points"], key=lambda p: p["jpm"])
    # A recorded "lower" maximum near the ice scale means the ring peak fell
    # below the temperature grid floor (|Jpm| so small that g_6 < 10 uK at the
    # reference Ja); that is not a resolved ring peak, so the point is dropped.
    RING_CEILING = 0.1
    good = [p for p in points
            if p["well_defined"]
            and p.get("winding_free", {}).get("T_low_Jzz", np.inf) < RING_CEILING]
    unresolved = [p for p in points
                  if p["well_defined"]
                  and p.get("winding_free", {}).get("T_low_Jzz", np.inf) >= RING_CEILING]
    bad = [p for p in points if not p["well_defined"]]

    figure, ax = plt.subplots(figsize=(3.6, 2.6))
    x = np.array([p["jpm"] for p in good])
    wf = np.array([p["winding_free"]["T_low_Jzz"] for p in good])
    pe = np.array([p["periodic"].get("T_low_Jzz", np.nan) for p in good])

    dense = np.linspace(-0.32, 0.06, 400)
    ax.plot(dense, 12.0 * np.abs(dense) ** 3, color=G6C, lw=1.1, ls=(0, (5, 1.6)),
            label=r"$g_6=12|J_\pm|^3/J_{zz}^2$")
    ax.plot(dense, 4.0 * dense ** 2, color=G4C, lw=1.0, ls=(0, (1.5, 1.5)),
            label=r"$g_4=4J_\pm^2/J_{zz}$")

    for sign in (-1, 1):
        m = np.sign(x) == sign
        ax.plot(x[m], pe[m], color=BARE, lw=1.2, marker="s", ms=2.6,
                label="periodic ED lower peak" if sign < 0 else None)
        ax.plot(x[m], wf[m], color=CLEAN, lw=1.4, marker="o", ms=3.0,
                label="winding-free lower peak" if sign < 0 else None)

    if bad:
        edge = max(p["jpm"] for p in bad if p["jpm"] < 0)
        ax.axvspan(-0.32, edge + 0.005, color="0.88", lw=0, zorder=0)
        ax.text(edge - 0.055, 2.5e-3, "band not\nisolated", fontsize=6.6,
                color="0.35", ha="center")

    ax.axvline(0.0, color="0.8", lw=0.6)
    ax.set_yscale("log")
    ax.set_xlim(-0.32, 0.06)
    ax.set_ylim(6e-6, 0.6)
    ax.set_xlabel(r"$J_\pm/J_{zz}$")
    ax.set_ylabel(r"lower-peak position $T/J_{zz}$")
    ax.text(-0.055, 1.5e-5, r"$\pi$ flux", fontsize=7.2, color="0.3", ha="center")
    ax.text(0.033, 1.5e-5, r"$0$ flux", fontsize=7.2, color="0.3", ha="center")
    ax.legend(loc="lower left", handlelength=1.7, borderaxespad=0.4,
              labelspacing=0.3, bbox_to_anchor=(0.03, 0.03))

    figure.tight_layout(pad=0.4)
    for suffix in ("pdf", "png"):
        figure.savefig(FIGS / f"fig_jpm_scan.{suffix}", bbox_inches="tight")
    plt.close(figure)

    print(f"{len(good)} resolved, {len(unresolved)} ring peak below grid floor, "
          f"{len(bad)} not isolated")
    for p in good:
        print(f"  jpm={p['jpm']:+.2f} wf={p['winding_free']['T_low_Jzz']:.5g} "
              f"g6={p['g6_Jzz']:.5g} ovl={p['model_overlap_min']:.3f}")
    print(f"wrote {FIGS / 'fig_jpm_scan.pdf'}")


if __name__ == "__main__":
    main()
