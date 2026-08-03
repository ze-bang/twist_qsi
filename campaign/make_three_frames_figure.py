#!/usr/bin/env python3
"""Ring and spinon peaks for all winding-free architectures vs Jpm."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

plt.rcParams.update({
    "font.family": "serif", "mathtext.fontset": "cm", "font.size": 8.5,
    "legend.frameon": False, "savefig.dpi": 400, "figure.facecolor": "white",
    "xtick.labelsize": 7.6, "ytick.labelsize": 7.6})

STY = {"periodic": ("#a95a45", "s", "periodic"),
       "ice": ("#356399", "o", "ice frame"),
       "dressed": ("#d1261f", "^", "dressed frame"),
       "clean": ("#2e8b57", "v", "clean frame"),
       "union": ("#7a3f9d", "D", "union frame")}


def main() -> None:
    rows = json.loads(
        (ROOT / "campaign/outputs/three_frames.json").read_text())["points"]
    rows.sort(key=lambda r: r["jpm"])
    x = np.array([r["jpm"] for r in rows])
    g6 = np.array([r["g6"] for r in rows])
    coh5 = np.array([r["coherence_rank_0p5"] for r in rows])
    coh7 = np.array([r["coherence_rank_0p7"] for r in rows])

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(4.8, 4.6), sharex=True,
                                 height_ratios=(1.6, 1.0))
    m = x < 0
    edge7 = x[m][coh7[m] >= 90].min() if np.any(coh7[m] >= 90) else 0
    edge5 = x[m][coh5[m] >= 90].min() if np.any(coh5[m] >= 90) else 0
    for a in (a1, a2):
        a.axvspan(-0.31, edge5, color="0.90", zorder=0)
        a.axvspan(-0.31, edge7, color="0.95", zorder=0)
    for key, (tone, mk, lab) in STY.items():
        low = np.array([r.get(f"{key}_low") or np.nan for r in rows])
        a1.plot(x[m], (low / g6)[m], color=tone, marker=mk, ms=3.0, lw=1.0,
                label=lab, zorder=6 if key == "union" else 3)
        high = np.array([r.get(f"{key}_high") or np.nan for r in rows])
        high = np.where(high > 0.15, high, np.nan)
        a2.plot(x[m], high[m], color=tone, marker=mk, ms=3.0, lw=1.0,
                zorder=6 if key == "union" else 3)
    a1.set_yscale("log")
    a1.set_ylim(0.03, 4.0)
    a1.set_ylabel(r"ring peak $T_{\rm low}/g_6$")
    a1.legend(fontsize=7.0, ncol=2, loc="lower left")
    a2.set_ylabel(r"spinon peak $T_{\rm high}\,[J_{zz}]$")
    a2.set_xlabel(r"$J_\pm/J_{zz}$")
    a2.set_ylim(0.25, 0.45)
    for a in (a1, a2):
        a.spines["top"].set_visible(False)
        a.spines["right"].set_visible(False)
    fig.tight_layout(pad=0.4)
    for suffix in ("png", "pdf"):
        fig.savefig(ROOT / f"paper/figs/fig_three_frames.{suffix}",
                    bbox_inches="tight")

    sel = m & (x >= edge5)
    icel = np.array([r.get("ice_low") or np.nan for r in rows])
    for key in ("dressed", "clean", "union"):
        low = np.array([r.get(f"{key}_low") or np.nan for r in rows])
        print(f"{key}: max |low/ice - 1| coherent window = "
              f"{np.nanmax(np.abs(low[sel] / icel[sel] - 1)):.4f}")
    print("figure updated")


if __name__ == "__main__":
    main()
