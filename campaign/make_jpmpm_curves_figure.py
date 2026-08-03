#!/usr/bin/env python3
"""Fold C(T) peak positions vs Jpm, one curve family per Jpmpm.

Gauge branch (highest peak below the spinon window) and spinon branch are
connected per Jpmpm; additional sub-gauge peaks appear as bare markers.
Cells where the winding-free band has dissolved into the continuum
(n_found < 45) are drawn as open markers: their fold is essentially the
raw spectrum and carries no winding-free meaning.  Guides: g6 and g4.

Outputs: outputs/jpmpm_fold_plane/fig_jpmpm_curves.{pdf,png}
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PLANE = ROOT / "campaign" / "outputs" / "jpmpm_fold_plane"

PP = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
SPINON_SPLIT = 0.12
RAMP = plt.cm.Blues


def main() -> None:
    data: dict[float, list] = {pp: [] for pp in PP}
    for path in PLANE.glob("point_*.json"):
        record = json.loads(path.read_text())
        pp = round(abs(record["jpmpm"]), 4)
        if pp not in data:
            continue
        data[pp].append((abs(record["jpm"]),
                         np.asarray(record["fold_peaks"], float),
                         record.get("n_found", 90) >= 45))

    fig, axis = plt.subplots(figsize=(7.6, 5.6), constrained_layout=True)

    guide = np.geomspace(0.04, 0.36, 100)
    axis.plot(guide, 12 * guide**3, ls=":", lw=1.3, c="0.4")
    axis.plot(guide, 4 * guide**2, ls="--", lw=1.0, c="0.65")
    axis.text(0.045, 12 * 0.045**3 * 0.5, r"$g_6=12|J_{\pm}|^3$",
              fontsize=8, color="0.3", rotation=25)
    axis.text(0.045, 4 * 0.045**2 * 1.5, r"$g_4=4J_{\pm}^2$",
              fontsize=8, color="0.55", rotation=17)

    for index, pp in enumerate(PP):
        unique = {round(m, 4): (m, p, c) for m, p, c in data[pp]}
        rows = [unique[k] for k in sorted(unique)]
        if not rows:
            continue
        color = RAMP(0.35 + 0.6 * index / (len(PP) - 1))
        for branch_low in (True, False):
            xs, ys, solid = [], [], []
            for magnitude, peaks, certified in rows:
                subset = (peaks[peaks < SPINON_SPLIT] if branch_low
                          else peaks[peaks >= SPINON_SPLIT])
                if len(subset):
                    xs.append(magnitude)
                    ys.append(subset.max())
                    solid.append(certified)
            if not xs:
                continue
            xs, ys, solid = map(np.array, (xs, ys, solid))
            axis.plot(xs, ys, "-", lw=1.3, color=color, alpha=0.85,
                      label=(rf"$J_{{\pm\pm}}={pp:g}$"
                             if branch_low else None))
            axis.plot(xs[solid], ys[solid], "o", ms=5, mfc=color,
                      mec="white", mew=0.5)
            axis.plot(xs[~solid], ys[~solid], "o", ms=5.5, mfc="none",
                      mec=color, mew=1.3)
        # sub-gauge extras as bare markers
        for magnitude, peaks, certified in rows:
            low = np.sort(peaks[peaks < SPINON_SPLIT])
            for extra in low[:-1]:
                axis.plot(magnitude, extra, "v", ms=4, mfc=color,
                          mec="white", mew=0.4, alpha=0.8)

    axis.plot([], [], "o", ms=5.5, mfc="none", mec="0.4", mew=1.3,
              label="band dissolved (fold $\\approx$ raw)")
    axis.plot([], [], "v", ms=4, mfc="0.4", mec="white",
              label="additional low-$T$ peak")

    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel(r"$|J_{\pm}|/J_{zz}$")
    axis.set_ylabel(r"$T_{\mathrm{peak}}/J_{zz}$")
    axis.set_title("winding-free $C(T)$ peaks across the "
                   r"$(J_{\pm}, J_{\pm\pm})$ plane (cubic-16 fold)",
                   fontsize=10)
    axis.legend(frameon=False, fontsize=7.5, loc="center left",
                bbox_to_anchor=(1.01, 0.5))
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.tick_params(labelsize=8)

    fig.savefig(PLANE / "fig_jpmpm_curves.pdf")
    fig.savefig(PLANE / "fig_jpmpm_curves.png", dpi=180)
    print(f"wrote {PLANE / 'fig_jpmpm_curves.pdf'}")


if __name__ == "__main__":
    main()
