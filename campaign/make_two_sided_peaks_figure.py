#!/usr/bin/env python3
"""Every specific-heat peak of every winding-free definition, vs coupling.

Reads the cached two-sided-fold curves (fold_<tag>.npz: raw ED, H + C
counterterm, splice, two-sided fold) and re-extracts peak positions as true
local maxima of C(T) via scipy.signal.find_peaks, because the campaign's
two-window ``peaks()`` helper clips whichever peak crosses its T = 0.05
split (visible as values pinned at 0.0497 in the fold_*.json records).

Left panel: all peak positions, log-log, against the artifact scale
g4 = 4 Jpm^2, the ring scale g6 = 12|Jpm|^3, and the spinon plateau.
Right panel: the lowest peak of each definition in units of g6 -- the
definitional-spread view (counterterm / splice / fold agree to a few
percent through -0.25, then diverge as the obstruction bound predicts).

Outputs: outputs/two_sided_fold/fig_fold_peaks.{pdf,png} + peaks_all.json
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.signal import find_peaks  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FOLD = ROOT / "campaign" / "outputs" / "two_sided_fold"

SERIES = [  # fixed order; CVD-validated palette; marker = secondary encoding
    ("raw", "raw ED", "#0072b2", "o"),
    ("hc", "counterterm $H+C$", "#e69f00", "s"),
    ("splice", "splice", "#009e73", "^"),
    ("fold", "two-sided fold", "#d55e00", "D"),
]
PROMINENCE = 0.005  # of the curve maximum


def all_peaks(grid: np.ndarray, curve: np.ndarray) -> np.ndarray:
    index, _ = find_peaks(curve, prominence=PROMINENCE * curve.max())
    return grid[index]


def main() -> None:
    files = sorted(FOLD.glob("fold_*.npz"),
                   key=lambda p: -float(np.load(p)["jpm"]))
    records = []
    for path in files:
        data = np.load(path)
        entry = {"jpm": float(data["jpm"])}
        for key, *_ in SERIES:
            entry[key] = all_peaks(data["grid"], data[f"curve_{key}"]).tolist()
        records.append(entry)
    (FOLD / "peaks_all.json").write_text(json.dumps(records, indent=1))

    coupling = np.array([r["jpm"] for r in records])
    magnitude = np.abs(coupling)

    fig, (left, right) = plt.subplots(
        1, 2, figsize=(10.6, 4.4), constrained_layout=True)

    # ---- left: every peak position -------------------------------------
    guide = np.geomspace(0.04, 0.45, 200)
    left.plot(guide, 4 * guide**2, ls="--", lw=1.0, c="0.62", zorder=1)
    left.plot(guide, 12 * guide**3, ls=":", lw=1.2, c="0.45", zorder=1)
    left.axhline(0.32, ls="-", lw=0.8, c="0.85", zorder=0)
    left.text(0.065, 4 * 0.065**2 * 1.45, r"$g_4=4J_{\pm}^2$",
              fontsize=8, color="0.45", rotation=14)
    left.text(0.042, 12 * 0.042**3 * 0.60, r"$g_6=12|J_{\pm}|^3$",
              fontsize=8, color="0.35", rotation=20)
    left.text(0.042, 0.335, "spinon plateau", fontsize=8, color="0.55")

    jitter = {"raw": 0.985, "hc": 1.0, "splice": 1.015, "fold": 1.03}
    for key, label, color, marker in SERIES:
        xs, ys = [], []
        for r in records:
            xs.extend([abs(r["jpm"]) * jitter[key]] * len(r[key]))
            ys.extend(r[key])
        left.plot(xs, ys, marker, ms=5.5, mfc=color, mec="white", mew=0.6,
                  ls="none", label=label, zorder=3)

    left.set_xscale("log")
    left.set_yscale("log")
    left.set_xlabel(r"$|J_{\pm}|/J_{zz}$")
    left.set_ylabel(r"$T_{\mathrm{peak}}/J_{zz}$")
    left.set_title("every $C(T)$ peak, every definition", fontsize=10)
    left.legend(frameon=False, fontsize=8, loc="upper left",
                bbox_to_anchor=(0.02, 0.82))
    left.tick_params(labelsize=8)

    # ---- right: the ring branch over g6 --------------------------------
    # The ring peak is the highest local maximum below the spinon shoulder;
    # min() would jump to the third (internal-gap) peak once it appears.
    RING_WINDOW = 0.15
    nudge = {"raw": 0, "hc": -6, "splice": 6, "fold": 0}
    for key, label, color, marker in SERIES:
        ring = np.array([
            max((p for p in r[key] if p < RING_WINDOW), default=np.nan)
            for r in records])
        ratio = ring / (12 * magnitude**3)
        right.plot(magnitude, ratio, marker + "-",
                   ms=5.5, lw=1.6, color=color, mfc=color, mec="white",
                   mew=0.6, label=label)
        finite = np.flatnonzero(np.isfinite(ratio))
        end = finite[-1]
        right.annotate(label, (magnitude[end], ratio[end]),
                       xytext=(6, nudge[key]), textcoords="offset points",
                       fontsize=7.5, color=color, va="center")

    right.axhline(1.0, ls=":", lw=1.0, c="0.55")
    right.text(0.052, 1.06, "perturbative ring peak", fontsize=8, color="0.4")
    right.set_xscale("log")
    right.set_yscale("log")
    right.set_xlim(0.045, 0.70)
    right.set_xlabel(r"$|J_{\pm}|/J_{zz}$")
    right.set_ylabel(r"$T_{\mathrm{ring\ peak}}\,/\,g_6$")
    right.set_title(
        "ring branch in ring units: the definitional spread", fontsize=10)
    right.tick_params(labelsize=8)

    for axis in (left, right):
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    fig.savefig(FOLD / "fig_fold_peaks.pdf")
    fig.savefig(FOLD / "fig_fold_peaks.png", dpi=180)
    for r in records:
        print(f"jpm={r['jpm']:+.2f}  " + "  ".join(
            f"{key}:[" + ",".join(f"{v:.4f}" for v in r[key]) + "]"
            for key, *_ in SERIES))
    print(f"wrote {FOLD / 'fig_fold_peaks.pdf'}")


if __name__ == "__main__":
    main()
