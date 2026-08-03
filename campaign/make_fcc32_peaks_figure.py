#!/usr/bin/env python3
"""Two-cluster winding-free peak comparison: cubic-16 vs FCC-32.

Band-only C(T) peak positions on both clusters, same estimator on each
side: cubic-16 from the exact full-Q masked band (masked_band_complete
cache), FCC-32 from the validated multi-anchor self-consistent band
(multi_anchor_band cache, L=1).  Left: every band peak vs coupling with
the g4/g6 guides.  Right: the ring branch in units of g6 -- the cluster
error bar made visible, growing from x3 at weak coupling.

Outputs: outputs/multi_anchor_band/fig_fcc32_peaks.{pdf,png}
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
BAND16 = ROOT / "campaign" / "outputs" / "masked_band_complete"
BAND32 = ROOT / "campaign" / "outputs" / "multi_anchor_band"

C16, C32, NEUTRAL = "#0072b2", "#d55e00", "0.5"
GRID = np.geomspace(1e-5, 0.3, 3000)
RING_WINDOW = 0.15

# cubic-16 truncation calibration: exact ring peak / L=1 ring peak,
# measured at |Jpm| = 0.05, 0.20, 0.30 (Aug-02).  Used as the transfer
# estimate of the L=1 underestimate on FCC-32; held flat beyond 0.30.
CAL_X = np.array([0.05, 0.20, 0.30, 0.40])
CAL_FACTOR = np.array([0.00126 / 0.00111, 0.02828 / 0.01632,
                       0.04214 / 0.02578, 0.04214 / 0.02578])


def band_peaks(levels):
    shifted = np.sort(levels) - np.min(levels)
    beta = 1.0 / GRID[:, None]
    boltzmann = np.exp(-beta * shifted[None, :])
    partition = boltzmann.sum(axis=1)
    first = (boltzmann @ shifted) / partition
    second = (boltzmann @ shifted**2) / partition
    heat = (second - first**2) / GRID**2
    index, _ = find_peaks(heat, prominence=0.005 * heat.max())
    return GRID[index]


def cubic16_data():
    rows = []
    for path in sorted(BAND16.glob("band_-0p*.json")):
        record = json.loads(path.read_text())
        levels = np.asarray(
            [x for x in record["levels"] if x is not None], float)
        levels = levels[np.isfinite(levels)]
        rows.append((abs(record["jpm"]), band_peaks(levels)))
    return sorted(rows)


def fcc32_data():
    rows = []
    for path in sorted(BAND32.glob("band_fcc32_L1_*.json")):
        record = json.loads(path.read_text())
        rows.append((abs(record["jpm"]), np.asarray(record["peaks"])))
    return sorted(rows)


def main() -> None:
    c16 = cubic16_data()
    c32 = fcc32_data()

    fig, (left, right) = plt.subplots(
        1, 2, figsize=(10.6, 4.4), constrained_layout=True)

    guide = np.geomspace(0.04, 0.55, 200)
    left.plot(guide, 12 * guide**3, ls=":", lw=1.2, c="0.45")
    left.plot(guide, 4 * guide**2, ls="--", lw=1.0, c="0.62")
    left.text(0.05, 12 * 0.05**3 * 1.8, r"$g_6$", fontsize=9, color="0.35")
    left.text(0.05, 4 * 0.05**2 * 1.8, r"$g_4$", fontsize=9, color="0.45")

    for rows, color, marker, label in (
            (c16, C16, "o", "cubic-16 (exact masked band)"),
            (c32, C32, "D", "FCC-32 (multi-anchor band, $L=1$)")):
        xs, ys = [], []
        for magnitude, peaks in rows:
            xs.extend([magnitude] * len(peaks))
            ys.extend(peaks)
        left.plot(xs, ys, marker, ms=5.5, mfc=color, mec="white", mew=0.6,
                  ls="none", label=label)

    x32 = np.array([m for m, _ in c32])
    ring32 = np.array([min(p) for _, p in c32])
    corrected = ring32 * np.interp(x32, CAL_X, CAL_FACTOR)
    left.fill_between(x32, ring32, corrected, color=C32, alpha=0.18, lw=0,
                      label="truncation band (cubic-16 transfer)")
    left.plot(x32, corrected, ls="--", lw=1.0, color=C32, alpha=0.7)

    left.set_xscale("log")
    left.set_yscale("log")
    left.set_xlabel(r"$|J_{\pm}|/J_{zz}$")
    left.set_ylabel(r"$T_{\mathrm{peak}}/J_{zz}$")
    left.set_title("every band peak, both clusters", fontsize=10)
    left.legend(frameon=False, fontsize=8, loc="upper left")
    left.tick_params(labelsize=8)

    for rows, color, marker, label in (
            (c16, C16, "o", "cubic-16"), (c32, C32, "D", "FCC-32")):
        xs = np.array([magnitude for magnitude, _ in rows])
        ring = np.array([
            max((p for p in peaks if p < RING_WINDOW), default=np.nan)
            for _, peaks in rows])
        ratio = ring / (12 * xs**3)
        right.plot(xs, ratio, marker + "-", ms=5.5, lw=1.6, color=color,
                   mfc=color, mec="white", mew=0.6, label=label)
        finite = np.flatnonzero(np.isfinite(ratio))
        right.annotate(label, (xs[finite[-1]], ratio[finite[-1]]),
                       xytext=(6, 0), textcoords="offset points",
                       fontsize=7.5, color=color, va="center")

    ratio32 = ring32 / (12 * x32**3)
    right.fill_between(x32, ratio32,
                       corrected / (12 * x32**3),
                       color=C32, alpha=0.18, lw=0)
    right.plot(x32, corrected / (12 * x32**3), ls="--", lw=1.0,
               color=C32, alpha=0.7)

    right.axhline(1.0, ls=":", lw=1.0, c="0.55")
    right.set_xscale("log")
    right.set_yscale("log")
    right.set_xlim(0.04, 0.72)
    right.set_xlabel(r"$|J_{\pm}|/J_{zz}$")
    right.set_ylabel(r"$T_{\mathrm{ring\ peak}}\,/\,g_6$")
    right.set_title("the cluster error bar, quantified", fontsize=10)
    right.tick_params(labelsize=8)

    for axis in (left, right):
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    fig.savefig(BAND32 / "fig_fcc32_peaks.pdf")
    fig.savefig(BAND32 / "fig_fcc32_peaks.png", dpi=180)
    for magnitude, peaks in c32:
        ring16 = dict((m, max((p for p in pk if p < RING_WINDOW),
                              default=np.nan)) for m, pk in c16)
        near = min(ring16, key=lambda m: abs(m - magnitude))
        print(f"|jpm|={magnitude:.2f}  fcc32 ring {min(peaks):.5f}  "
              f"cubic16 ring {ring16[near]:.5f} (at {near:.2f})  "
              f"ratio {ring16[near] / min(peaks):.2f}")
    print(f"wrote {BAND32 / 'fig_fcc32_peaks.pdf'}")


if __name__ == "__main__":
    main()
