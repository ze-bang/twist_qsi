#!/usr/bin/env python3
"""Peak-structure heatmaps over the (Jpm, Jpmpm) plane, cubic-16 fold.

Panels: (a) position of the highest sub-spinon peak (the gauge-sector
anomaly), (b) the spinon peak, (c) their ratio -- the experimental
observable (Ce2Hf2O7: 24/65 mK ~ 0.37; Ce2Zr2O7: single peak).  Cells are
annotated with the number of fold C(T) maxima; points where the whole
winding-free band is embedded in the continuum (n_found = 0) are hatched.
Material anchors: CHO at (-0.125, 0.085) (NLC-A ratios), CZO near
(-0.30, ~0).

Outputs: outputs/jpmpm_fold_plane/fig_jpmpm_heatmap.{pdf,png}
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PLANE = ROOT / "campaign" / "outputs" / "jpmpm_fold_plane"

JPM = [-0.05, -0.09, -0.125, -0.16, -0.20, -0.24, -0.27, -0.30]
PP = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
SPINON_SPLIT = 0.12   # peaks above this are the spinon anomaly


def load_grid():
    low = np.full((len(PP), len(JPM)), np.nan)
    high = np.full_like(low, np.nan)
    counts = np.full_like(low, np.nan)
    embedded = np.zeros_like(low, dtype=bool)
    for path in PLANE.glob("point_*.json"):
        record = json.loads(path.read_text())
        jpm, pp = record["jpm"], record["jpmpm"]
        try:
            column = JPM.index(round(jpm, 4))
            row = PP.index(round(abs(pp), 4))
        except ValueError:
            continue
        peaks = np.asarray(record["fold_peaks"], float)
        counts[row, column] = len(peaks)
        # a mostly-embedded band means the fold is essentially the raw
        # spectrum: its peaks must not be read as winding-free statements
        if record.get("n_found", 90) < 45:
            embedded[row, column] = True
        above = peaks[peaks >= SPINON_SPLIT]
        below = peaks[peaks < SPINON_SPLIT]
        if len(above):
            high[row, column] = above.max()
        if len(below):
            low[row, column] = below.max()
    return low, high, counts, embedded


def panel(axis, data, title, cmap, norm=None):
    x = np.abs(JPM)
    mesh = axis.pcolormesh(
        np.arange(len(JPM) + 1), np.arange(len(PP) + 1), data,
        cmap=cmap, norm=norm, shading="flat", edgecolors="white",
        linewidth=0.8)
    axis.set_xticks(np.arange(len(JPM)) + 0.5)
    axis.set_xticklabels([f"{v:g}" for v in x], fontsize=7, rotation=45)
    axis.set_yticks(np.arange(len(PP)) + 0.5)
    axis.set_yticklabels([f"{v:g}" for v in PP], fontsize=7)
    axis.set_xlabel(r"$|J_{\pm}|/J_{zz}$", fontsize=9)
    axis.set_title(title, fontsize=9.5)
    return mesh


def mark(axis, jpm, pp, label, dy=0.0):
    column = min(range(len(JPM)), key=lambda i: abs(JPM[i] - jpm))
    row = min(range(len(PP)), key=lambda i: abs(PP[i] - pp))
    axis.plot(column + 0.5, row + 0.5, "*", ms=13, mfc="white",
              mec="black", mew=1.0, zorder=5)
    axis.annotate(label, (column + 0.5, row + 0.5), xytext=(0, 9 + dy),
                  textcoords="offset points", ha="center", fontsize=8,
                  fontweight="bold", zorder=6)


def main() -> None:
    low, high, counts, embedded = load_grid()
    ratio = low / high

    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.1),
                             constrained_layout=True)

    specs = [
        (low, r"(a) gauge peak $T_{\rm low}/J_{zz}$", "Blues",
         LogNorm(vmin=np.nanmin(low), vmax=np.nanmax(low))),
        (high, r"(b) spinon peak $T_{\rm high}/J_{zz}$", "Blues", None),
        (ratio, r"(c) ratio $T_{\rm low}/T_{\rm high}$", "Oranges",
         LogNorm(vmin=np.nanmin(ratio), vmax=1.0)),
    ]
    for axis, (data, title, cmap, norm) in zip(axes, specs):
        mesh = panel(axis, data, title, cmap, norm)
        fig.colorbar(mesh, ax=axis, shrink=0.85)
        for row in range(len(PP)):
            for column in range(len(JPM)):
                if embedded[row, column]:
                    axis.add_patch(plt.Rectangle(
                        (column, row), 1, 1, fill=False, hatch="////",
                        edgecolor="0.35", linewidth=0))
                elif np.isfinite(counts[row, column]):
                    axis.text(column + 0.5, row + 0.15,
                              f"{int(counts[row, column])}",
                              ha="center", fontsize=5.5, color="0.35")
        axis.add_patch(plt.Rectangle((0, 0), 0, 0, fill=False, hatch="////",
                                     edgecolor="0.35", linewidth=0,
                                     label="band dissolved into continuum"))
        mark(axis, -0.125, 0.085, "CHO")
        mark(axis, -0.30, 0.0, "CZO", dy=0)
    axes[0].set_ylabel(r"$J_{\pm\pm}/J_{zz}$", fontsize=9)
    axes[0].legend(loc="upper right", fontsize=6.5, frameon=False)

    fig.savefig(PLANE / "fig_jpmpm_heatmap.pdf")
    fig.savefig(PLANE / "fig_jpmpm_heatmap.png", dpi=180)

    finite = np.isfinite(ratio)
    if finite.any():
        best = np.nanmax(ratio)
        where = np.unravel_index(np.nanargmax(ratio), ratio.shape)
        print(f"max ratio {best:.3f} at jpm={JPM[where[1]]}, "
              f"pp={PP[where[0]]}")
    print(f"{int(np.isfinite(counts).sum())}/{counts.size} cells filled")
    print(f"wrote {PLANE / 'fig_jpmpm_heatmap.pdf'}")


if __name__ == "__main__":
    main()
