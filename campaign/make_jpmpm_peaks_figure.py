#!/usr/bin/env python3
"""Certified winding-free peak positions across the (Jpm, Jpmpm) plane.

Panel (a): the gauge (ring-exchange) and spinon peak temperatures against
|Jpm|, one curve per Jpmpm.  Panel (b): their ratio, against the measured
Ce2Hf2O7 separation.  Both are read from the ``*_bordered.json`` records --
the complete-band fold, in which all 90 ice-descended raw levels are removed
and replaced.  The earlier records removed only the ``n_found`` levels the
anchor solve had located, so wherever the band was incomplete the remaining
ice-descended raw levels stayed in the ensemble and the peaks were not
winding-free statements.

Ratio convention matches ``make_jpmpm_heatmap.py``: the gauge peak is the
highest maximum below ``SPINON_SPLIT`` and the spinon peak the highest above.

Outputs: outputs/jpmpm_fold_plane/fig_jpmpm_peaks.{pdf,png}
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

SPINON_SPLIT = 0.12
CHO_RATIO = 24.4 / 63.5          # Smith et al., PRL 135, 086702 (2025)
PPS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]

# Evenly lightness-spaced steps of one blue hue: an ordinal ramp for Jpmpm.
# The palette's own named steps cannot carry seven levels (adjacent dL 0.047
# against a 0.06 floor); these were generated between its light and dark
# endpoints and pass the ordinal checks in both modes.
RAMP = ["#86b6ef", "#719fd8", "#5c89c1", "#4874ab",
        "#345f95", "#214a80", "#0d366b"]
ACCENT = "#d03b3b"
MUTED = "#898781"
INK = "#0b0b0b"
SECOND = "#52514e"


def load():
    rows = []
    for path in sorted(PLANE.glob("point_*_bordered.json")):
        record = json.loads(path.read_text())
        peaks = np.asarray(record["fold_peaks"], float)
        below = peaks[peaks < SPINON_SPLIT]
        above = peaks[peaks >= SPINON_SPLIT]
        if not len(below) or not len(above):
            continue
        rows.append({
            "jpm": float(record["jpm"]),
            "jpmpm": float(record["jpmpm"]),
            "n_found": int(record["n_found"]),
            "low": float(below.max()),
            "high": float(above.max()),
            "peaks": peaks,
            "n_sub": int(len(below)),
        })
    if not rows:
        raise SystemExit(f"no bordered records under {PLANE}")
    return rows


def series(rows, jpmpm):
    chosen = [r for r in rows if abs(r["jpmpm"] - jpmpm) < 1e-9]
    chosen.sort(key=lambda r: abs(r["jpm"]))
    x = np.array([abs(r["jpm"]) for r in chosen])
    return (x,
            np.array([r["low"] for r in chosen]),
            np.array([r["high"] for r in chosen]),
            np.array([r["n_found"] == 90 for r in chosen]))


def scatter_split(axis, x, y, complete, colour):
    """Filled markers for a complete band, open markers otherwise."""
    if complete.any():
        axis.plot(x[complete], y[complete], "o", ms=4.6, color=colour,
                  mec="white", mew=0.8, ls="none", zorder=4)
    if (~complete).any():
        axis.plot(x[~complete], y[~complete], "o", ms=4.6, mfc="white",
                  mec=colour, mew=1.4, ls="none", zorder=4)


def main() -> None:
    rows = load()
    figure, (left, right) = plt.subplots(
        1, 2, figsize=(9.2, 4.0), constrained_layout=True)

    # every resolved maximum, so a branch line that jumps is visibly a
    # merge in the underlying peak set rather than an artefact of the plot
    for row in rows:
        colour = RAMP[PPS.index(round(row["jpmpm"], 2))]
        left.plot(np.full(len(row["peaks"]), abs(row["jpm"])), row["peaks"],
                  "o", ms=2.2, color=colour, alpha=0.4, ls="none", zorder=2)

    for index, jpmpm in enumerate(PPS):
        colour = RAMP[index]
        x, low, high, complete = series(rows, jpmpm)
        for branch in (high, low):
            left.plot(x, branch, "-", lw=1.7, color=colour, zorder=3)
            scatter_split(left, x, branch, complete, colour)
        ratio = low / high
        right.plot(x, ratio, "-", lw=1.7, color=colour, zorder=3,
                   label=f"{jpmpm:.2f}")
        scatter_split(right, x, ratio, complete, colour)

    left.set_yscale("log")
    left.set_ylim(2e-5, 0.85)
    left.set_ylabel(r"peak temperature  $T/J_{zz}$", fontsize=10.5)
    left.text(0.048, 0.56, "spinon branch", fontsize=9.5, color=SECOND,
              style="italic")
    left.text(0.048, 0.075, "gauge branch", fontsize=9.5, color=SECOND,
              style="italic")
    left.text(0.307, 3.2e-5, "sub-gauge maxima", fontsize=8.5, color=MUTED,
              style="italic", ha="right")
    left.set_title(r"(a) winding-free peak positions", fontsize=10.5,
                   loc="left", color=INK)

    right.set_yscale("log")
    right.set_ylim(3e-3, 0.62)
    right.set_ylabel(r"$T_{\rm gauge}/T_{\rm spinon}$", fontsize=10.5)
    right.axhline(CHO_RATIO, color=ACCENT, lw=1.7, ls="--", zorder=2)
    right.text(0.305, CHO_RATIO * 1.13,
               r"Ce$_2$Hf$_2$O$_7$ measured, 0.37", fontsize=9,
               color=ACCENT, ha="right", fontweight="semibold")
    ceiling = max(r["low"] / r["high"] for r in rows if r["n_found"] == 90)
    right.axhline(ceiling, color=MUTED, lw=1.2, ls=":", zorder=2)
    right.text(0.305, ceiling * 1.13, f"computed ceiling, {ceiling:.3f}",
               fontsize=9, color=MUTED, ha="right")
    right.set_title(r"(b) gauge/spinon separation", fontsize=10.5,
                    loc="left", color=INK)
    legend = right.legend(title=r"$J_{\pm\pm}/J_{zz}$", fontsize=8.5,
                          title_fontsize=8.5, loc="lower right", ncol=2,
                          frameon=True, framealpha=0.95, borderpad=0.5)
    legend.get_frame().set_edgecolor("#e1e0d9")

    for axis in (left, right):
        axis.set_xlabel(r"$|J_{\pm}|/J_{zz}$", fontsize=10.5)
        axis.set_xlim(0.035, 0.315)
        axis.grid(True, which="major", lw=0.5, color="#e1e0d9", zorder=0)
        axis.grid(True, which="minor", lw=0.35, color="#eeede8", zorder=0)
        axis.tick_params(labelsize=9, colors=SECOND)
        for spine in ("top", "right"):
            axis.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            axis.spines[spine].set_color("#c3c2b7")

    incomplete = [r for r in rows if r["n_found"] != 90]
    for suffix in ("pdf", "png"):
        figure.savefig(PLANE / f"fig_jpmpm_peaks.{suffix}", dpi=300)
    print(f"{len(rows)} cells plotted, {len(incomplete)} with an incomplete "
          f"band (open markers): "
          + ", ".join(f"({r['jpm']},{r['jpmpm']}):{r['n_found']}"
                      for r in incomplete))
    # where only one sub-spinon maximum survives, the gauge and sub-gauge
    # peaks have merged and the >=SPINON_SPLIT convention necessarily
    # reports the merged position -- the cause of the branch-line jumps
    merged = [r for r in rows if r["n_sub"] == 1 and abs(r["jpm"]) >= 0.2]
    print(f"{len(merged)} strong-coupling cells with a single sub-spinon "
          f"maximum (gauge/sub-gauge merged): "
          + ", ".join(f"({r['jpm']},{r['jpmpm']}):{r['low']:.4f}"
                      for r in merged))
    print(f"certified ratio ceiling {ceiling:.4f} vs Ce2Hf2O7 {CHO_RATIO:.3f}")
    print(f"wrote {PLANE}/fig_jpmpm_peaks.pdf and .png")


if __name__ == "__main__":
    main()
