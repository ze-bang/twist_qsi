#!/usr/bin/env python3
"""fig_peak_tracking.pdf: every maximum of C(T), and what converging the anchor does.

Three panels, all showing every labelled maximum of C(T) against |J_pm| with the
two candidate scales drawn in:

  (a) periodic ED -- the baseline, which tracks the artifact scale g_4;
  (b) the counterterm with the anchor iterated three times, as in the earlier
      version of this campaign: a spurious transport-sector branch runs along
      the bottom at every coupling;
  (c) the counterterm with the Brillouin-Wigner condition solved to 1e-13: the
      sector branch is gone and only the ring and spinon anomalies remain.

Panel (b) is kept deliberately.  It is the measurement that shows the branch was
an unconverged fixed point rather than a property of the construction, and
removing it from the figure would leave that claim unsupported.

Filled symbols joined by lines are the pi-flux side; open symbols the zero-flux
side.  Marker shape is the label: circle = ring (gauge for periodic), triangle =
transport sector, square = spinon.

Usage: make_spine_tracking_figure.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
LEGACY = ROOT / "campaign" / "outputs" / "method_comparison"
SPINE = ROOT / "campaign" / "outputs" / "counterterm_spine"
FIGS = ROOT / "paper" / "figs"
FIGS.mkdir(parents=True, exist_ok=True)

BLUE, VERM, PINK, GREY = "#0072B2", "#D55E00", "#CC79A7", "#5C5C5C"
COLOUR = {"ring": BLUE, "gauge": BLUE, "sector": PINK, "spinon": VERM}
MARKER = {"ring": "o", "gauge": "o", "sector": "^", "spinon": "s"}
SHOWN = ("ring", "gauge", "sector", "spinon")

plt.rcParams.update({
    "font.size": 8.5, "axes.labelsize": 8.5, "axes.titlesize": 9,
    "legend.fontsize": 7.0, "xtick.labelsize": 7.6, "ytick.labelsize": 7.6,
    "axes.linewidth": 0.7, "xtick.direction": "in", "ytick.direction": "in",
    "lines.linewidth": 1.4, "lines.markersize": 4.6,
    "axes.grid": True, "grid.alpha": 0.22, "grid.linewidth": 0.5,
    "figure.dpi": 160, "savefig.bbox": "tight", "savefig.facecolor": "white",
})


def load_series():
    """(title, [(jpm, peaks)]) for each panel."""
    spine = sorted(json.loads((SPINE / "spine_summary.json").read_text()),
                   key=lambda r: r["jpm"])
    series = [
        ("(a) periodic ED (baseline)",
         [(r["jpm"], r["periodic_peaks"]) for r in spine]),
    ]
    legacy_path = LEGACY / "peak_labels.json"
    if legacy_path.exists():
        legacy = sorted(json.loads(legacy_path.read_text()),
                        key=lambda r: r["jpm"])
        series.append(
            (r"(b) counterterm, anchor iterated $3\times$",
             [(r["jpm"], r.get("hc", [])) for r in legacy]))
    series.append(
        (r"(c) counterterm, anchor solved to $10^{-13}$",
         [(r["jpm"], r["counterterm_peaks"]) for r in spine]))
    return series


def main() -> None:
    series = load_series()
    figure, axes = plt.subplots(1, len(series),
                                figsize=(3.6 * len(series), 3.3), sharey=True)
    axes = np.atleast_1d(axes)
    grid = np.geomspace(0.008, 0.33, 200)

    for axis, (title, points) in zip(axes, series):
        axis.loglog(grid, 4 * grid**2, ":", color=GREY, lw=1.3,
                    label=r"$g_4=4J_\pm^2$", zorder=1)
        axis.loglog(grid, 12 * grid**3, "--", color=GREY, lw=1.3,
                    label=r"$g_6=12|J_\pm|^3$", zorder=1)
        for label in SHOWN:
            for negative, fill, style in ((True, True, "-"),
                                          (False, False, "")):
                x, y = [], []
                for jpm, peaks in points:
                    if (jpm < 0) != negative:
                        continue
                    for peak in peaks or []:
                        if peak["label"] != label or peak.get("at_grid_edge"):
                            continue
                        x.append(abs(jpm))
                        y.append(peak["T"])
                if not x:
                    continue
                order = np.argsort(x)
                axis.loglog(np.asarray(x)[order], np.asarray(y)[order],
                            MARKER[label] + style, color=COLOUR[label],
                            mfc=COLOUR[label] if fill else "none",
                            label=(label if negative else None), zorder=3)
        axis.set_title(title, loc="left", pad=4)
        axis.set_xlabel(r"$|J_\pm|/J_{zz}$")
        axis.set_xlim(0.008, 0.35)
        axis.set_ylim(1e-6, 2.0)
    axes[0].set_ylabel(r"$T_{\rm peak}/J_{zz}$")
    axes[0].legend(frameon=False, loc="upper left", ncol=1)
    figure.suptitle(r"filled: $J_\pm<0$ ($\pi$ flux)   open: $J_\pm>0$ "
                    r"(zero flux)", y=1.03, fontsize=8.5)
    figure.tight_layout()
    figure.savefig(FIGS / "fig_peak_tracking.pdf")
    plt.close(figure)
    print(f"wrote {FIGS / 'fig_peak_tracking.pdf'}")

    for title, points in series:
        counts = {}
        for _, peaks in points:
            for peak in peaks or []:
                if peak.get("at_grid_edge"):
                    continue
                counts[peak["label"]] = counts.get(peak["label"], 0) + 1
        print(f"  {title}: {counts}")


if __name__ == "__main__":
    main()
