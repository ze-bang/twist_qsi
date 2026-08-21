#!/usr/bin/env python3
"""fig_fcc32_parity.pdf: E and B channels never light the same level.

Panels (a) and (b) are level diagrams -- every computed level drawn at its own
energy, in the column of its momentum star, coloured by which probe reaches it.
Panel (c) is the whole point in one plot: every level's electric weight against
its magnetic weight, log-log.  If the two channels were merely *different* the
points would scatter; they instead pin to the axes, because `S^z` is odd and
the ring exchange even under the global spin flip `S^z -> -S^z`, an exact
symmetry of the XXZ Hamiltonian.  The empty diagonal is a selection rule.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DSSF = ROOT / "campaign" / "outputs" / "wp3_dssf"
FIGS = ROOT / "paper" / "figs"

INK = "#17191d"
GRID = "#e2e5e9"
ELECTRIC = "#a95a45"     # S^zz, the longitudinal probe
MAGNETIC = "#356399"     # ring exchange
DARK = "#b9bec6"         # reached by neither
FLOOR = 1e-30            # plotting floor for the machine-zero entries

plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "dejavuserif",
    "axes.labelsize": 8.6,
    "axes.titlesize": 8.4,
    "xtick.labelsize": 7.6,
    "ytick.labelsize": 7.4,
    "legend.fontsize": 7.0,
    "axes.linewidth": 0.7,
    "savefig.bbox": None,
})

COLUMNS = ("Gamma", "X", "L")
LABELS = {"Gamma": r"$\Gamma$", "X": "X", "L": "L"}


def load(coupling: float) -> dict:
    tag = f"{coupling:+.3f}".replace(".", "p")
    return json.loads((DSSF / f"flux_diagnostic_fcc32_{tag}.json").read_text())


def classify(level, threshold: float = 1e-12) -> str:
    electric = float(level["e_channel"])
    magnetic = float(level["b_channel"])
    if electric > threshold and magnetic > threshold:
        return "both"
    if electric > threshold:
        return "E"
    if magnetic > threshold:
        return "B"
    return "dark"


def level_panel(axis, record, title):
    levels = [row for row in record["levels"]
              if row["star"] in COLUMNS and row["excitation"] > 1e-9]
    strongest = max(
        max(float(row["e_channel"]), float(row["b_channel"]))
        for row in levels)
    for row in levels:
        column = COLUMNS.index(row["star"])
        kind = classify(row)
        colour = {"E": ELECTRIC, "B": MAGNETIC,
                  "both": "#6d4d86", "dark": DARK}[kind]
        weight = max(float(row["e_channel"]), float(row["b_channel"]))
        # width encodes how strongly the level is reached; a dark level still
        # gets a hairline so the reader can see it exists and is simply unlit
        span = 0.34 * (0.22 + 0.78 * np.sqrt(max(weight, 0.0) / strongest))
        if kind == "dark":
            span = 0.34 * 0.16
        axis.plot([column - span, column + span],
                  [row["excitation"]] * 2, "-", color=colour,
                  linewidth=2.6 if kind != "dark" else 1.1,
                  solid_capstyle="butt",
                  alpha=1.0 if kind != "dark" else 0.9, zorder=3)
    axis.set_xticks(range(len(COLUMNS)))
    axis.set_xticklabels([LABELS[c] for c in COLUMNS])
    axis.set_xlim(-0.55, len(COLUMNS) - 0.45)
    axis.set_ylim(bottom=0.0)
    axis.set_xlabel("momentum star")
    axis.set_title(title, loc="left", color=INK)
    axis.grid(True, axis="y", color=GRID, linewidth=0.6, zorder=0)
    axis.set_axisbelow(True)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)


def scatter_panel(axis, records):
    markers = {0.046: "o", -0.200: "s"}
    for coupling, record in records.items():
        for row in record["levels"]:
            if row["excitation"] <= 1e-9 or row["star"] not in COLUMNS:
                continue
            kind = classify(row)
            colour = {"E": ELECTRIC, "B": MAGNETIC,
                      "both": "#6d4d86", "dark": DARK}[kind]
            axis.plot(max(float(row["b_channel"]), FLOOR),
                      max(float(row["e_channel"]), FLOOR),
                      markers[coupling], color=colour, markersize=4.6,
                      markeredgecolor="white", markeredgewidth=0.5, zorder=3)
    # the region a level would occupy if it were lit in both channels
    axis.fill_between([1e-3, 1e2], [1e-3, 1e-3], [1e2, 1e2],
                      color="#d8cfc2", alpha=0.45, zorder=1, linewidth=0)
    axis.text(3e-1, 3e0, "both channels\n(empty)", fontsize=6.6,
              color="#7d6c58", ha="center", va="center", zorder=2)
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlim(FLOOR * 0.4, 3e2)
    axis.set_ylim(FLOOR * 0.4, 3e2)
    axis.set_xlabel(r"magnetic weight, $\Phi(\mathbf{q})$")
    axis.set_ylabel(r"electric weight, $S^z(\mathbf{q})$")
    axis.set_title("(c) every level pins to one axis", loc="left", color=INK)
    axis.grid(True, color=GRID, linewidth=0.6, zorder=0)
    axis.set_axisbelow(True)


def main() -> None:
    records = {0.046: load(+0.046), -0.200: load(-0.200)}
    figure, axes = plt.subplots(1, 3, figsize=(7.6, 3.35),
                                gridspec_kw={"wspace": 0.46,
                                             "width_ratios": [1, 1, 1.3]})
    level_panel(axes[0], records[0.046],
                r"(a) $J_\pm/J_{zz}=+0.046$")
    level_panel(axes[1], records[-0.200],
                r"(b) $J_\pm/J_{zz}=-0.200$")
    axes[0].set_ylabel(r"$\omega/J_{zz}$")
    scatter_panel(axes[2], records)

    handles = [
        Line2D([], [], color=ELECTRIC, linewidth=2.6,
               label=r"reached by $S^z$ (electric)"),
        Line2D([], [], color=MAGNETIC, linewidth=2.6,
               label=r"reached by $\Phi$ (magnetic)"),
        Line2D([], [], color=DARK, linewidth=1.1, label="reached by neither"),
    ]
    figure.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
                  bbox_to_anchor=(0.5, 0.005))
    figure.subplots_adjust(left=0.085, right=0.985, bottom=0.30, top=0.90)
    FIGS.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGS / "fig_fcc32_parity.pdf")
    figure.savefig(FIGS / "fig_fcc32_parity.png", dpi=220)
    plt.close(figure)
    print("wrote", FIGS / "fig_fcc32_parity.pdf")
    for coupling, record in records.items():
        counts = {}
        for row in record["levels"]:
            if row["excitation"] > 1e-9 and row["star"] in COLUMNS:
                counts[classify(row)] = counts.get(classify(row), 0) + 1
        print(f"  jpm={coupling:+.3f}: {counts}")


if __name__ == "__main__":
    main()
