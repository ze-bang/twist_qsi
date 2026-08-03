#!/usr/bin/env python3
"""Figure for the two-sided fold: curves, definition spread, Z-rank gap.

Panels
  (a) C(T) at the weakest computed coupling: the fold must sit on top of the
      splice and H+C.  The naive weighted union is drawn too -- its corrupted
      T -> 0 limit (raw band remnants hijack the ensemble) is the documented
      failure mode that forced the Z-ranked construction.
  (b) C(T) at the strongest coupling with a complete band: the definitions
      now disagree -- that disagreement is the measured ambiguity.
  (c) low-peak position over g6 versus |Jpm| for every definition, with the
      three-way spread shaded.
  (d) the classification-sharpness diagnostic: ice weight of the last level
      the fold removes versus the first it keeps.  The gap is bimodal-clean
      at weak coupling; its closure is dissolution, seen in the data.

Reads campaign/outputs/two_sided_fold/fold_*.json|npz written by
run_two_sided_fold.py.  Writes fig_two_sided_fold.(pdf|png) next to them.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "campaign" / "outputs" / "two_sided_fold"

# CVD-validated categorical assignment (fixed order, entity-locked)
COLOR = {"raw": "#cc79a7", "hc": "#e69f00", "splice": "#0072b2",
         "fold": "#d55e00", "fold_naive": "0.55"}
LABEL = {"raw": "raw ED", "hc": "counterterm $H+C$", "splice": "splice",
         "fold": "two-sided fold", "fold_naive": "naive weighted union (fails)"}
STYLE = {"raw": {"ls": ":", "lw": 1.4}, "hc": {"ls": "-", "lw": 1.4},
         "splice": {"ls": "--", "lw": 1.4}, "fold": {"ls": "-", "lw": 2.2},
         "fold_naive": {"ls": (0, (1, 1)), "lw": 1.2}}


def load_all():
    records = []
    for path in sorted(DATA.glob("fold_*.json")):
        record = json.loads(path.read_text())
        record["npz"] = np.load(DATA / (path.stem + ".npz"))
        records.append(record)
    return sorted(records, key=lambda r: abs(r["jpm"]))


def curve_panel(ax, record, title, series):
    npz = record["npz"]
    grid = npz["grid"]
    g6 = record["g6"]
    for name in series:
        key = f"curve_{name}"
        if key not in npz:
            continue
        ax.plot(grid / g6, npz[key], color=COLOR[name], label=LABEL[name],
                **STYLE[name])
    ax.axvline(1.0, color="0.75", lw=0.8, zorder=0)
    ax.set_xscale("log")
    ax.set_xlabel(r"$T/g_6$   (grey line: $T=g_6$)")
    ax.set_ylabel(r"$C$ per site")
    ax.set_title(title, fontsize=9.5)
    ax.grid(True, which="major", color="0.92", lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=7.5, loc="upper left")


def main() -> None:
    records = load_all()
    if not records:
        raise SystemExit("no fold_*.json found; run run_two_sided_fold.py first")

    weak, strong = records[0], records[-1]
    for record in records:
        if record["n_band_lost"] == 0:
            strong = record  # strongest coupling with a complete band

    fig, axes = plt.subplots(2, 2, figsize=(9.6, 7.2))
    fig.subplots_adjust(hspace=0.42, wspace=0.30)

    curve_panel(axes[0, 0], weak,
                f"(a) validation at $J_\\pm={weak['jpm']:+.2f}$",
                ("raw", "hc", "splice", "fold", "fold_naive"))
    curve_panel(axes[0, 1], strong,
                f"(b) dissolution at $J_\\pm={strong['jpm']:+.2f}$",
                ("raw", "hc", "splice", "fold"))

    # (c) low peak / g6 across definitions
    ax = axes[1, 0]
    jpms = np.array([abs(r["jpm"]) for r in records])
    for name in ("raw", "hc", "splice", "fold"):
        values = np.array([r.get(f"{name}_low_over_g6", np.nan)
                           for r in records], float)
        ok = np.isfinite(values)
        ax.plot(jpms[ok], values[ok], marker="o", ms=4, color=COLOR[name],
                lw=STYLE[name]["lw"], ls=STYLE[name]["ls"],
                label=LABEL[name])
    banded = [(abs(r["jpm"]),
               min(v for k in ("hc", "splice", "fold")
                   if np.isfinite(v := r.get(f"{k}_low_over_g6", np.nan))),
               max(v for k in ("hc", "splice", "fold")
                   if np.isfinite(v := r.get(f"{k}_low_over_g6", np.nan))))
              for r in records]
    bx, blo, bhi = map(np.array, zip(*banded))
    ax.fill_between(bx, blo, bhi, color="0.85", alpha=0.7, zorder=0,
                    label="definition spread")
    ax.axhline(1.0, color="0.75", lw=0.8, zorder=0)
    ax.set_xlabel(r"$|J_\pm|$")
    ax.set_ylabel(r"low peak $/\,g_6$")
    ax.set_title("(c) ring-scale peak across definitions", fontsize=9.5)
    ax.grid(True, color="0.92", lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=7.5)

    # (d) classification sharpness: the Z-rank gap
    ax = axes[1, 1]
    z_removed = np.array([r["z_last_removed"] for r in records])
    z_kept = np.array([r["z_first_kept"] for r in records])
    ax.plot(jpms, z_removed, "o-", ms=4, color=COLOR["fold"], lw=1.8,
            label="last level removed (lowest-$Z$ band state)")
    ax.plot(jpms, z_kept, "s-", ms=4, color=COLOR["splice"], lw=1.8,
            label="first level kept (highest-$Z$ continuum state)")
    ax.fill_between(jpms, z_kept, z_removed, color=COLOR["fold"], alpha=0.12)
    ax.text(0.97, 0.45,
            "$Z$-rank gap: bimodal $\\Rightarrow$ classification sharp\n"
            "closed on a degenerate multiplet $\\Rightarrow$ harmless\n"
            "(cut energy split $= 0$ at every closed point)",
            fontsize=7.5, color="0.3", ha="right", va="top",
            transform=ax.transAxes)
    ax.set_xlabel(r"$|J_\pm|$")
    ax.set_ylabel(r"ice weight $Z$ at the cut")
    ax.set_ylim(0, 1)
    ax.set_title("(d) how sharp is the band/continuum cut?", fontsize=9.5)
    ax.grid(True, color="0.92", lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=7.5, loc="upper right")

    for stem in ("fig_two_sided_fold.pdf", "fig_two_sided_fold.png"):
        fig.savefig(DATA / stem, dpi=220, bbox_inches="tight")
    print("wrote", DATA / "fig_two_sided_fold.pdf")


if __name__ == "__main__":
    main()
