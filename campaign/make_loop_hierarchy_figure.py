"""fig_loop_hierarchy.pdf: the four-panel operator-hierarchy figure.

Panels, in the order the argument runs in the manuscript:

  (a) FCC-32 loop taxonomy.  Mask-surviving simple-loop elements per
      perimeter, with the annihilated L=4 winding class shown at zero and the
      excluded vertex-joined composites shown separately.  This panel exists
      because the simple/composite separation is the single largest hazard in
      the measurement: 87% of connected L=12 elements are composites, and
      pooling them destroys the hierarchy.
  (b) The hierarchy itself, log K_L/K_6 against L at all seven couplings,
      with straight-line fits over L >= 8.
  (c) The fitted operator range xi_loop against coupling, with the ell=3
      convergence check overplotted as an open marker.
  (d) The order-counting test: successive ratios divided by |Jpm|/Jzz.  Naive
      degenerate perturbation theory makes these constant; they are not.

All inputs are already on disk; this script computes nothing new.  Sources:
  campaign/outputs/wp1_tomography/census_fcc32.json
  campaign/outputs/wp1_tomography/tomo_fcc32_L2_-0p*.json   (seven couplings)
  campaign/outputs/wp1_tomography/tomo_fcc32_L{2,3}_-0p200_n512.json
  campaign/outputs/wp1_tomography/tomo_fcc32_L{2,3}_-0p300_n512.json  (if present)

Panels (c) and (d) degrade gracefully: an ell=3 point that has not finished
yet is simply absent from the plot rather than fabricated.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TOMO = ROOT / "campaign" / "outputs" / "wp1_tomography"
FIGS = ROOT / "paper" / "figs"

INK = "#16181d"
INK_MUTED = "#5c6169"
GRID = "#e6e8ec"
CLEAN = "#356399"        # winding-free / FCC-32 primary
BARE = "#a95a45"         # contrast series
GOLD = "#c9a53a"
PERIMETERS = (6, 8, 10, 12)


def load_line():
    """The seven production couplings, ell=2, full column set."""
    records = []
    for path in sorted(TOMO.glob("tomo_fcc32_L2_-0p*.json")):
        if "_n" in path.stem:
            continue
        record = json.loads(path.read_text())
        if "k6" not in record:
            continue
        records.append(record)
    return sorted(records, key=lambda r: r["jpm"])


def load_sampled(levels, jpm):
    tag = f"{jpm:+.3f}".replace(".", "p")
    path = TOMO / f"tomo_fcc32_L{levels}_{tag}_n512.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def ratios(record):
    """K_L/K_6 for L = 8, 10, 12, or nan where the class is absent."""
    out = []
    for length in PERIMETERS[1:]:
        out.append(float(record.get(f"k{length}_over_k6", np.nan)))
    return np.array(out)


def main() -> int:
    records = load_line()
    if not records:
        print("no tomography records found; nothing to plot")
        return 1
    couplings = np.array([r["jpm"] for r in records])

    figure, axes = plt.subplots(2, 2, figsize=(7.0, 5.6))
    figure.subplots_adjust(hspace=0.34, wspace=0.30,
                           left=0.098, right=0.982, top=0.955, bottom=0.095)
    for axis in axes.ravel():
        axis.grid(True, color=GRID, linewidth=0.6, zorder=0)
        axis.set_axisbelow(True)
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)

    # ---------------------------------------------------- (a) taxonomy
    axis = axes[0, 0]
    census_path = TOMO / "census_fcc32.json"
    if census_path.exists():
        census = json.loads(census_path.read_text())["per_perimeter"]
        lengths = sorted(int(k) for k in census)
        kept = [census[str(L)]["kept"] for L in lengths]
        pairs = [census[str(L)]["pairs"] for L in lengths]
        excluded = [p - k for p, k in zip(pairs, kept)]
        index = np.arange(len(lengths))
        axis.bar(index - 0.19, kept, width=0.36, color=CLEAN,
                 label="simple, mask-surviving", zorder=3)
        axis.bar(index + 0.19, excluded, width=0.36, color=INK_MUTED,
                 alpha=0.55, label="composite or masked away", zorder=3)
        axis.set_xticks(index)
        axis.set_xticklabels([f"$L={L}$" for L in lengths])
        axis.set_yscale("symlog", linthresh=1e3)
        axis.set_ylabel("matrix elements")
        axis.legend(frameon=False, fontsize=6.4, loc="upper left")
        axis.annotate("winding class\nannihilated", xy=(0, 1), xytext=(0.05, 0.42),
                      textcoords="axes fraction", fontsize=6.2, color=INK_MUTED,
                      ha="left")
    axis.set_title("(a) FCC-32 loop taxonomy", fontsize=8, color=INK, loc="left")

    # ---------------------------------------------------- (b) hierarchy
    axis = axes[0, 1]
    colours = plt.cm.viridis(np.linspace(0.08, 0.86, len(records)))
    for record, colour in zip(records, colours):
        values = np.concatenate([[1.0], ratios(record)])
        axis.plot(PERIMETERS, values, "o-", color=colour, markersize=3.0,
                  linewidth=1.0, zorder=3,
                  label=f"${record['jpm']:+.2f}$")
        good = np.isfinite(values) & (values > 0)
        tail = np.array(PERIMETERS)[good & (np.array(PERIMETERS) >= 8)]
        if len(tail) >= 2:
            tail_values = values[good & (np.array(PERIMETERS) >= 8)]
            slope, intercept = np.polyfit(tail, np.log(tail_values), 1)
            span = np.linspace(8, 12, 10)
            axis.plot(span, np.exp(slope * span + intercept), "--",
                      color=colour, linewidth=0.7, alpha=0.7, zorder=2)
    axis.set_yscale("log")
    axis.set_xticks(PERIMETERS)
    axis.set_xlabel("loop perimeter $L$")
    axis.set_ylabel(r"$K_L/K_6$")
    axis.legend(frameon=False, fontsize=5.6, ncol=2, title=r"$J_\pm/J_{zz}$",
                title_fontsize=5.8, loc="lower left")
    axis.set_title("(b) simple-loop hierarchy", fontsize=8, color=INK, loc="left")

    # ---------------------------------------------------- (c) operator range
    axis = axes[1, 0]
    xi = np.array([float(r.get("xi_gauge", np.nan)) for r in records])
    axis.plot(np.abs(couplings), xi, "o-", color=CLEAN, markersize=3.6,
              linewidth=1.1, zorder=3, label=r"$\ell=2$, full columns")
    for jpm in (-0.20, -0.30):
        deep = load_sampled(3, jpm)
        shallow = load_sampled(2, jpm)
        if deep is None:
            continue
        axis.plot([abs(jpm)], [float(deep["xi_gauge"])], "o",
                  markerfacecolor="none", markeredgecolor=INK,
                  markersize=7.0, markeredgewidth=1.1, zorder=4,
                  label=r"$\ell=3$ check" if jpm == -0.20 else None)
        if shallow is not None:
            axis.plot([abs(jpm)] * 2,
                      [float(shallow["xi_gauge"]), float(deep["xi_gauge"])],
                      "-", color=INK, linewidth=0.8, zorder=4)
    axis.set_xlabel(r"$|J_\pm|/J_{zz}$")
    axis.set_ylabel(r"$\xi_{\rm loop}$")
    axis.legend(frameon=False, fontsize=6.4, loc="lower right")
    axis.set_title("(c) operator range", fontsize=8, color=INK, loc="left")

    # ---------------------------------------------------- (d) order counting
    axis = axes[1, 1]
    magnitude = np.abs(couplings)
    r8, r10 = [], []
    for record in records:
        k8, k10, k12 = ratios(record)
        r8.append(k10 / k8 if k8 else np.nan)
        r10.append(k12 / k10 if k10 else np.nan)
    r8 = np.array(r8) / magnitude
    r10 = np.array(r10) / magnitude
    axis.plot(magnitude, r8, "o-", color=CLEAN, markersize=3.6, linewidth=1.1,
              zorder=3, label=r"$K_{10}/K_8$")
    axis.plot(magnitude, r10, "s-", color=GOLD, markersize=3.4, linewidth=1.1,
              zorder=3, label=r"$K_{12}/K_{10}$")
    axis.axhline(r8[0], color=BARE, linestyle=":", linewidth=1.0, zorder=2)
    axis.annotate("order counting:\nconstant", xy=(magnitude[-1], r8[0]),
                  xytext=(0.42, 0.80), textcoords="axes fraction",
                  fontsize=6.2, color=BARE)
    axis.set_xlabel(r"$|J_\pm|/J_{zz}$")
    axis.set_ylabel(r"$(K_{L+2}/K_L)\,/\,(|J_\pm|/J_{zz})$")
    axis.legend(frameon=False, fontsize=6.4, loc="lower left")
    axis.set_title("(d) departure from order counting", fontsize=8,
                   color=INK, loc="left")

    FIGS.mkdir(parents=True, exist_ok=True)
    destination = FIGS / "fig_loop_hierarchy.pdf"
    figure.savefig(destination)
    figure.savefig(destination.with_suffix(".png"), dpi=220)
    print("wrote", destination)
    print(f"  couplings plotted: {[float(c) for c in couplings]}")
    for jpm in (-0.20, -0.30):
        deep = load_sampled(3, jpm)
        print(f"  ell=3 check at {jpm:+.2f}: "
              f"{'present' if deep else 'MISSING (panel c degrades)'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
