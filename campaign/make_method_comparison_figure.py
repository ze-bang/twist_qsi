#!/usr/bin/env python3
"""The four methods side by side, with every anomaly labelled.

  fig_method_comparison.pdf
    (a) C(T)/N at three couplings, all four methods;
    (b) every labelled anomaly against |J_pm|, with g_4 and g_6 drawn in;
    (c) the ring anomaly as a ratio to each candidate scale;
    (d) the diagnostics that say which method to believe where.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "campaign" / "outputs" / "method_comparison"
FIGS = ROOT / "paper" / "figs"
FIGS.mkdir(parents=True, exist_ok=True)

BLUE, VERM, GREEN, ORANGE, PINK = ("#0072B2", "#D55E00", "#009E73",
                                   "#E69F00", "#CC79A7")
GREY = "#5c5c5c"
COLOUR = {"periodic": VERM, "polar": ORANGE, "feshbach": BLUE, "hc": GREEN}
MARKER = {"ring": "o", "gauge": "o", "spinon": "s", "sector": "^"}
# the raw cluster has no transport sectors, so its low anomaly is called
# "gauge" rather than "ring"; both are the same series in the plots
RINGLIKE = {"periodic": "gauge", "polar": "ring", "feshbach": "ring",
            "hc": "ring"}
NAME = {"periodic": "periodic ED (baseline)", "polar": "polar pullback",
        "feshbach": "masked Feshbach", "hc": r"$H+C$"}

plt.rcParams.update({
    "font.size": 8.5, "axes.labelsize": 8.5, "axes.titlesize": 9,
    "legend.fontsize": 6.8, "xtick.labelsize": 7.6, "ytick.labelsize": 7.6,
    "axes.linewidth": 0.7, "xtick.direction": "in", "ytick.direction": "in",
    "lines.linewidth": 1.4, "lines.markersize": 4.2,
    "axes.grid": True, "grid.alpha": 0.22, "grid.linewidth": 0.5,
    "figure.dpi": 160, "savefig.bbox": "tight",
})


def load():
    with open(OUT / "peak_labels.json") as handle:
        return json.load(handle)


def series(records, method, label, negative=True):
    x, y = [], []
    for record in records:
        if (record["jpm"] < 0) != negative:
            continue
        for peak in record.get(method, []):
            if peak["label"] == label and not peak["at_grid_edge"]:
                x.append(abs(record["jpm"]))
                y.append(peak["T"])
    order = np.argsort(x)
    return np.asarray(x)[order], np.asarray(y)[order]


def main():
    records = load()

    figure, axes = plt.subplots(1, 4, figsize=(14.2, 3.2))

    # ---- (a) curves
    ax = axes[0]
    for target, style in zip((-0.05, -0.15, -0.30), ("-", "--", ":")):
        record = min(records, key=lambda r: abs(r["jpm"] - target))
        tag = f"{record['jpm']:+.3f}".replace(".", "p")
        path = OUT / f"compare_{tag}.npz"
        if not path.exists():
            continue
        data = np.load(path)
        for method in ("periodic", "feshbach", "hc"):
            key = f"{method}_curve"
            if key in data:
                ax.semilogx(data["grid"], data[key], style,
                            color=COLOUR[method], lw=1.2,
                            label=(NAME[method]
                                   if target == -0.05 else None))
    ax.set_xlim(1e-5, 3)
    ax.set_xlabel(r"$T/J_{zz}$")
    ax.set_ylabel(r"$C/N$")
    ax.set_title(r"(a) $C(T)$ at $-0.05$, $-0.15$, $-0.30$", loc="left", pad=4)
    ax.legend(frameon=False, loc="upper left")

    # ---- (b) every labelled anomaly
    ax = axes[1]
    grid = np.geomspace(0.008, 0.32, 100)
    ax.loglog(grid, 4 * grid**2, ":", color=GREY, lw=1.3,
              label=r"$g_4=4J_\pm^2$")
    ax.loglog(grid, 12 * grid**3, "--", color=GREY, lw=1.3,
              label=r"$g_6=12|J_\pm|^3$")
    for method in ("periodic", "polar", "feshbach", "hc"):
        for label in (RINGLIKE[method], "spinon", "sector"):
            x, y = series(records, method, label)
            if not len(x):
                continue
            ax.loglog(x, y, MARKER[label] + "-", color=COLOUR[method],
                      alpha=0.9 if label == RINGLIKE[method] else 0.45,
                      mfc="none" if label != RINGLIKE[method] else None,
                      label=(NAME[method] if label == RINGLIKE[method]
                             else None))
    ax.set_xlabel(r"$|J_\pm|/J_{zz}$")
    ax.set_ylabel(r"$T_{\rm peak}/J_{zz}$")
    ax.set_title("(b) every anomaly: o ring, s spinon, ^ sector",
                 loc="left", pad=4)
    ax.legend(frameon=False, loc="lower right")

    # ---- (c) ring anomaly against both scales
    ax = axes[2]
    for method in ("periodic", "polar", "feshbach", "hc"):
        x, y = series(records, method, RINGLIKE[method])
        if not len(x):
            continue
        ax.semilogx(x, y / (12 * x**3), "o-", color=COLOUR[method],
                    label=NAME[method])
    x, y = series(records, "periodic", "gauge")
    ax.semilogx(x, y / (4 * x**2), "s--", color=VERM, alpha=0.5,
                label=r"periodic $/\,g_4$")
    ax.axhline(1.0, color=GREY, lw=0.9)
    ax.set_yscale("log")
    ax.set_xlabel(r"$|J_\pm|/J_{zz}$")
    ax.set_ylabel(r"ring peak $/\,g_6$  (and $/\,g_4$)")
    ax.set_title("(c) ring anomaly / scale", loc="left", pad=4)
    ax.legend(frameon=False, loc="lower left")

    # ---- (d) diagnostics
    ax = axes[3]
    x = np.asarray([abs(r["jpm"]) for r in records if r["jpm"] < 0])
    meta = [r["meta"] for r in records if r["jpm"] < 0]
    order = np.argsort(x)
    x = x[order]
    meta = [meta[i] for i in order]
    ax.plot(x, [m.get("gram_min", np.nan) for m in meta], "o-", color=ORANGE,
            label="polar: Gram minimum")
    ax.plot(x, [m.get("gap_above_band", np.nan) for m in meta], "s--",
            color=ORANGE, alpha=0.6, label="polar: gap above band")
    ax.plot(x, [m.get("n_roots", np.nan) / 90 for m in meta], "^-", color=BLUE,
            label="Feshbach: roots / 90")
    ax.plot(x, [m.get("hc_residue_over_width", np.nan) for m in meta], "d-",
            color=GREEN, label=r"$H+C$: residue / band width")
    ax.plot(x, [m.get("z_mean", np.nan) for m in meta], ":", color=GREY,
            label=r"residue $Z$")
    ax.set_xlabel(r"$|J_\pm|/J_{zz}$")
    ax.set_ylabel("diagnostic")
    ax.set_ylim(-0.05, 1.15)
    ax.set_title("(d) method diagnostics", loc="left", pad=4)
    ax.legend(frameon=False, loc="lower left")

    figure.tight_layout()
    figure.savefig(FIGS / "fig_method_comparison.pdf")
    plt.close(figure)
    print("figure written to", FIGS / "fig_method_comparison.pdf")

    # ---- the table
    print()
    print("ring anomaly, all methods (T_peak / g6):")
    header = f"{'jpm':>7} {'g4':>9} {'g6':>10}"
    for method in ("periodic", "polar", "feshbach", "hc"):
        header += f" {method:>10}"
    print(header + f" {'roots':>6} {'gram':>6}")
    for record in sorted(records, key=lambda r: r["jpm"]):
        if record["jpm"] >= 0:
            continue
        row = f"{record['jpm']:>7.3f} {record['g4']:>9.5f} {record['g6']:>10.6f}"
        for method in ("periodic", "polar", "feshbach", "hc"):
            ring = [p for p in record.get(method, [])
                    if p["label"] == RINGLIKE[method]]
            row += (f" {ring[0]['T']/record['g6']:>10.4f}" if ring
                    else f" {'-':>10}")
        meta = record["meta"]
        row += f" {meta.get('n_roots', 0):>6d} {meta.get('gram_min', float('nan')):>6.3f}"
        print(row)


if __name__ == "__main__":
    main()
