#!/usr/bin/env python3
"""Specific heat across the coupling range: the two anomalies, and their scales.

Reads the records written by run_specific_heat_scan.py and produces

  fig_specific_heat.pdf
    (a) C(T)/N for a selection of couplings, raw versus winding-free;
    (b) the lower (gauge) peak position against |J_pm|, on log-log, with the
        two candidate scales g_4 = 4 J_pm^2 and g_6 = 12 |J_pm|^3 drawn in;
    (c) the same data as ratios T_low/g_4 and T_low/g_6 -- a horizontal line
        is the statement "this peak tracks that scale";
    (d) the upper (spinon) anomaly, for completeness.

The gauge anomaly is identified as the TALLEST local maximum below the spinon
anomaly, which is robust against the small extra low-temperature structure the
zero-flux side develops.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "campaign" / "outputs" / "specific_heat"
FIGS = ROOT / "paper" / "figs"
FIGS.mkdir(parents=True, exist_ok=True)

BLUE, VERM, GREEN, ORANGE, PINK = ("#0072B2", "#D55E00", "#009E73",
                                   "#E69F00", "#CC79A7")
GREY = "#5c5c5c"

plt.rcParams.update({
    "font.size": 8.5, "axes.labelsize": 8.5, "axes.titlesize": 9,
    "legend.fontsize": 7.2, "xtick.labelsize": 7.6, "ytick.labelsize": 7.6,
    "axes.linewidth": 0.7, "xtick.direction": "in", "ytick.direction": "in",
    "lines.linewidth": 1.4, "lines.markersize": 4.0,
    "axes.grid": True, "grid.alpha": 0.22, "grid.linewidth": 0.5,
    "figure.dpi": 160, "savefig.bbox": "tight",
})


DIP = 0.70   # a valley between two maxima must fall to this fraction of the
             # smaller peak height for them to count as separately resolved


def branch_track(records, key):
    """Assign maxima to a gauge branch and a spinon branch by continuity.

    Peaks are followed from weak coupling, where the two anomalies are far
    apart and unambiguous, to strong coupling.  At each step every maximum is
    attached to whichever branch it is closest to in log T.  When the two
    anomalies merge, the single surviving maximum is attached to the branch it
    continues, and the other branch is recorded as absorbed -- rather than
    being silently relabelled, which is what produces a spurious jump.
    """
    gauge = spinon = None
    history = []          # (|jpm|, T) of the gauge branch, for extrapolation
    lost = False          # a branch that has been absorbed does not come back
    for record in sorted(records, key=lambda r: abs(r["jpm"])):
        peaks = record[f"{key}_peaks"]
        if not peaks:
            record[f"{key}_gauge"] = record[f"{key}_spinon"] = None
            record[f"{key}_absorbed"] = True
            continue
        if gauge is None:                       # weakest coupling: seed
            gauge, spinon = min(peaks, key=lambda q: q[0]), max(peaks, key=lambda q: q[0])
        else:
            def nearest(target):
                if target is None:
                    return None
                return min(peaks, key=lambda q: abs(np.log(q[0] / target[0])))
            g, sp = nearest(gauge), nearest(spinon)
            if g is not None and sp is not None and g is sp:
                # the two anomalies have merged: keep the branch it continues
                if abs(np.log(g[0] / gauge[0])) < abs(np.log(g[0] / spinon[0])):
                    gauge, spinon = g, None
                else:
                    gauge, spinon = None, g
            else:
                gauge, spinon = g, sp
        # continuity test on the gauge branch: extrapolate log T linearly in
        # log|Jpm| from the last two points.  A maximum that lands far off that
        # line is a different feature -- the one we were following has ceased
        # to be a local maximum -- so the branch is declared lost rather than
        # silently jumping.
        if gauge is not None and len(history) >= 2:
            (x1, t1), (x2, t2) = history[-2], history[-1]
            slope = np.log(t2 / t1) / np.log(x2 / x1)
            predicted = t2 * (abs(record["jpm"]) / x2) ** slope
            if abs(np.log(gauge[0] / predicted)) > np.log(1.4):
                gauge, lost = None, True
        if lost:
            gauge = None
        record[f"{key}_gauge"], record[f"{key}_spinon"] = gauge, spinon
        record[f"{key}_absorbed"] = (gauge is None) or (spinon is None)
        if gauge is not None:
            history.append((abs(record["jpm"]), gauge[0]))
    return records


def resolved(record, key, grid_cache):
    """Do the two anomalies still separate, by the dip criterion?"""
    gauge, spinon = record[f"{key}_gauge"], record[f"{key}_spinon"]
    if gauge is None or spinon is None:
        return False, np.nan
    grid, curve = grid_cache
    lo = int(np.argmin(np.abs(grid - gauge[0])))
    hi = int(np.argmin(np.abs(grid - spinon[0])))
    if hi <= lo:
        return False, np.nan
    ratio = float(curve[lo:hi].min()) / min(gauge[1], spinon[1])
    return ratio < DIP, ratio


def load():
    records = []
    for path in sorted(OUT.glob("heat_*.json")):
        with open(path) as handle:
            records.append(json.load(handle))
    negative = [r for r in records if r["jpm"] < 0]
    positive = [r for r in records if r["jpm"] > 0]
    for group in (negative, positive):
        for key in ("raw", "wf"):
            branch_track(group, key)
    for record in records:
        tag = f"{record['jpm']:+.3f}".replace(".", "p")
        data = np.load(OUT / f"heat_{tag}.npz")
        for key in ("raw", "wf"):
            record[f"{key}_resolved"], record[f"{key}_dip"] = resolved(
                record, key, (data["grid"], data[f"{key}_curve"]))
    records.sort(key=lambda r: r["jpm"])
    return records


def power_law(x, y):
    """Fit y = a x^n, return n."""
    mask = (x > 0) & (y > 0)
    if mask.sum() < 3:
        return np.nan
    return float(np.polyfit(np.log(x[mask]), np.log(y[mask]), 1)[0])


def masked_band():
    """Ring peak from the self-consistent masked Feshbach roots (gauge sector
    only: this route has no spinon states and hence no upper anomaly)."""
    out = []
    for path in sorted((ROOT / "campaign" / "outputs"
                        / "feshbach_anatomy").glob("anatomy_*.json")):
        with open(path) as handle:
            record = json.load(handle)
        if record["jpm"] < 0 and record.get("ring_peak"):
            out.append((abs(record["jpm"]), record["ring_peak"],
                        record["n_masked_roots"]))
    return sorted(out)


def main():
    records = load()
    masked = masked_band()
    negative = [r for r in records if r["jpm"] < 0]
    positive = [r for r in records if r["jpm"] > 0]

    figure, axes = plt.subplots(1, 4, figsize=(13.6, 3.1))

    # ---- (a) a few curves
    ax = axes[0]
    show = [-0.05, -0.15, -0.30]
    colours = [BLUE, VERM, GREEN]
    for target, colour in zip(show, colours):
        record = min(records, key=lambda r: abs(r["jpm"] - target))
        tag = f"{record['jpm']:+.3f}".replace(".", "p")
        data = np.load(OUT / f"heat_{tag}.npz")
        ax.semilogx(data["grid"], data["raw_curve"], "--", color=colour,
                    alpha=0.55, lw=1.1)
        ax.semilogx(data["grid"], data["wf_curve"], "-", color=colour,
                    label=rf"$J_\pm={record['jpm']:+.2f}$")
    ax.set_xlim(1e-5, 3)
    ax.set_xlabel(r"$T/J_{zz}$")
    ax.set_ylabel(r"$C/N$")
    ax.set_title("(a) heat capacity: dashed raw, solid winding-free",
                 loc="left", pad=4)
    ax.legend(frameon=False, loc="upper left")

    # ---- (b) lower peak against the two scales
    ax = axes[1]
    x = np.asarray([abs(r["jpm"]) for r in negative])
    def series(records, key, want_resolved):
        return np.asarray([
            r[f"{key}_gauge"][0] if (r[f"{key}_gauge"]
                                     and r[f"{key}_resolved"] == want_resolved)
            else np.nan for r in records])
    raw, wf = series(negative, "raw", True), series(negative, "wf", True)
    raw_merged = series(negative, "raw", False)
    wf_merged = series(negative, "wf", False)
    grid = np.geomspace(x.min(), x.max(), 100)
    ax.loglog(grid, 4 * grid**2, ":", color=GREY, lw=1.2,
              label=r"$g_4=4J_\pm^2$")
    ax.loglog(grid, 12 * grid**3, "--", color=GREY, lw=1.2,
              label=r"$g_6=12|J_\pm|^3$")
    ax.loglog(x, raw, "o-", color=VERM, label="raw periodic ED")
    ax.loglog(x, wf, "s-", color=BLUE, label="winding-free $H+C$")
    if masked:
        mx = np.asarray([m[0] for m in masked])
        my = np.asarray([m[1] for m in masked])
        full = np.asarray([m[2] == 90 for m in masked])
        ax.loglog(mx[full], my[full], "^", color=GREEN,
                  label="masked Feshbach band")
        ax.loglog(mx[~full], my[~full], "^", color=GREEN, mfc="none")
    ax.loglog(x, wf_merged, "s--", color=BLUE, mfc="none", alpha=0.8,
              label="single merged anomaly")
    ax.loglog(x, raw_merged, "o--", color=VERM, mfc="none", alpha=0.8)
    ax.set_xlabel(r"$|J_\pm|/J_{zz}$")
    ax.set_ylabel(r"$T_{\rm peak}/J_{zz}$")
    ax.set_title("(b) the lower anomaly and the two candidate scales",
                 loc="left", pad=4)
    ax.legend(frameon=False, loc="lower right")

    # ---- (c) ratios
    ax = axes[2]
    ax.semilogx(x, raw / (4 * x**2), "o-", color=VERM,
                label=r"raw $/\,g_4$")
    ax.semilogx(x, wf / (12 * x**3), "s-", color=BLUE,
                label=r"winding-free $/\,g_6$")
    if masked:
        ax.semilogx(mx, my / (12 * mx**3), "^", color=GREEN,
                    label=r"masked band $/\,g_6$")
    ax.semilogx(x, raw / (12 * x**3), "^--", color=ORANGE, alpha=0.7,
                label=r"raw $/\,g_6$")
    ax.set_yscale("log")
    ax.axhline(1.0, color=GREY, lw=0.9)
    ax.set_xlabel(r"$|J_\pm|/J_{zz}$")
    ax.set_ylabel("peak / scale")
    ax.set_title("(c) which scale each peak follows", loc="left", pad=4)
    ax.legend(frameon=False, loc="upper left")

    # ---- (d) upper anomaly
    ax = axes[3]
    hi_raw = np.asarray([r["raw_spinon"][0] if (r["raw_spinon"]
                         and r["raw_resolved"]) else np.nan for r in negative])
    hi_raw_m = np.asarray([r["raw_spinon"][0] if (r["raw_spinon"]
                           and not r["raw_resolved"]) else np.nan
                           for r in negative])
    hi_wf = np.asarray([r["wf_spinon"][0] if r["wf_spinon"] else np.nan
                        for r in negative])
    ax.plot(x, hi_raw, "o-", color=VERM, label="raw, two anomalies")
    ax.plot(x, hi_raw_m, "o", color=VERM, mfc="none",
            label="raw, single merged anomaly")
    ax.plot(x, hi_wf, "s--", color=BLUE, label="winding-free")
    ax.set_xlabel(r"$|J_\pm|/J_{zz}$")
    ax.set_ylabel(r"$T_{\rm peak}/J_{zz}$")
    ax.set_title("(d) the upper (spinon) anomaly", loc="left", pad=4)
    ax.legend(frameon=False)

    figure.tight_layout()
    figure.savefig(FIGS / "fig_specific_heat.pdf")
    plt.close(figure)

    # ---- numbers
    print(f"{'jpm':>7} {'g4':>9} {'g6':>10} | {'raw low':>9} {'/g4':>6} "
          f"{'/g6':>8} | {'wf low':>9} {'/g6':>6} {'/g4':>7} | "
          f"{'raw hi':>7} {'wf hi':>7} {'dipR':>6} {'dipW':>5} {'!':>3}")
    for r in records:
        rg = r["raw_gauge"][0] if r["raw_gauge"] else np.nan
        wg = r["wf_gauge"][0] if r["wf_gauge"] else np.nan
        flag = (("" if r["raw_resolved"] else "R")
                + ("" if r["wf_resolved"] else "W"))
        rh = r["raw_spinon"][0] if r["raw_spinon"] else np.nan
        wh = r["wf_spinon"][0] if r["wf_spinon"] else np.nan
        print(f"{r['jpm']:>7.3f} {r['g4']:>9.5f} {r['g6']:>10.6f} | "
              f"{rg:>9.5f} {rg/r['g4']:>6.3f} {rg/r['g6']:>8.2f} | "
              f"{wg:>9.5f} {wg/r['g6']:>6.3f} {wg/r['g4']:>7.4f} | "
              f"{rh:>7.4f} {wh:>7.4f} {r['raw_dip']:>6.2f} "
              f"{r['wf_dip']:>5.2f} {flag:>3}")

    for cut in (0.09, 0.05, 0.03):
        w = [r for r in negative if abs(r["jpm"]) <= cut and r["raw_resolved"]]
        xs2 = np.asarray([abs(r["jpm"]) for r in w])
        print("  |Jpm| <= %.2f : raw ~ |Jpm|^%.2f , wf ~ |Jpm|^%.2f" % (
            cut,
            power_law(xs2, np.asarray([r["raw_gauge"][0] for r in w])),
            power_law(xs2, np.asarray([r["wf_gauge"][0] for r in w]))))
    window = [r for r in negative if abs(r["jpm"]) <= 0.09]
    xs = np.asarray([abs(r["jpm"]) for r in window])
    print()
    print("power-law fits over |Jpm| <= 0.09:")
    print("  raw  lower peak ~ |Jpm|^%.2f  (g4 would give 2)" % power_law(
        xs, np.asarray([r["raw_gauge"][0] for r in window])))
    print("  wf   lower peak ~ |Jpm|^%.2f  (g6 would give 3)" % power_law(
        xs, np.asarray([r["wf_gauge"][0] for r in window])))
    print("figure written to", FIGS / "fig_specific_heat.pdf")


if __name__ == "__main__":
    main()
