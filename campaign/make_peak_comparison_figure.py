#!/usr/bin/env python3
"""fig_peak_comparison.pdf: the low-temperature C(T) maximum, all four methods.

Every series is recomputed here on one footing rather than collected from the
per-method runs, because the earlier numbers were located on a 3000-point
logarithmic grid whose spacing (0.585% per step) is coarser than the differences
being compared.  Peaks are found on a 12000-point grid and refined by a parabola
in log T, and labelled by the same variance-share classifier throughout.

Series, in the order the constructions were tried:

    periodic ED        the baseline; tracks the finite-torus scale g_4
    polar pullback     the original winding-free route, needs an isolated band
    masked Feshbach    the frame-free reference (exact, but band only)
    counterterm        the self-consistent operator (this work)

Panel (a) is absolute: T_peak against |J_pm| with both candidate scales drawn
in.  Panel (b) divides by g_6, which is the sharp test -- a horizontal line
means the anomaly follows the ring-exchange scale, and periodic ED instead
climbs as g_4/g_6 = J_zz/3|J_pm|.

Usage: make_peak_comparison_figure.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "campaign"))
from refine_peaks import refined_maxima, classify, named  # noqa: E402

COMPARE = ROOT / "campaign" / "outputs" / "method_comparison"
SPINE = ROOT / "campaign" / "outputs" / "counterterm_spine"
COMPLETE = ROOT / "campaign" / "outputs" / "masked_band_complete"
FIGS = ROOT / "paper" / "figs"
FIGS.mkdir(parents=True, exist_ok=True)

# Okabe-Ito, assigned in the order the constructions were introduced and never
# recycled: a series keeps its hue whether or not the others are present.
STYLE = {
    "periodic":    ("#D55E00", "s", "periodic ED"),
    "polar":       ("#E69F00", "^", "polar pullback"),
    "feshbach":    ("#0072B2", "D", "masked Feshbach (reference)"),
    "counterterm": ("#009E73", "o", "self-consistent counterterm"),
}
ORDER = ("periodic", "polar", "feshbach", "counterterm")
GREY = "#5C5C5C"

plt.rcParams.update({
    "font.size": 8.5, "axes.labelsize": 9, "axes.titlesize": 9,
    "legend.fontsize": 7.4, "xtick.labelsize": 7.8, "ytick.labelsize": 7.8,
    "axes.linewidth": 0.7, "xtick.direction": "in", "ytick.direction": "in",
    "lines.linewidth": 1.3, "figure.dpi": 160,
    "savefig.bbox": "tight", "savefig.facecolor": "white",
})


def low_peak(levels, has_sectors=True):
    """The ring anomaly (gauge anomaly for periodic ED), grid-refined."""
    peaks = classify(levels, refined_maxima(levels), has_sectors=has_sectors)
    return named(peaks, "ring") or named(peaks, "gauge")


def ring_count(levels, has_sectors=True):
    """How many maxima the classifier assigns to the ring anomaly.

    Past |Jpm| ~ 0.3 this returns 2: the band and the spinon continuum have
    merged and the label no longer picks out a unique feature, which is why
    those points are drawn as unreliable rather than as measurements.
    """
    peaks = classify(levels, refined_maxima(levels), has_sectors=has_sectors)
    return sum(1 for p in peaks if p["label"] == "ring")


def collect():
    records = {}
    for path in sorted(SPINE.glob("spine_*.npz")):
        data = np.load(path)
        jpm = float(data["jpm"])
        tag = f"{jpm:+.3f}".replace(".", "p")
        entry = {"jpm": jpm, "g6": 12.0 * abs(jpm) ** 3, "g4": 4.0 * jpm ** 2}

        entry["counterterm"] = low_peak(np.asarray(data["levels"]))
        entry["ring_labels"] = ring_count(np.asarray(data["levels"]))

        compare = COMPARE / f"compare_{tag}.npz"
        if compare.exists():
            cached = np.load(compare)
            entry["periodic"] = low_peak(cached["periodic_levels"],
                                         has_sectors=False)
            if "polar_levels" in cached.files:
                entry["polar"] = low_peak(cached["polar_levels"])
            if "feshbach_levels" in cached.files:
                entry["feshbach"] = low_peak(cached["feshbach_levels"])
        elif "levels_periodic" in data.files:
            # couplings past the cached sweep carry their own periodic spectrum;
            # neither the polar nor the masked route was ever run there
            entry["periodic"] = low_peak(np.asarray(data["levels_periodic"]),
                                         has_sectors=False)

        # where the escaped branches were recovered, the reference is complete
        # and supersedes the cached splice
        complete = COMPLETE / f"band_{tag}.json"
        if complete.exists():
            record = json.loads(complete.read_text())
            band = np.array(record["levels"], float)
            origin = np.array(record["origin"])
            weight = np.array(record["ice_weight"], float)
            recovered = origin == "above"
            # A recovered branch continues a band branch only if it still
            # carries band-like ice weight.  Compare TYPICAL weights, not the
            # best one: at -0.30 the single best candidate (Z = 0.125) sits just
            # above half the band median and would pass, while the recovered set
            # as a whole has median 0.049 against a band median of 0.222.  The
            # median ratio separates the two cases cleanly, 1.67 at -0.28
            # against 0.22 at -0.30.
            ratio = (np.median(weight[recovered])
                     / np.median(weight[origin == "below"])
                     if recovered.any() else np.inf)
            credible = bool(ratio >= 0.5)
            entry["recovered_weight_ratio"] = float(ratio)
            entry["reference_complete"] = bool(credible)
            if (np.isfinite(band).all() and (origin == "lost").sum() == 0
                    and credible):
                full = np.asarray(data["levels"])
                upper = full[full > np.sort(data["band"])[-1]]
                spliced = np.sort(np.concatenate([np.sort(band), upper]))
                entry["feshbach"] = low_peak(spliced)
        records[round(jpm, 4)] = entry
    return records


CACHE = SPINE / "peak_comparison.json"


def main() -> None:
    if CACHE.exists() and "--recompute" not in sys.argv:
        records = {float(k): v for k, v in
                   json.loads(CACHE.read_text()).items()}
        print(f"loaded cached peaks from {CACHE} (--recompute to rebuild)")
    else:
        records = collect()
        CACHE.write_text(json.dumps({str(k): v for k, v in records.items()},
                                    indent=1))
        print(f"wrote {CACHE}")
    couplings = np.array(sorted(records))

    figure, axes = plt.subplots(1, 2, figsize=(7.6, 3.2))

    # ---------------------------------------------------------- panel (a)
    ax = axes[0]
    grid = np.geomspace(0.008, 0.34, 200)
    ax.loglog(grid, 4 * grid ** 2, ":", color=GREY, lw=1.1, zorder=1)
    ax.loglog(grid, 12 * grid ** 3, "--", color=GREY, lw=1.1, zorder=1)
    ax.text(0.011, 4 * 0.011 ** 2 * 1.7, r"$g_4$", color=GREY, ha="left",
            fontsize=8)
    ax.text(0.011, 12 * 0.011 ** 3 * 0.35, r"$g_6$", color=GREY, ha="left",
            fontsize=8)

    for name in ORDER:
        colour, marker, label = STYLE[name]
        for negative, fill in ((True, True), (False, False)):
            x, y, ok = [], [], []
            for key in couplings:
                entry = records[key]
                if (entry["jpm"] < 0) != negative:
                    continue
                value = entry.get(name)
                if value:
                    x.append(abs(entry["jpm"]))
                    y.append(value)
                    ok.append(entry.get("ring_labels", 1) <= 1)
            if not x:
                continue
            order = np.argsort(x)
            x, y = np.asarray(x)[order], np.asarray(y)[order]
            good = np.asarray(ok)[order]
            if name == "feshbach":
                # the reference is drawn wide and soft so that the counterterm,
                # which lies on top of it, does not hide it
                ax.loglog(x, y, color=colour, lw=3.2, alpha=0.35, zorder=2,
                          solid_capstyle="round")
                ax.loglog(x, y, marker=marker, color=colour, ms=4.4, lw=0,
                          mfc=colour if fill else "white", mew=1.0, zorder=2,
                          label=label if negative else None)
            else:
                ax.loglog(x[good], y[good], marker=marker, color=colour,
                          ms=3.4, mfc=colour if fill else "white", mew=1.0,
                          label=label if negative else None, zorder=3)
                if (~good).any():
                    ax.loglog(x[~good], y[~good], marker="x", color=colour,
                              ms=4.2, lw=0, mew=1.2, alpha=0.75, zorder=3)
    ax.axvspan(1.0 / 3.0, 0.36, color="#E6E8EC", zorder=0)
    ax.text(0.345, 2.2e-5, r"$g_6>g_4$", color=GREY, fontsize=7,
            rotation=90, va="bottom", ha="center")
    ax.set_xlabel(r"$|J_\pm|/J_{zz}$")
    ax.set_ylabel(r"$T_{\rm peak}/J_{zz}$")
    ax.set_xlim(0.008, 0.36)
    ax.set_title("(a) low-temperature maximum", loc="left", pad=4)
    ax.grid(color="#E6E8EC", lw=0.45, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper left", handlelength=1.4)

    # ---------------------------------------------------------- panel (b)
    ax = axes[1]
    ax.axhline(1.0, color=GREY, lw=0.8, ls="--", zorder=1)
    ax.text(0.0105, 1.28, r"$T_{\rm peak}=g_6$", color=GREY, ha="left",
            fontsize=7.4)
    for name in ORDER:
        colour, marker, label = STYLE[name]
        for negative, fill in ((True, True), (False, False)):
            x, y, ok = [], [], []
            for key in couplings:
                entry = records[key]
                if (entry["jpm"] < 0) != negative:
                    continue
                value = entry.get(name)
                if value:
                    x.append(abs(entry["jpm"]))
                    y.append(value / entry["g6"])
                    ok.append(entry.get("ring_labels", 1) <= 1)
            if not x:
                continue
            order = np.argsort(x)
            x, y = np.asarray(x)[order], np.asarray(y)[order]
            good = np.asarray(ok)[order]
            if name == "feshbach":
                ax.loglog(x, y, color=colour, lw=3.2, alpha=0.35, zorder=2,
                          solid_capstyle="round")
                ax.loglog(x, y, marker=marker, color=colour, ms=4.4, lw=0,
                          mfc=colour if fill else "white", mew=1.0, zorder=2)
            else:
                ax.loglog(x[good], y[good], marker=marker, color=colour,
                          ms=3.4, mfc=colour if fill else "white", mew=1.0,
                          zorder=3)
                if (~good).any():
                    ax.loglog(x[~good], y[~good], marker="x", color=colour,
                              ms=4.2, lw=0, mew=1.2, alpha=0.75, zorder=3)
    ax.axvspan(1.0 / 3.0, 0.36, color="#E6E8EC", zorder=0)
    ax.loglog(grid, (1.0 / (3.0 * grid)), ":", color=GREY, lw=1.1, zorder=1)
    ax.text(0.028, 1.0 / (3 * 0.028) * 1.45, r"$g_4/g_6$", color=GREY,
            fontsize=7.4, rotation=-30)
    ax.set_xlabel(r"$|J_\pm|/J_{zz}$")
    ax.set_ylabel(r"$T_{\rm peak}/g_6$")
    ax.set_xlim(0.008, 0.36)
    ax.set_title("(b) in units of the ring-exchange scale", loc="left", pad=4)
    ax.grid(color="#E6E8EC", lw=0.45, zorder=0)
    ax.set_axisbelow(True)

    figure.suptitle(r"filled: $J_\pm<0$ ($\pi$ flux)    open: $J_\pm>0$ "
                    r"(zero flux)", y=1.02, fontsize=8)
    figure.tight_layout()
    figure.savefig(FIGS / "fig_peak_comparison.pdf")
    plt.close(figure)
    print(f"wrote {FIGS / 'fig_peak_comparison.pdf'}")

    header = f"{'jpm':>7} " + " ".join(f"{n:>13}" for n in ORDER)
    print("\nlow-temperature maximum, in units of g6:")
    print(header)
    for key in sorted(records, reverse=True):
        entry = records[key]
        row = f"{key:>7.3f} "
        for name in ORDER:
            value = entry.get(name)
            row += (f"{'---':>13} " if not value
                    else f"{value / entry['g6']:>13.4f} ")
        print(row)
    for name in ORDER:
        available = [k for k in records if records[k].get(name)]
        span = f"down to {min(available):+.3f}" if available else "no points"
        print(f"{name:>13}: {len(available)} couplings, {span}")
    for key in sorted(records):
        ratio = records[key].get("recovered_weight_ratio")
        if ratio is not None and np.isfinite(ratio):
            verdict = "credible" if ratio >= 0.5 else "REJECTED"
            print(f"  recovered branches at {key:+.3f}: typical ice weight is "
                  f"{ratio:.2f}x the band's -> {verdict}")


if __name__ == "__main__":
    main()
