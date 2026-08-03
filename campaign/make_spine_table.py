#!/usr/bin/env python3
"""Emit paper/tab_spine.tex and every number quoted in the main text.

Sources, all produced by the drivers in this directory:

  spine_summary.json        the sweep: anchors, residuals, multiplicities,
                            splittings, entropy, peak counts
  refined_peaks.json        peak positions with the temperature grid refined
                            away (parabolic interpolation in log T)
  reference_comparison.json the counterterm band against the exact masked roots
                            with no match_trace shift
  sum_rule.json             the entropy sum rule on a converged T range

Usage: make_spine_table.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SPINE = ROOT / "campaign" / "outputs" / "counterterm_spine"
PAPER = ROOT / "paper"


def load(name):
    path = SPINE / name
    return json.loads(path.read_text()) if path.exists() else None


def scientific(value, digits=1):
    if value is None:
        return "---"
    if value == 0:
        return "$0$"
    exponent = int(np.floor(np.log10(abs(value))))
    mantissa = value / 10.0 ** exponent
    return rf"${mantissa:.{digits}f}\times10^{{{exponent}}}$"


def main() -> None:
    sweep = {round(r["jpm"], 4): r for r in load("spine_summary.json")}
    refined = {round(r["jpm"], 4): r for r in load("refined_peaks.json")}
    comparison = {round(r["jpm"], 4): r
                  for r in (load("reference_comparison.json") or [])}
    sum_rule = load("sum_rule.json")
    entropy = ({round(r["jpm"], 4): r for r in sum_rule["rows"]}
               if sum_rule else {})

    rows = []
    for key in sorted(sweep, reverse=True):
        s, p = sweep[key], refined[key]
        c = comparison.get(key, {})
        g6 = s["g6"]
        ring = p["counterterm_ring"]
        reference = p.get("reference_ring")
        deviation = p.get("ring_deviation")
        band = c.get("band_deviation")
        total = entropy.get(key, {}).get("counterterm",
                                        s["counterterm_entropy_total"])
        rows.append(" & ".join([
            f"${key:+.3f}$",
            f"${s['anchor']:.6f}$",
            scientific(s["self_consistency"]),
            f"${s['ground_multiplicity']}$",
            scientific(s["ground_splitting"]),
            f"${int(c.get('n_roots', 0)) or '---'}$",
            "---" if ring is None else f"${ring / g6:.4f}$",
            "---" if reference is None else f"${reference / g6:.4f}$",
            "---" if deviation is None else f"${deviation * 100:.3f}\\%$",
            "---" if band is None else f"${band * 100:.1f}\\%$",
            f"${p['counterterm_n_peaks']}$",
            f"${total:.7f}$",
        ]) + r" \\")

    header = (r"\begin{tabular}{lrrrrrrrrrrr}" "\n" r"\toprule" "\n"
              r"$\Jpm/\Jzz$ & $z_0/\Jzz$ & $|g|$ & $g_0$ & $\delta/\Jzz$ "
              r"& roots & $T_{\rm ring}/g_6$ & ref$/g_6$ & dev & band "
              r"& peaks & $S/N$\\" "\n" r"\midrule")
    (PAPER / "tab_spine.tex").write_text(
        header + "\n" + "\n".join(rows) + "\n"
        + r"\bottomrule" + "\n" + r"\end{tabular}" + "\n")
    print(f"wrote {PAPER / 'tab_spine.tex'} ({len(rows)} couplings)")

    # ------------------------------------------------ the quoted numbers
    couplings = np.array(sorted(refined))
    ring = np.array([refined[k]["counterterm_ring"] or np.nan
                     for k in couplings])
    gauge = np.array([refined[k]["periodic_gauge"] or np.nan
                      for k in couplings])
    weak = np.abs(couplings) <= 0.031

    def slope(y):
        good = weak & np.isfinite(y)
        return float(np.polyfit(np.log(np.abs(couplings[good])),
                                np.log(y[good]), 1)[0])

    print("\n--- for the abstract and results section ---")
    print(f"power law over |jpm| <= 0.031 ({int(weak.sum())} points):")
    print(f"  winding free  n = {slope(ring):.3f}  (ideal 3)")
    print(f"  periodic      n = {slope(gauge):.3f}  (ideal 2)")

    g6 = 12.0 * np.abs(couplings) ** 3
    g4 = 4.0 * couplings ** 2
    for target in (-0.010, 0.010):
        i = int(np.argmin(np.abs(couplings - target)))
        print(f"  at jpm={couplings[i]:+.3f}: ring/g6={ring[i]/g6[i]:.4f}, "
              f"periodic/g4={gauge[i]/g4[i]:.3f}, "
              f"periodic/g6={gauge[i]/g6[i]:.1f}")

    deviations = {k: refined[k]["ring_deviation"] for k in refined
                  if refined[k].get("ring_deviation") is not None}
    inner = {k: v for k, v in deviations.items() if abs(k) <= 0.18}
    print(f"ring deviation from the exact masked reference:")
    print(f"  |jpm| <= 0.18 : max {max(inner.values())*100:.3f}% "
          f"({len(inner)} couplings)")
    for k in sorted(deviations):
        if abs(k) > 0.18:
            print(f"  jpm = {k:+.3f}  : {deviations[k]*100:.3f}%")

    if comparison:
        complete = [r for r in comparison.values() if r["n_roots"] == 90]
        print(f"ground energy vs the lowest masked root, unshifted: "
              f"max |diff| = "
              f"{max(abs(r['ground_difference']) for r in complete):.1e} "
              f"over {len(complete)} couplings")
        print(f"band level deviation: max "
              f"{max(r['band_deviation'] for r in complete)*100:.1f}% "
              f"of the band width")

    splits = np.array([sweep[k]["ground_splitting"] for k in sweep])
    print(f"worst ground-multiplet splitting: {splits.max():.1e}; "
          f"its Schottky temperature {splits.max()/2.4:.1e}")
    print(f"measured ground multiplicities: "
          f"{sorted({sweep[k]['ground_multiplicity'] for k in sweep})}")
    print(f"peak counts: "
          f"{sorted({refined[k]['counterterm_n_peaks'] for k in refined})}")
    third = [k for k in sorted(refined)
             if refined[k]["counterterm_sector"] is not None]
    print(f"couplings with a resolved multiplet (third) anomaly: {third}")
    missing = [k for k in sorted(refined)
               if refined[k]["counterterm_ring"] is None]
    print(f"couplings where the ring anomaly is not a separate maximum: "
          f"{missing}")
    if sum_rule:
        worst = max(abs(r["counterterm"] - sum_rule["target"])
                    for r in sum_rule["rows"])
        worst_periodic = max(abs(r["periodic"] - sum_rule["target"])
                             for r in sum_rule["rows"])
        print(f"entropy sum rule on T_max={sum_rule['tmax']:g}: worst "
              f"deviation {worst:.1e} (counterterm), "
              f"{worst_periodic:.1e} (periodic)")
    symmetrised = [(k, sweep[k].get("counterterm_ring_deviation"),
                    sweep[k].get("symmetrised_ring_deviation"))
                   for k in sorted(sweep)]
    worse = [k for k, a, b in symmetrised
             if a is not None and b is not None and b > a + 1e-12]
    print(f"symmetrisation strictly worse at {len(worse)} couplings: {worse}")


if __name__ == "__main__":
    main()
