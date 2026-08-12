"""fig_causal_ladder.pdf: what the loop hierarchy does, and does not, do.

The figure was designed to show a convergent operator ladder.  It shows the
opposite, and the norm budget explains why, so the panels are ordered to make
the mechanism legible rather than to hide the negative result.

  (a) Norm budget per coupling: the Frobenius norm carried by each simple-loop
      perimeter, by the vertex-joined and disconnected COMPOSITE processes, and
      by the beyond-12 tail.  The point of the panel is a single comparison:
      the composite norm exceeds the L=10 and L=12 simple-loop norms at every
      coupling, even though simple loops carry >99% of the total.  Both facts
      are true simultaneously because L=6 dominates the total.

  (b) Relative error of the ring-scale heat-capacity maximum against the full
      masked map, as operators are added.  Non-monotone, and not converging.

  (c) The mechanism.  At each rung, the norm ADDED by that rung against the
      norm still OMITTED (composites plus the beyond-12 tail plus any
      higher simple loops).  Past L=8 the ladder adds less than it omits, so
      it is not an expansion in a small parameter and has no reason to
      converge.

  (d) Four observables at one coupling, showing that the failure is not
      uniform: the longitudinal spectral weight converges monotonically while
      the heat-capacity peak does not.  Whatever is being missed matters for
      band-bottom near-degeneracies far more than for integrated weight.

All inputs are on disk; this script computes nothing new.  Source:
  campaign/outputs/causal_ladder/ladder_-0p*.json   (seven couplings)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "campaign" / "outputs" / "causal_ladder"
FIGS = ROOT / "paper" / "figs"

RUNGS = ["H6", "H6+H8", "H6+H8+H10", "H6+H8+H10+H12"]
RUNG_LABEL = [r"$H^{(6)}$", r"$H^{(6+8)}$", r"$H^{(6+8+10)}$",
              r"$H^{(6+8+10+12)}$"]
DETAIL_COUPLING = -0.20


def load():
    records = []
    for path in sorted(DATA.glob("ladder_-0p*.json")):
        records.append(json.loads(path.read_text()))
    records.sort(key=lambda r: r["jpm"], reverse=True)
    if not records:
        raise SystemExit(f"no ladder records in {DATA}")
    return records


def quad(*values):
    """Frobenius norms combine in quadrature, not linearly."""
    return float(np.sqrt(sum(v * v for v in values)))


def main() -> int:
    records = load()
    couplings = [r["jpm"] for r in records]
    print(f"{len(records)} couplings: {couplings}", flush=True)

    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.6))
    (ax_a, ax_b), (ax_c, ax_d) = axes

    # ---------------------------------------------------------------- (a)
    keys = ["simple_L6", "simple_L8", "simple_L10", "simple_L12",
            "composite", "long"]
    labels = [r"$L{=}6$", r"$L{=}8$", r"$L{=}10$", r"$L{=}12$",
              "composite", r"$L{>}12$"]
    x = np.arange(len(records))
    width = 0.13
    for index, (key, label) in enumerate(zip(keys, labels)):
        values = [r["norm_budget"][key] for r in records]
        emphasis = dict(edgecolor="k", linewidth=0.9) if key == "composite" else {}
        ax_a.bar(x + (index - 2.5) * width, values, width, label=label,
                 **emphasis)
    ax_a.set_yscale("log")
    ax_a.set_xticks(x)
    ax_a.set_xticklabels([f"{c:.2f}" for c in couplings], fontsize=7)
    ax_a.set_xlabel(r"$\Jpm/\Jzz$".replace("\\Jpm", r"J_\pm").replace(
        "\\Jzz", "J_{zz}"), fontsize=8)
    ax_a.set_ylabel("Frobenius norm", fontsize=8)
    ax_a.legend(fontsize=6, ncol=3, frameon=False)
    ax_a.set_title("(a) norm budget: composite exceeds $L{=}10,12$",
                   fontsize=8)
    ax_a.tick_params(labelsize=7)

    # ---------------------------------------------------------------- (b)
    for record in records:
        errors = [100.0 * record["ladder"][k]["heat_peak_rel_error"]
                  for k in RUNGS]
        ax_b.plot(range(len(RUNGS)), errors, "o-", markersize=3.2,
                  linewidth=1.0, label=f"{record['jpm']:.2f}")
    ax_b.axhline(0.0, color="k", linewidth=0.8, linestyle=":")
    ax_b.set_xticks(range(len(RUNGS)))
    ax_b.set_xticklabels(RUNG_LABEL, fontsize=6.5, rotation=20)
    ax_b.set_ylabel("ring-peak error vs full map (\\%)", fontsize=8)
    ax_b.legend(fontsize=6, ncol=2, frameon=False, title=r"$J_\pm/J_{zz}$",
                title_fontsize=6)
    ax_b.set_title("(b) the ladder does not converge", fontsize=8)
    ax_b.tick_params(labelsize=7)

    # ---------------------------------------------------------------- (c)
    for record in records:
        budget = record["norm_budget"]
        added, omitted = [], []
        simple = [budget["simple_L6"], budget["simple_L8"],
                  budget["simple_L10"], budget["simple_L12"]]
        fixed = quad(budget["composite"], budget["long"])
        for rung in range(4):
            added.append(simple[rung])
            # everything still missing after including simple[0..rung]
            omitted.append(quad(fixed, *simple[rung + 1:]))
        ax_c.plot(range(4), np.array(added) / np.array(omitted), "s-",
                  markersize=3.0, linewidth=1.0, label=f"{record['jpm']:.2f}")
    ax_c.axhline(1.0, color="k", linewidth=0.9, linestyle="--")
    ax_c.set_yscale("log")
    ax_c.set_xticks(range(4))
    ax_c.set_xticklabels([r"$+L6$", r"$+L8$", r"$+L10$", r"$+L12$"],
                         fontsize=6.5)
    ax_c.set_ylabel("norm added / norm still omitted", fontsize=8)
    ax_c.set_title("(c) past $L{=}8$ the ladder adds less than it omits",
                   fontsize=8)
    ax_c.tick_params(labelsize=7)

    # ---------------------------------------------------------------- (d)
    detail = min(records, key=lambda r: abs(r["jpm"] - DETAIL_COUPLING))
    metrics = [("heat_peak_rel_error", "ring-scale $C(T)$ peak"),
               ("spectrum_rms_rel", "band spectrum (rms)"),
               ("ground_rel_error", "ground energy"),
               ("weight_rel_error", "longitudinal weight")]
    for key, label in metrics:
        values = [100.0 * detail["ladder"][k][key] for k in RUNGS]
        ax_d.plot(range(len(RUNGS)), values, "o-", markersize=3.2,
                  linewidth=1.0, label=label)
    ax_d.set_xticks(range(len(RUNGS)))
    ax_d.set_xticklabels(RUNG_LABEL, fontsize=6.5, rotation=20)
    ax_d.set_ylabel("relative error (\\%)", fontsize=8)
    ax_d.legend(fontsize=6, frameon=False)
    ax_d.set_title(rf"(d) not uniform, at $J_\pm/J_{{zz}}={detail['jpm']:.2f}$",
                   fontsize=8)
    ax_d.tick_params(labelsize=7)

    fig.tight_layout()
    FIGS.mkdir(parents=True, exist_ok=True)
    out = FIGS / "fig_causal_ladder.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    print(f"wrote {out}", flush=True)

    # ------------------------------------------------- printed summary
    print("\nnorm budget and ladder behaviour:")
    for record in records:
        budget = record["norm_budget"]
        peak = [100.0 * record["ladder"][k]["heat_peak_rel_error"]
                for k in RUNGS]
        print(f"  jpm={record['jpm']:+.3f} simple_fraction="
              f"{budget['simple_fraction']:.4f}  composite={budget['composite']:.4f}"
              f"  L10={budget['simple_L10']:.4f} L12={budget['simple_L12']:.4f}"
              f"  peak err {peak[0]:.1f} -> {peak[1]:.1f} -> {peak[2]:.1f}"
              f" -> {peak[3]:.1f} %")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
