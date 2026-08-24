"""Does anything distinguish the soft level from its partner AT THE SAME q?

Two objections this script is written to settle, either of which would sink
the roton reading:

  A. "The real-space mechanism works for everything in the spectrum."
     Correct as stated: the vanishing curl on one hexagon family is a property
     of the MOMENTUM, so every level at L sees it.  It explains a low centroid
     at L; it does not, by itself, explain why ONE level is soft and its
     partner is not.

     The escape, if there is one: the moments are not scalars.  S^z(q)|0> lives
     in a two-dimensional transverse channel space, and the zeroth and first
     moments can be resolved WITHIN it.  If the Feynman bound differs BETWEEN
     the two channels at the same q, then the ground state already knows which
     polarization goes soft, and the mechanism does discriminate.  If the two
     channel bounds are equal, it does not, and "roton" reduces to "lower".

  B. "The plot makes it look like more than two branches."
     Quantified here honestly: how much of the response actually sits in the
     top two levels, and how many levels are needed to reach 90%.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

D = Path(__file__).resolve().parent / "outputs" / "ring_model_dssf"
SC = 0.58684


def sep(t):
    print("\n" + "=" * 76)
    print(t)
    print("=" * 76)


def main() -> int:
    d = json.load(open(D / "ring48_polarization_all.json"))

    # one representative entry per star
    reps = {}
    for v in d.values():
        if "sigma" not in v:
            continue
        reps.setdefault(v["star"], v)

    # ---------------------------------------------------------------- A
    sep("A.  Channel-resolved Feynman bound: does the ground state know "
        "which\n    polarization goes soft?")
    print("""
    For each transverse channel lam the moments are
        S^lam    = sum_n  w_n^lam
        f1^lam   = sum_n  omega_n w_n^lam
        omegaF^lam = f1^lam / S^lam
    each a ground-state quantity resolved within the two-dimensional
    transverse space.  omegaF^lam is the variational energy of the trial
    state built from that channel alone, hence an upper bound for it.
""")
    print(f"  {'star':<24s} {'ch':>3s} {'S^lam':>9s} {'f1^lam':>9s} "
          f"{'omegaF':>9s} {'w_dom':>8s} {'omega_dom':>10s}")
    print("  " + "-" * 76)

    summary = []
    for star, v in reps.items():
        lines = v["lines"]
        wG = SC * v["sigma"][0]
        chan = []
        for lam in (0, 1):
            w = np.array([ln["pol"][lam] for ln in lines])
            om = np.array([ln["omega"] for ln in lines])
            S = w.sum()
            if S < 1e-14:
                continue
            f1 = float((om * w).sum())
            wF = f1 / S
            k = int(np.argmax(w))
            chan.append((wF, S, f1, om[k], w[k] / S))
        if len(chan) < 2:
            continue
        chan.sort()                      # softest channel bound first
        for i, (wF, S, f1, od, wd) in enumerate(chan):
            print(f"  {star if i == 0 else '':<24s} {i:>3d} {S:>9.5f} "
                  f"{f1:>9.5f} {wF:>9.4f} {wd:>8.4f} {od:>10.4f}")
        ratio = chan[0][0] / chan[1][0]
        dom = chan[0][3] / chan[1][3]
        summary.append((star, ratio, dom, wG, chan[0][3], chan[1][3]))
        print(f"  {'':<24s} {'-->':>3s} channel bound ratio "
              f"{ratio:.4f}   dominant-pole ratio {dom:.4f}")

    sep("A (verdict).  Channel bound vs the splitting it is supposed to "
        "predict")
    print(f"  {'star':<24s} {'omegaF_0/omegaF_1':>18s} "
          f"{'omega_0/omega_1':>16s} {'agree?':>8s}")
    print("  " + "-" * 70)
    for star, ratio, dom, wG, o0, o1 in sorted(summary, key=lambda r: r[1]):
        agree = "YES" if (ratio < 0.98) == (dom < 0.98) else "no"
        print(f"  {star:<24s} {ratio:>18.4f} {dom:>16.4f} {agree:>8s}")
    print("""
  Read this as follows.  If the channel-resolved bound is EQUAL between the
  two channels (ratio 1.000) while the measured poles are split, then the
  ground-state moments do NOT distinguish the polarizations and the
  mechanism cannot explain the splitting -- only the overall scale at that
  momentum.  If the bound is split in the same sense and comparable size,
  the ground state does already know.
""")

    # ---------------------------------------------------------------- B
    sep("B.  How many branches are there really?")
    print(f"  {'star':<24s} {'top1':>7s} {'top2':>7s} {'top3':>7s} "
          f"{'n for 90%':>10s} {'n levels':>9s}")
    print("  " + "-" * 70)
    for star, v in reps.items():
        w = np.array(sorted((ln["w"] for ln in v["lines"]), reverse=True))
        w = w / w.sum()
        c = np.cumsum(w)
        n90 = int(np.searchsorted(c, 0.90) + 1)
        print(f"  {star:<24s} {w[0]:>7.3f} {c[1]:>7.3f} {c[2]:>7.3f} "
              f"{n90:>10d} {len(w):>9d}")
    print("""
  'top2' is the cumulative fraction of the sublattice-summed response in the
  two strongest levels.  A two-branch description is honest only to the
  extent that number is close to 1.
""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
