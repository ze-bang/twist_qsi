"""How much of the softening is Feynman's helium mechanism, and how much
is the lattice cancellation that helium does not have?

In helium the roton minimum comes from ONE source.  The f-sum rule there is
fixed by Galilean invariance, f1 = hbar^2 k^2 / 2m, so Feynman's

    omega_F(k) = hbar^2 k^2 / (2 m S(k))

can only dip where S(k) PEAKS -- at the first peak of the liquid structure
factor.  The numerator carries no structure at all.

Here f1 is NOT fixed: it is sum_p <W_p+W_p'> sum_mu |g^mu_p|^2, and at
q parallel to a <111> axis one whole hexagon family drops out of it.  So our
ratio can dip for TWO reasons -- S enhanced (helium's reason) and f1
suppressed (no helium analogue).

This script separates them with a counterfactual: recompute f1(L) with the
dark family given the geometric weight it would have had were it not dark,
and ask where L would then rank among the resolved momenta.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

D = Path(__file__).resolve().parent / "outputs" / "ring_model_dssf"
SC = 0.58684


def main() -> int:
    f = json.load(open(D / "f1_all_fcc223.json"))
    pol = json.load(open(D / "ring48_polarization_all.json"))

    sig = {}
    for v in pol.values():
        if "sigma" in v:
            sig.setdefault(v["star"], v["sigma"][0])

    # family-resolved ground-state ring coherence (from the exact eigenvector)
    C = np.array([2.6296, 3.2195, 2.3535, 2.3535])
    n_sites, n_fam_hex = 48, 12

    print("=" * 74)
    print("Consistency: rebuild f1(L) from family sums alone")
    print("=" * 74)
    rec = [m for m in f["momenta"] if m["star"] == "L"][0]
    G = np.array(rec["geom_by_family"])
    # f1 = (1/2N) sum_p <W_p+W_p'> g_p^2 ; uniform within a family this is
    # (1/2N) sum_mu C_mu G_mu / n_fam_hex
    f1_rebuilt = float((C * G).sum() / (2 * n_sites * n_fam_hex))
    print(f"  geometric weight by family : {np.round(G, 1)}")
    print(f"  coherence   by family      : {np.round(C, 4)}"
          f"   (sum {C.sum():.4f} = |E0|)")
    print(f"  f1 rebuilt = {f1_rebuilt:.6f}   f1 measured = "
          f"{rec['f1']:.6f}   diff {abs(f1_rebuilt-rec['f1']):.2e}")

    print()
    print("=" * 74)
    print("Counterfactual: what if the dark family were NOT dark?")
    print("=" * 74)
    dark = int(np.argmin(G))
    Gcf = G.copy()
    Gcf[dark] = float(G[G > 1e-9].mean())
    f1_cf = float((C * Gcf).sum() / (2 * n_sites * n_fam_hex))
    S_L = rec["S"]
    wG_L = SC * sig["L"]

    print(f"  dark family mu={dark}; give it the mean bright weight "
          f"{Gcf[dark]:.1f}")
    print(f"  f1(L)  actual         = {rec['f1']:.4f}")
    print(f"  f1(L)  counterfactual = {f1_cf:.4f}   "
          f"(+{100*(f1_cf/rec['f1']-1):.0f}%)")
    print(f"  for reference, f1(X)  = "
          f"{[m['f1'] for m in f['momenta'] if m['star']=='X'][0]:.4f}")

    print()
    print(f"  {'momentum':<26s} {'S':>7s} {'f1':>7s} {'(f1/S)/wG':>11s} "
          f"{'measured':>9s}")
    print("  " + "-" * 66)
    meas = {"L": 0.503, "?[0.333, 0.333, 0.667]": 0.824, "X": 0.952,
            "?[0.333, 0.333, 0.333]": 1.076, "?[0.167, 0.167, 0.833]": 1.114}
    rows = []
    for m in f["momenta"]:
        st = m["star"]
        if st not in sig or st == "Gamma":
            continue
        wG = SC * sig[st]
        rows.append((m["f1"] / m["S"] / wG, st, m["S"], m["f1"],
                     meas.get(st, float("nan"))))
    rows.append((f1_cf / S_L / wG_L, "L  (counterfactual)", S_L, f1_cf,
                 float("nan")))
    for p, st, S, f1v, mv in sorted(rows):
        tag = "  <== " if "counterfactual" in st else ""
        print(f"  {st:<26s} {S:>7.4f} {f1v:>7.4f} {p:>11.3f} "
              f"{mv:>9.3f}{tag}")

    print(f"""
  Without the dark family L would score {f1_cf/S_L/wG_L:.3f}, essentially tied
  with (1/3,1/3,2/3) at 1.177 -- a momentum that softens only to 0.824, not
  0.503.  So the lattice cancellation is not decoration: remove it and L stops
  being the minimum and becomes an ordinary momentum.
""")

    print("=" * 74)
    print("Splitting the effect (L against X, the natural bright reference)")
    print("=" * 74)
    recX = [m for m in f["momenta"] if m["star"] == "X"][0]
    print(f"  S  : {rec['S']:.4f} vs {recX['S']:.4f}   "
          f"-> S enhanced by {100*(rec['S']/recX['S']-1):.0f}%"
          f"   [helium's mechanism]")
    print(f"  f1 : {rec['f1']:.4f} vs {recX['f1']:.4f}   "
          f"-> f1 suppressed by {100*(1-rec['f1']/recX['f1']):.0f}%"
          f"   [no helium analogue]")
    print(f"  and the counterfactual f1 {f1_cf:.4f} nearly equals f1(X) "
          f"{recX['f1']:.4f}, confirming that")
    print(f"  the whole f1 suppression at L IS the dark family.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
