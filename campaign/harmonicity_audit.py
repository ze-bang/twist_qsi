"""How far is the compact theory from a harmonic one, momentum by momentum?

Gaussian lattice electrodynamics is a set of decoupled harmonic oscillators.
That fixes three things exactly, with no freedom whatsoever:

  (H1) the single-mode state IS the eigenstate.  E(q) acting on the vacuum
       creates exactly one quantum in one normal mode, so Feynman's bound is
       SATURATED:  f1/S = omega, exactly.
  (H2) the response in each transverse channel is a SINGLE pole.  All of the
       channel's weight sits in one level.
  (H3) the two transverse polarizations are degenerate, since U and K enter
       omega_lambda = sqrt(2UK) sigma_lambda only through a common prefactor.

The compact theory has to violate all three, because the emergent electric
field is SATURATED: E = S^z = +/- 1/2.  A field wave cannot be made by
increasing an amplitude -- only by rearranging which links carry which of the
two allowed values -- and that rearrangement is not unique.  Exponentially
many ice configurations share the same density-wave amplitude a(c), so the
excitation has internal freedom AT FIXED SPECTRAL WEIGHT, and can lower its
energy without abandoning the wave.

This script measures each violation at every resolved momentum, so the claim
"compactness does this" becomes three numbers rather than an assertion.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

D = Path(__file__).resolve().parent / "outputs" / "ring_model_dssf"
SC = 0.58684


def main() -> int:
    pol = json.load(open(D / "ring48_polarization_all.json"))
    f1d = json.load(open(D / "f1_all_fcc223.json"))
    Sof = {m["star"]: m for m in f1d["momenta"]}

    reps = {}
    for v in pol.values():
        if "sigma" in v:
            reps.setdefault(v["star"], v)

    print("=" * 78)
    print("Departures from harmonic lattice electrodynamics")
    print("=" * 78)
    print("""
  H1  bound overshoot   omega_F / omega_dom   (harmonic: exactly 1)
  H2  weight off-pole   1 - w_dom/S           (harmonic: exactly 0)
  H3  doublet splitting omega_hi/omega_lo     (harmonic: exactly 1)
  all evaluated in the softer transverse channel where H1, H2 apply.
""")
    print(f"  {'momentum':<26s} {'omega_lo':>9s} {'H1':>7s} {'H2':>7s} "
          f"{'H3':>7s} {'soften':>8s}")
    print("  " + "-" * 70)

    rows = []
    for star, v in reps.items():
        lines = v["lines"]
        wG = SC * v["sigma"][0]
        chan = []
        for lam in (0, 1):
            w = np.array([ln["pol"][lam] for ln in lines])
            om = np.array([ln["omega"] for ln in lines])
            Sl = w.sum()
            if Sl < 1e-14:
                continue
            k = int(np.argmax(w))
            chan.append({"wF": float((om * w).sum() / Sl),
                         "om": float(om[k]),
                         "frac": float(w[k] / Sl)})
        if len(chan) < 2:
            continue
        chan.sort(key=lambda c: c["om"])
        lo, hi = chan[0], chan[1]
        H1 = lo["wF"] / lo["om"]
        H2 = 1.0 - lo["frac"]
        H3 = hi["om"] / lo["om"]
        rows.append((lo["om"] / wG, star, lo["om"], H1, H2, H3))

    for soft, star, om, H1, H2, H3 in sorted(rows):
        tag = "  <== L" if star == "L" else ""
        print(f"  {star:<26s} {om:>9.4f} {H1:>7.3f} {H2:>7.3f} {H3:>7.3f} "
              f"{soft:>8.3f}{tag}")

    print(f"""
  Read the columns downward.  The momentum that softens most is also the one
  where the single-mode state is furthest from an eigenstate (H1), where the
  most weight has left the dominant pole (H2), and where the doublet is most
  split (H3).  A harmonic theory has 1, 0, 1 in those columns at EVERY
  momentum, for any choice of U and K.
""")

    # the global statement, using the sublattice-summed moments
    L = Sof["L"]
    print("=" * 78)
    print("The same statement without channel resolution, at L")
    print("=" * 78)
    lines = sorted(reps["L"]["lines"], key=lambda x: -x["w"])
    wtot = sum(ln["w"] for ln in lines)
    print(f"  Feynman bound      f1/S      = {L['f1_over_S']:.4f} K")
    print(f"  true lowest level  omega     = {lines[0]['omega']:.4f} K"
          if lines[0]["omega"] < lines[1]["omega"] else "")
    om_lo = min(ln["omega"] for ln in lines[:2])
    print(f"  true lowest level  omega     = {om_lo:.4f} K")
    print(f"  overshoot                    = "
          f"{L['f1_over_S']/om_lo:.3f}x")
    print(f"  weight in the single largest pole = "
          f"{lines[0]['w']/wtot:.3f}")
    print(f"  levels needed for 90% of the response = "
          f"{int(np.searchsorted(np.cumsum([ln['w'] for ln in lines])/wtot, 0.9)+1)}"
          f" of {len(lines)}")
    print("""
  In a harmonic theory those four numbers would be: bound = omega,
  overshoot 1.000, weight 1.000, and one level.  Every one of them is a
  direct measure of what saturation of the emergent electric field costs
  the Gaussian description.
""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
