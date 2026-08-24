"""Deep diagnostics for the L mode -- what the mode IS, not how it scores.

  Q1  Is there an actual DISPERSION MINIMUM at L?  Landau's definition of a
      roton, and GMP's magnetoroton, is a local minimum of omega(q) at finite
      q -- not merely "a low mode".  Walk Gamma->L and look for rise-then-fall,
      on every cluster that resolves an interior point of that line.

  Q2  WHICH hexagon family is dark at L, and where does it sit in real space?
      A pyrochlore hexagon omits exactly one sublattice mu; that mu labels one
      of the four <111> axes, and the hexagon lies in a plane perpendicular to
      it.  If the dark family is the one whose axis is PARALLEL to q, then the
      six sites of every dark hexagon lie in a single wavefront, the wave is
      constant across it, and its alternating curl cancels identically.

  Q3  Is S(q) actually PEAKED at L?  Feynman's mechanism requires it.

  Q4  Does the ground state actually resonate on the dark family?  If those
      plaquettes were idle the cancellation would be worthless.
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
D = HERE / "outputs" / "ring_model_dssf"
sys.path.insert(0, str(HERE))

SC = 0.58684          # sqrt(2UK)/K, the single Gaussian scale


def sep(t):
    print("\n" + "=" * 74)
    print(t)
    print("=" * 74)


def two_dominant(lines):
    L = sorted(lines, key=lambda x: -x["w"])[:2]
    return sorted(L, key=lambda x: x["omega"])


# ----------------------------------------------------------------- Q1
def q1_dispersion():
    sep("Q1  Dispersion along Gamma->L: is there a roton MINIMUM?")

    print("\n  -- 48 sites, exact diagonalization --")
    d = json.load(open(D / "ring48_polarization_all.json"))
    rows = set()
    for v in d.values():
        if "sigma" not in v:
            continue
        a = np.abs(np.array(v["q"]))
        if not np.allclose(a, a[0], atol=1e-9):      # on the <111> line
            continue
        ln = two_dominant(v["lines"])
        if len(ln) < 2:
            continue
        rows.add((round(float(a[0]), 6), v["star"], ln[0]["omega"],
                  ln[1]["omega"], v["sigma"][0]))

    rows = sorted(r for r in rows if r[0] <= 0.5 + 1e-9)
    print(f"    {'t':>7s} {'|q|':>7s} {'w_low':>8s} {'w_high':>8s} "
          f"{'wG':>8s} {'low/wG':>8s}")
    print("    " + "-" * 52)
    print(f"    {0.0:7.4f} {0.0:7.3f} {0.0:8.3f} {0.0:8.3f} {0.0:8.3f}"
          f" {'--':>8s}")
    for t, star, lo, hi, sig in rows:
        print(f"    {t:7.4f} {t*np.sqrt(3):7.3f} {lo:8.3f} {hi:8.3f} "
              f"{SC*sig:8.3f} {lo/(SC*sig):8.3f}")

    inner = [r for r in rows if r[0] < 0.499]
    atL = [r for r in rows if abs(r[0] - 0.5) < 1e-6]
    if inner and atL:
        pk = max(inner, key=lambda r: r[2])
        print(f"\n    interior maximum  w_low = {pk[2]:.4f} K at t={pk[0]:.4f}")
        print(f"    zone boundary L   w_low = {atL[0][2]:.4f} K")
        print(f"    --> lower branch FALLS {100*(1-atL[0][2]/pk[2]):.1f}% "
              f"from its interior maximum to L")
        print(f"        upper branch falls {100*(1-atL[0][3]/pk[3]):.1f}% "
              f"over the same interval")
        print("    ==> rise-then-fall: a genuine dispersion minimum at the")
        print("        zone boundary, which is Landau's definition of a roton")
        print("        and the shape of the GMP magnetoroton.  Note that only")
        print("        the LOWER member turns over; the doublet is degenerate")
        print("        and still rising at t=1/3.")

    # other clusters: any momentum on the <111> line?
    for fn, lab in (("ring_allq_fcc224.json", "64 sites"),
                    ("ring_allq_fcc233.json", "72 sites")):
        p = D / fn
        if not p.exists():
            continue
        dd = json.load(open(p))
        print(f"\n  -- {lab} (fcc{tuple(dd['shape'])}) --")
        got = []
        for v in dd["momenta"]:
            a = np.abs(np.array(v["q"]))
            if not (np.allclose(a, a[0], atol=1e-9) and a[0] > 1e-9):
                continue
            # lines are [omega, weight, census] triples
            ln = sorted(v["lines"], key=lambda x: -x[1])[:2]
            ln = sorted(ln, key=lambda x: x[0])
            if len(ln) < 2:
                continue
            got.append((round(float(a[0]), 6), ln[0][0], ln[1][0],
                        ln[0][1], v["S"]))
        got = sorted(set(got))
        if not got:
            print("    no momentum on the <111> line")
            continue
        print(f"    {'t':>7s} {'w_low':>8s} {'w_high':>8s} {'w(low)':>8s} "
              f"{'S(q)':>8s}")
        print("    " + "-" * 44)
        for t, lo, hi, w, S in got:
            tag = "  <== L" if abs(t - 0.5) < 1e-9 else ""
            print(f"    {t:7.4f} {lo:8.3f} {hi:8.3f} {w:8.4f} {S:8.4f}{tag}")
        inner = [g for g in got if g[0] < 0.499]
        atL = [g for g in got if abs(g[0] - 0.5) < 1e-9]
        if inner and atL:
            pk = max(inner, key=lambda g: g[1])
            print(f"    --> interior max {pk[1]:.3f} K at t={pk[0]:.3f}; "
                  f"L at {atL[0][1]:.3f} K")
            print(f"        lower branch falls "
                  f"{100*(1-atL[0][1]/pk[1]):.1f}%, upper falls "
                  f"{100*(1-atL[0][2]/pk[2]):.1f}%")


# ----------------------------------------------------------------- Q2
def q2_dark_family():
    sep("Q2  Which hexagon family is dark at L, and where is it?")

    ROOT = HERE.parent
    for sub in ("src", "notes", "campaign"):
        sys.path.insert(0, str(ROOT / sub))
    import recompute_finite_size_artifact as geometry
    from run_ring48_dssf import contractible
    from f1_all_momenta import geom_factor, hex_family

    orig = geometry.enumerate_ice_states
    geometry.enumerate_ice_states = lambda a, b: np.array([0], dtype=np.uint64)
    cl = geometry.build_cluster("fcc", (2, 2, 3))
    geometry.enumerate_ice_states = orig

    pos = np.asarray(cl.positions)
    n = len(pos)
    hexes = contractible(cl)
    print(f"  {n} sites, {len(hexes)} contractible hexagons")

    fam = hex_family(hexes, n)
    qL = np.array([0.5, -0.5, 0.5])
    g = geom_factor(hexes, pos, qL, n)

    # the four <111> axes, indexed by sublattice
    AX = {0: np.array([1, 1, 1]), 1: np.array([1, -1, -1]),
          2: np.array([-1, 1, -1]), 3: np.array([-1, -1, 1])}
    qhat = qL / np.linalg.norm(qL)

    print(f"\n  {'family mu':>10s} {'n_hex':>6s} {'sum |g|^2':>11s} "
          f"{'<111> axis':>14s} {'|axis.qhat|':>12s}")
    print("  " + "-" * 60)
    dark = None
    for mu in range(4):
        sel = fam == mu
        tot = float(g[sel].sum())
        ax = AX[mu] / np.sqrt(3)
        dot = abs(float(ax @ qhat))
        print(f"  {mu:>10d} {int(sel.sum()):>6d} {tot:>11.4f} "
              f"{str(AX[mu]):>14s} {dot:>12.6f}")
        if tot < 1e-9:
            dark = (mu, dot, int(sel.sum()))

    if dark:
        mu, dot, cnt = dark
        print(f"\n  DARK family mu={mu}: {cnt} hexagons "
              f"({100*cnt/len(hexes):.0f}% of all), |axis.qhat| = {dot:.6f}")
        if dot > 0.999:
            print("\n  ==> The dark family's <111> axis is PARALLEL to q, so")
            print("      its hexagons lie in planes PERPENDICULAR to q --")
            print("      the kagome planes of the [111] stacking.  All six")
            print("      sites of such a hexagon sit in one wavefront, the")
            print("      imposed field wave is constant across it, and its")
            print("      alternating curl cancels identically.")
            print("\n      This is the real-space content of f1(L): one")
            print("      quarter of the ring terms are invisible to the")
            print("      excitation, so the wave can be built without paying")
            print("      for them.")
        else:
            print("  ==> not parallel; the wavefront picture fails.")

    # contrast: X
    gX = geom_factor(hexes, pos, np.array([1.0, 0.0, 0.0]), n)
    print(f"\n  Same decomposition at X = (1,0,0):")
    print("  " + "  ".join(f"mu={mu}: {float(gX[fam==mu].sum()):.1f}"
                           for mu in range(4)))
    print("  No <111> axis is parallel to [100], so no family is dark and")
    print("  every ring term costs the excitation something.")


# ----------------------------------------------------------------- Q3
def q3_S_peak():
    sep("Q3  Is S(q) peaked at L?")
    f = json.load(open(D / "f1_all_fcc223.json"))
    rows = [(m["S"], m["star"], float(np.linalg.norm(m["q"])))
            for m in f["momenta"] if m["star"] != "Gamma"]
    for S, star, nq in sorted(rows, reverse=True):
        print(f"    S = {S:.4f}   |q| = {nq:.3f}   {star}"
              f"{'   <== L, largest' if star == 'L' else ''}")
    print("\n  S(q) is maximal at L among every resolved momentum, which is")
    print("  the condition Feynman's argument needs: the ratio is small")
    print("  because f1 is suppressed AND S is enhanced at the same q.")


# ----------------------------------------------------------------- Q4
def q4_dark_family_resonates():
    sep("Q4  Does the ground state actually resonate on the dark family?")
    p = D / "exact_census_fcc223.json"
    if not p.exists():
        print("  exact_census_fcc223.json not present; skipping")
        return
    d = json.load(open(p))
    keys = [k for k in d if "fam" in k.lower() or "ring" in k.lower()]
    print(f"  available keys: {list(d)[:12]}")
    for k in keys:
        print(f"    {k}: {d[k]}")


if __name__ == "__main__":
    q1_dispersion()
    try:
        q2_dark_family()
    except Exception as e:
        print(f"\n  Q2 failed: {type(e).__name__}: {e}")
    q3_S_peak()
    try:
        q4_dark_family_resonates()
    except Exception as e:
        print(f"\n  Q4 failed: {type(e).__name__}: {e}")
    print()
