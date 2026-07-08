#!/usr/bin/env python3
r"""E_g thermal-expansion softening B(Jpm) in the ice-manifold effective model,
bare (PBC) vs zero-transport clean (delta=0), for Jpm = -0.10 ... +0.05.

Framework = recompute_finite_size_artifact.py (ice manifold + Schrieffer-Wolff
rows H2 [four-loops, order Jpm^2] and H3 [hexagons, order Jpm^3] + delta=0
projector).  The E_g source of strength lambda softens each *ring* (off-diagonal)
coupling by its established per-ring coefficient,

    g_ring(lambda) = g_ring (1 - kappa_ring lambda^2),
    kappa_ring = -B_ring/(Jpm Jzz),   B_4 = 6 (four-loop),  B_6 = 15 (hexagon),

i.e. the row coefficient c*Jpm^k -> c*Jpm^k + c*B_ring*Jpm^(k-1)*lambda^2/Jzz on
off-diagonal ring rows (B_4=6, B_6=15 are the path-counted values verified in the
companion gauge-note).  The measured band softening is

    B(Jpm) = [1 - Tpk(lambda)/Tpk(0)] / lambda^2  *  |Jpm| Jzz,

read from the lower (gauge) C(T) peak.  Expectation:
  bare  -> B ~ 6   (dominant winding four-loops, kappa_4),
  clean -> B  = 15  (only contractible hexagons survive; spectrum rescales -> kappa_6).
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

NOTES = Path(__file__).resolve().parent
sys.path.insert(0, str(NOTES))
import recompute_finite_size_artifact as R

JZZ = 1.0
B4, B6 = 6.0, 15.0
T = np.geomspace(3e-6, 0.3, 4000)   # low enough for small |Jpm| hexagon scale


def assemble_dressed(cl, pt, jpm, lam, mode):
    n = cl.n_ice
    H = np.zeros((n, n), dtype=float)
    for key, power, Bring in (("H2", 2, B4), ("H3", 3, B6)):
        rows = pt[key]
        if mode == "all":
            keep = np.ones(len(rows["c"]), dtype=bool)
        else:  # delta0
            keep = (R.transport_delta(cl, rows) == 0).all(axis=1)
        s = rows["s"][keep]; t = rows["t"][keep]; c = rows["c"][keep]
        offdiag = s != t
        coeff = c * jpm**power
        # E_g source softening on ring (off-diagonal) rows only
        coeff = coeff + offdiag * (c * Bring * jpm**(power - 1) * lam**2 / JZZ)
        np.add.at(H, (t, s), coeff)
    return 0.5 * (H + H.T)


def _peak_tracked(C, ref_k=None):
    """Lower gauge peak: local maxima, parabolic-refined; if ref_k given, pick the
    local max nearest ref_k in log-T (tracks the same feature across lambda)."""
    loc = [k for k in range(1, len(T) - 1) if C[k] > C[k - 1] and C[k] > C[k + 1]]
    if not loc:
        k = int(np.argmax(C))
    elif ref_k is None:
        k = max(loc, key=lambda k: C[k])            # most prominent at lambda=0
    else:
        k = min(loc, key=lambda k: abs(np.log(T[k]) - np.log(T[ref_k])))
    x = np.log(T[k - 1:k + 2]); y = C[k - 1:k + 2]
    den = y[0] - 2 * y[1] + y[2]
    tp = float(np.exp(x[1] + 0.5 * (y[0] - y[2]) / den * (x[1] - x[0]))) if abs(den) > 1e-30 else float(T[k])
    return tp, k


def tpk_series(cl, pt, jpm, mode, lams):
    """Tpk(lambda) tracking the lambda=0 gauge peak."""
    out = {}
    ref_k = None
    for lam in lams:
        H = assemble_dressed(cl, pt, jpm, lam, mode)
        C = R.specific_heat(np.linalg.eigvalsh(H), T)
        tp, k = _peak_tracked(C, ref_k)
        if ref_k is None:
            ref_k = k
        out[lam] = tp
    return out


def main():
    cl = R.build_cluster("cubic", (1, 1, 1))
    pt = R.sw_order23(cl, verbose=False)
    print(f"cubic 1x1x1: N={cl.n_sites} ice={cl.n_ice} "
          f"4-cycles={len(cl.loops4)} hexagons={len(cl.hexes)} "
          f"H2rows={len(pt['H2']['c'])} H3rows={len(pt['H3']['c'])}")
    grid = [round(x, 2) for x in np.arange(-0.10, 0.051, 0.01)]
    grid = [j for j in grid if abs(j) > 1e-9]
    LAMS = [0.0, 0.01, 0.02]
    print(f"\n{'Jpm':>7} {'flux':>5} | {'Tpk0_bare':>9} {'B_bare':>7} | "
          f"{'Tpk0_clean':>10} {'B_clean':>8}")
    res = {}
    for jpm in grid:
        row = {}
        for mode, tag in (("all", "bare"), ("delta0", "clean")):
            tp = tpk_series(cl, pt, jpm, mode, LAMS)
            kap = np.mean([(1 - tp[l] / tp[0.0]) / l**2 for l in LAMS[1:]])
            row[tag] = dict(tp0=tp[0.0], B=kap * abs(jpm) * JZZ)
        res[jpm] = row
        flux = "pi" if jpm < 0 else "0"
        print(f"{jpm:>7.2f} {flux:>5} | {row['bare']['tp0']:>9.5f} "
              f"{row['bare']['B']:>7.2f} | {row['clean']['tp0']:>10.6f} "
              f"{row['clean']['B']:>8.2f}", flush=True)
    np.savez_compressed(NOTES / "thermal_expansion_B_ice_manifold.npz",
                        jpms=np.array(grid),
                        B_bare=np.array([res[j]["bare"]["B"] for j in grid]),
                        B_clean=np.array([res[j]["clean"]["B"] for j in grid]),
                        tp0_bare=np.array([res[j]["bare"]["tp0"] for j in grid]),
                        tp0_clean=np.array([res[j]["clean"]["tp0"] for j in grid]))
    print("\nsaved thermal_expansion_B_ice_manifold.npz")


if __name__ == "__main__":
    main()
