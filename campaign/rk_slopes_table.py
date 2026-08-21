"""The Rokhsar-Kivelson slopes, recomputed from scratch and tabulated.

H(V) = -K sum_p (W_p + W_p^dag) + V sum_p n_p^flip.

n_p^flip is the projector onto hexagons whose six spins alternate (the
flippable ones); sum_p n_p^flip simply COUNTS them.  By Hellmann-Feynman
the slope of a level is exactly the change in that count:

    d omega_n / dV = <n| sum_p n_p^flip |n> - <0| sum_p n_p^flip |0>,

so the slope is a census of flippable hexagons, not a fitted parameter.
Here we (i) compute the slope directly as that expectation-value
difference at V=0, and (ii) confirm it by a finite-difference fit of the
tracked level across V in [-0.5, +0.5] -- two independent routes.

Reported for the two dominant (branch-resolved) lines at every momentum,
which is what fixes the photon range quoted in the paper.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "notes"))
sys.path.insert(0, str(ROOT / "campaign"))

import recompute_finite_size_artifact as geometry               # noqa: E402
from run_momentum_resolved_spectrum import star                 # noqa: E402
from run_ring48_dssf import contractible                        # noqa: E402
from ring48_momenta_fix import fcc_momenta                      # noqa: E402
from ring64_pipeline import (fast_labels, fast_ring,            # noqa: E402
                             hex_masks_patterns, probe_columns)

OUT = ROOT / "campaign" / "outputs" / "ring_model_dssf"
VLIST = [-0.5, -0.25, 0.0, 0.25, 0.5]
EDEGEN = 1e-8


def nflip_diag(states, hmp):
    """The diagonal operator sum_p n_p^flip: how many hexagons of each
    configuration are flippable."""
    n = np.zeros(len(states))
    for mask, p1, p2 in hmp:
        sel = states & mask
        n += ((sel == p1) | (sel == p2)).astype(float)
    return n


def main() -> int:
    t0 = time.perf_counter()
    cl = geometry.build_cluster("fcc", (2, 2, 3))
    n = int(cl.n_sites)
    states = np.sort(np.asarray(cl.ice_states, dtype=np.uint64))
    positions = np.asarray(cl.positions, dtype=float)
    labels, spins_all = fast_labels(states, positions)
    hexes = contractible(cl)
    hmp = hex_masks_patterns(hexes)
    momenta, _ = fcc_momenta(cl)

    ham, _ = fast_ring(states, hmp, 0.0)
    _, v0 = eigsh(ham, k=1, which="SA", tol=1e-11, ncv=60)
    per = {int(s): float(np.sum(v0[labels == s, 0] ** 2))
           for s in np.unique(labels)}
    sector = max(per, key=per.get)
    rows = np.where(labels == sector)[0]
    hs = np.asarray(ham[np.ix_(rows, rows)].todense())
    values, vectors = eigh(hs)
    e0 = float(values[0])
    omega = values - e0
    gs = vectors[:, 0].astype(complex)

    nf = nflip_diag(states[rows], hmp)
    nf_gs = float(np.sum(np.abs(gs) ** 2 * nf))
    print(f"sector {sector} dim {len(rows)}   "
          f"<n_flip> ground = {nf_gs:.4f} of {len(hexes)} hexagons")

    # track the two dominant lines per momentum through the V scan
    tracks = {}
    for v in VLIST:
        hv, _ = fast_ring(states, hmp, v)
        hsv = np.asarray(hv[np.ix_(rows, rows)].todense())
        val, vec = eigh(hsv)
        g = vec[:, 0].astype(complex)
        om = val - val[0]
        for qi, q in enumerate(momenta):
            lab = star(q)
            w = np.zeros(len(val))
            for col in probe_columns(spins_all[rows], positions, q, g, n):
                w += np.abs(vec.conj().T @ col) ** 2 / n
            if w.sum() < 1e-10:
                continue
            grp = []
            i = 0
            while i < len(val):
                j = i
                while j + 1 < len(val) and val[j + 1] - val[i] < EDEGEN:
                    j += 1
                grp.append((float(om[i]), float(w[i:j + 1].sum())))
                i = j + 1
            top = sorted(grp, key=lambda p: -p[1])[:2]
            top.sort()
            tracks.setdefault(lab, {}).setdefault(v, top)

    print("\nslopes of the two dominant lines "
          "(Hellmann-Feynman census vs finite-difference fit):")
    print("  momentum              branch  omega(V=0)  d<n_flip>   "
          "fit slope")
    rec = []
    for lab in sorted(tracks):
        vs = tracks[lab]
        if not all(v in vs and len(vs[v]) == 2 for v in VLIST):
            continue
        for b in (0, 1):
            es = [vs[v][b][0] for v in VLIST]
            fit = float(np.polyfit(VLIST, es, 1)[0])
            # direct census for the same level at V = 0
            e00 = vs[0.0][b][0]
            j = int(np.argmin(np.abs(omega - e00)))
            grp = np.where(np.abs(values - values[j]) < EDEGEN)[0]
            cen = float(np.mean([np.sum(np.abs(vectors[:, k]) ** 2 * nf)
                                 for k in grp])) - nf_gs
            print(f"  {lab:<20s}  {b+1}      {e00:7.4f}   "
                  f"{cen:+8.4f}   {fit:+8.4f}")
            rec.append({"star": lab, "branch": b + 1, "omega": e00,
                        "census": cen, "fit": fit})

    ph = [r for r in rec if not (r["star"] == "L" and r["branch"] == 1)]
    ro = [r for r in rec if r["star"] == "L" and r["branch"] == 1]
    if ph:
        lo = min(r["fit"] for r in ph)
        hi = max(r["fit"] for r in ph)
        print(f"\nphoton-like lines: slope range [{lo:+.3f}, {hi:+.3f}]")
    if ro:
        print(f"roton (L, lower branch): slope {ro[0]['fit']:+.3f} "
              f"(census {ro[0]['census']:+.3f})")
    with open(OUT / "rk_slopes.json", "w") as fh:
        json.dump({"nflip_ground": nf_gs, "n_hex": len(hexes),
                   "rows": rec}, fh, indent=1)
    print(f"\nwrote rk_slopes.json ({time.perf_counter()-t0:.0f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
