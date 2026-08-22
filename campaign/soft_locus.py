"""Where is the spectrum soft, besides L?  The full landscape.

Sorts every allowed momentum of the 864- and 2048-site grids by the
Feynman ratio, after excising the small-|q| photon regime where the
ratio tends to zero trivially.  The question: is the softness an
isolated point at L, or a connected region -- and if a region, what
is its geometry?  Also prints the (t,1/2,1/2) line containing the
72-site companion momentum (1/6,1/2,1/2).
"""
import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parents[1] / "campaign" / "outputs" / "ring_model_dssf"


def fold(q):
    """Fold into the first fcc Brillouin zone: minimize |q| over
    reciprocal-lattice translations (RLVs = integer combos of
    (1,1,-1),(1,-1,1),(-1,1,1) in cubic units)."""
    G = []
    for a in (-2, -1, 0, 1, 2):
        for b in (-2, -1, 0, 1, 2):
            for c in (-2, -1, 0, 1, 2):
                G.append(a * np.array([1, 1, -1]) + b * np.array([1, -1, 1])
                         + c * np.array([-1, 1, 1]))
    q = np.asarray(q, dtype=float)
    best = None
    for g in G:
        v = q - g
        if best is None or np.linalg.norm(v) < np.linalg.norm(best) - 1e-12:
            best = v
    return np.sort(np.abs(best))


for n in (6, 8):
    d = json.load(open(OUT / f"gfmc_fullgrid_n{n}.json"))
    rows = [r for r in d["rows"] if r["ratio"] is not None]
    for r in rows:
        r["qf"] = fold(r["q"])
        r["absq"] = float(np.linalg.norm(r["qf"]))
    # merge symmetry copies (same folded |components|)
    merged = {}
    for r in rows:
        k = tuple(np.round(r["qf"], 6))
        merged.setdefault(k, []).append(r)
    stars = []
    for k, rr in merged.items():
        w = np.array([1 / x["ratio_err"] ** 2 for x in rr])
        v = np.array([x["ratio"] for x in rr])
        stars.append({"qf": np.array(k), "absq": rr[0]["absq"],
                      "ratio": float((v * w).sum() / w.sum()),
                      "err": float(1 / np.sqrt(w.sum())),
                      "copies": len(rr)})
    big = [s for s in stars if s["absq"] > 0.72]   # exclude photon regime
    big.sort(key=lambda s: s["ratio"])
    print(f"\n=== fcc({n}): {len(stars)} stars, {len(big)} with |q|>0.72 ===")
    print("   folded |q| components      |q|    ratio        copies")
    for s in big[:14]:
        q = s["qf"]
        print(f"  [{q[0]:6.3f} {q[1]:6.3f} {q[2]:6.3f}]  {s['absq']:.3f}"
              f"   {s['ratio']:.4f} +/- {s['err']:.4f}   {s['copies']}")
    print("   ... highest:")
    for s in big[-3:]:
        q = s["qf"]
        print(f"  [{q[0]:6.3f} {q[1]:6.3f} {q[2]:6.3f}]  {s['absq']:.3f}"
              f"   {s['ratio']:.4f} +/- {s['err']:.4f}   {s['copies']}")
    if n == 8:
        print("\n  the (t, 1/2, 1/2) line (companion of the 72-site test):")
        for s in sorted(stars, key=lambda s: s["absq"]):
            q = np.sort(s["qf"])
            if abs(q[1] - 0.5) < 1e-6 and abs(q[2] - 0.5) < 1e-6:
                print(f"    t = {q[0]:.3f}   ratio = {s['ratio']:.4f} "
                      f"+/- {s['err']:.4f}")
