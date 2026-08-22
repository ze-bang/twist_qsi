"""Zone-boundary landscape of f_1/S on the full allowed grids."""
import json
import sys
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parents[1] / "campaign" / "outputs" / "ring_model_dssf"


def fold(q):
    """Fold to the first zone: minimal image mod the cubic reciprocal
    cell of side 2 (fcc RLVs are (1,1,-1) type; components mod 2)."""
    v = (np.asarray(q) + 1.0) % 2.0 - 1.0
    return np.abs(v)


def star_of(q):
    a = np.sort(fold(q))
    for name, ref in (("G", [0, 0, 0]), ("L", [0.5, 0.5, 0.5]),
                      ("X", [0, 0, 1]), ("W", [0, 0.5, 1]),
                      ("K", [0, 0.75, 0.75]), ("U", [0.25, 0.25, 1])):
        if np.allclose(a, np.sort(np.abs(ref)), atol=1e-6):
            return name
    return None


for n in (6, 8):
    d = json.load(open(OUT / f"gfmc_fullgrid_n{n}.json"))
    print(f"\n=== fcc({n},{n},{n}): {d['n_sites']} sites, "
          f"E/site = {d['E_per_site']:.6f} ===")
    stars = {}
    for r in d["rows"]:
        if r["ratio"] is None:
            continue
        s = star_of(r["q"])
        if s:
            stars.setdefault(s, []).append(r)
    for s in ("L", "X", "W", "K", "U"):
        if s not in stars:
            continue
        rr = stars[s]
        vals = np.array([x["ratio"] for x in rr])
        errs = np.array([x["ratio_err"] for x in rr])
        w = 1.0 / errs ** 2
        m = float((vals * w).sum() / w.sum())
        e = float(1.0 / np.sqrt(w.sum()))
        sc = float(vals.std())
        print(f"  {s}: {len(rr):3d} grid points   ratio = {m:.4f} "
              f"+/- {e:.4f}   (scatter {sc:.4f})")
    # the L neighborhood along the diagonal
    print("  diagonal (q,q,q):")
    for r in sorted(d["rows"], key=lambda r: r["q"]):
        f = fold(r["q"])
        if abs(f[0] - f[1]) < 1e-6 and abs(f[1] - f[2]) < 1e-6 \
                and r["ratio"] is not None and f[0] > 0.24:
            print(f"    q = {f[0]:.4f}(1,1,1)   ratio = "
                  f"{r['ratio']:.4f} +/- {r['ratio_err']:.4f}")
    # global zone-boundary minimum: points with |fold| >= 0.4 each comp sum
    zb = [r for r in d["rows"] if r["ratio"] is not None
          and np.linalg.norm(fold(r["q"])) > 0.85]
    zb.sort(key=lambda r: r["ratio"])
    print("  five lowest ratios among |q| > 0.85 (zone boundary):")
    for r in zb[:5]:
        f = fold(r["q"])
        print(f"    [{f[0]:.3f} {f[1]:.3f} {f[2]:.3f}] ({star_of(r['q']) or '-'})"
              f"   ratio = {r['ratio']:.4f} +/- {r['ratio_err']:.4f}")
