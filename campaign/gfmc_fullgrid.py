"""S(q) and f_1/S on the COMPLETE allowed momentum grid of a cubic box.

The path scan proved the L minimum; this produces the definitive data
set: every allowed momentum of the torus, where S(q) is a true
eigenmode weight with no interpolation artifact.  On fcc(n,n,n) the
allowed momenta are q = v/n in cubic reciprocal units with v an
integer vector whose components share one parity (v = m1 b1 + m2 b2 +
m3 b3 with b1=(1,1,-1) etc.), giving n^3 distinct points -- 512 at
n=8.  The measurement is one GEMM per snapshot regardless of how many
momenta are tabulated, so the full grid costs no more than the path.

Usage: gfmc_fullgrid.py n n_steps
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "notes"))
sys.path.insert(0, str(ROOT / "campaign"))

from pyro_lattice import build_lattice, extract_templates       # noqa: E402
from gfmc_production import geom_factor                         # noqa: E402
import gfmc_ice as G                                            # noqa: E402

OUT = ROOT / "campaign" / "outputs" / "ring_model_dssf"


def log(m, t0):
    print(f"[{time.perf_counter()-t0:8.1f}s] {m}", flush=True)


def allowed_grid(n):
    """All distinct allowed momenta of fcc(n,n,n), folded to the cell
    v in [0, n)^3 of the reciprocal supercell, in cubic units."""
    qs = []
    for m1 in range(n):
        for m2 in range(n):
            for m3 in range(n):
                v = np.array([m1 + m2 - m3, m1 - m2 + m3,
                              -m1 + m2 + m3])
                qs.append(((v % (2 * n)) / n))   # fold by RLV of 2
    qs = np.unique(np.round(np.array(qs), 9), axis=0)
    return qs


def main() -> int:
    t0 = time.perf_counter()
    n = int(sys.argv[1])
    n_steps = int(sys.argv[2]) if len(sys.argv) >= 3 else 60000

    templates = extract_templates()
    lat = build_lattice((n, n, n), templates)
    qs = allowed_grid(n)
    log(f"fcc({n},{n},{n}): {lat.n_sites} sites, {len(qs)} allowed "
        f"momenta", t0)

    res, lam = G.gfmc(lat, n_walkers=1024, n_steps=n_steps,
                      n_equil=max(3000, n_steps // 10), forward=80,
                      qs=list(qs), t0=t0)

    Em, Ee = G.binned(res["E_mix"])
    log(f"E_mix = {Em:.4f} +/- {Ee:.4f} K  per site "
        f"{Em/lat.n_sites:.6f} +/- {Ee/lat.n_sites:.6f}", t0)
    w_p = -Em / (1.0 * len(lat.hexes))
    gf = geom_factor(lat, qs)
    f1 = w_p * gf / (2.0 * lat.n_sites)

    rows = []
    for qi, q in enumerate(qs):
        S, Serr = G.binned(res["S"][qi])
        ratio = f1[qi] / S if S > 1e-10 else None
        rerr = ratio * Serr / S if S > 1e-10 else None
        rows.append({"q": [float(x) for x in q], "S": S, "S_err": Serr,
                     "f1": float(f1[qi]), "ratio": ratio,
                     "ratio_err": rerr})
    good = [r for r in rows if r["ratio"] is not None]
    good.sort(key=lambda r: r["ratio"])
    log("\nlowest fifteen f1/S on the allowed grid:", t0)
    for r in good[:15]:
        log(f"  q = [{r['q'][0]:6.3f} {r['q'][1]:6.3f} {r['q'][2]:6.3f}]"
            f"   S = {r['S']:.5f}({r['S_err']:.5f})   "
            f"ratio = {r['ratio']:.4f} +/- {r['ratio_err']:.4f}", t0)

    with open(OUT / f"gfmc_fullgrid_n{n}.json", "w") as fh:
        json.dump({"n": n, "n_sites": lat.n_sites, "E_mix": Em,
                   "E_err": Ee, "E_per_site": Em / lat.n_sites,
                   "w_p": w_p, "n_steps": n_steps, "rows": rows},
                  fh, indent=1)
    log(f"wrote gfmc_fullgrid_n{n}.json", t0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
