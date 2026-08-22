"""The referee's request: f_1/S on large lattices and a dense momentum grid.

Runs GFMC on a cubic-symmetric pyrochlore supercell and measures
S(q) along a dense Brillouin-zone path plus the full grid of allowed
momenta.  f_1(q) then follows with no further statistics: on these
lattices every hexagon is symmetry-equivalent, so the exact sum rule
sum_p <W_p+W_p^dag> = -E_0/K fixes the uniform plaquette coherence,
and f_1 is that number times an exactly computable geometric factor.

The output is the Feynman ratio f_1(q)/S(q) across the zone: the
ground-state predictor whose minimum, if it sits at L and deepens or
persists with system size, is what the roton claim needs -- and whose
absence would refute it.  Either answer is a result.

Usage: gfmc_production.py n1 n2 n3 n_steps
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
import gfmc_ice as G                                            # noqa: E402

OUT = ROOT / "campaign" / "outputs" / "ring_model_dssf"


def log(m, t0):
    print(f"[{time.perf_counter()-t0:8.1f}s] {m}", flush=True)


def bz_path(n_per=12):
    """Gamma -> X -> W -> L -> Gamma -> K -> X in cubic reciprocal
    units, plus the L-centred segments the roton claim cares about."""
    pts = {"G": [0, 0, 0], "X": [0, 0, 1], "W": [0.5, 0, 1],
           "L": [0.5, 0.5, 0.5], "K": [0.75, 0.75, 0]}
    legs = [("G", "X"), ("X", "W"), ("W", "L"), ("L", "G"),
            ("G", "K"), ("K", "X")]
    qs, labels = [], []
    for a, b in legs:
        pa, pb = np.array(pts[a], float), np.array(pts[b], float)
        for i in range(n_per):
            qs.append(pa + (pb - pa) * i / n_per)
            labels.append(a if i == 0 else "")
    qs.append(np.array(pts["X"], float))
    labels.append("X")
    return np.array(qs), labels


def geom_factor(lat, qs):
    """sum_p sum_mu |g^mu_p(q)|^2 for each q, exactly."""
    out = np.zeros(len(qs))
    sub = lat.sublattice
    for qi, q in enumerate(qs):
        phase = np.exp(-2j * np.pi * (lat.positions @ q))
        tot = 0.0
        for path in lat.hexes:
            sites = np.asarray(path, dtype=int)
            signs = np.where(np.arange(len(sites)) % 2 == 0, 1.0, -1.0)
            z = signs * phase[sites]
            for mu in range(4):
                sel = sub[sites] == mu
                if sel.any():
                    tot += abs(z[sel].sum()) ** 2
        out[qi] = tot
    return out


def main() -> int:
    t0 = time.perf_counter()
    shape = tuple(int(x) for x in sys.argv[1:4])
    n_steps = int(sys.argv[4]) if len(sys.argv) >= 5 else 30000

    templates = extract_templates()
    lat = build_lattice(shape, templates)
    qs, labels = bz_path()
    log(f"fcc{shape}: {lat.n_sites} sites, {len(lat.hexes)} hexagons, "
        f"{len(qs)} path momenta", t0)

    res, lam = G.gfmc(lat, n_walkers=1024, n_steps=n_steps,
                      n_equil=max(2000, n_steps // 10), forward=80,
                      qs=list(qs), t0=t0)

    Em, Ee = G.binned(res["E_mix"])
    nf, nfe = G.binned(res["nflip"])
    log(f"\nE_mix = {Em:.4f} +/- {Ee:.4f} K   per site "
        f"{Em/lat.n_sites:.6f} +/- {Ee/lat.n_sites:.6f}", t0)
    log(f"<n_flip> = {nf:.2f} +/- {nfe:.2f} of {len(lat.hexes)} "
        f"({nf/len(lat.hexes):.4f} per hexagon)", t0)

    # exact sum rule: uniform plaquette coherence
    w_p = -Em / (1.0 * len(lat.hexes))
    gf = geom_factor(lat, qs)
    f1 = w_p * gf / (2.0 * lat.n_sites)

    log("\n   q (cubic units)        label   S(q)        err       f1"
        "        f1/S", t0)
    rows = []
    for qi, q in enumerate(qs):
        S, Serr = G.binned(res["S"][qi])
        ratio = f1[qi] / S if S > 1e-10 else float("nan")
        rerr = ratio * Serr / S if S > 1e-10 else 0.0
        log(f"  [{q[0]:6.3f} {q[1]:6.3f} {q[2]:6.3f}]  {labels[qi]:>2s}  "
            f"{S:9.5f}  {Serr:8.5f}  {f1[qi]:8.5f}  {ratio:8.4f} "
            f"+/- {rerr:.4f}", t0)
        rows.append({"q": [float(x) for x in q], "label": labels[qi],
                     "S": S, "S_err": Serr, "f1": float(f1[qi]),
                     "ratio": ratio, "ratio_err": rerr})

    tag = "".join(str(s) for s in shape)
    with open(OUT / f"gfmc_path_fcc{tag}.json", "w") as fh:
        json.dump({"shape": list(shape), "n_sites": lat.n_sites,
                   "E_mix": Em, "E_err": Ee, "E_per_site": Em/lat.n_sites,
                   "w_p": w_p, "nflip": nf, "n_steps": n_steps,
                   "rows": rows}, fh, indent=1)
    log(f"wrote gfmc_path_fcc{tag}.json", t0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
