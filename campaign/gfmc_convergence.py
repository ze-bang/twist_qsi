"""GFMC convergence scan: walker number and forward-walking window.

The referee asks that convergence be quantified rather than asserted.
On the 48-site cluster the exact answers are known
(E0 = -10.556012, S(L) = 0.370365), so a scan over the walker number
and the forward-walking window T is directly meaningful: every entry
can be compared with the exact value, and the equal-amplitude value
S = 0.297 marks where a broken forward walker would land.
"""

from __future__ import annotations

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


def main() -> int:
    t0 = time.perf_counter()
    templates = extract_templates()
    lat = build_lattice((2, 2, 3), templates)
    qL = [np.array([0.5, 0.5, -0.5])]
    print("exact: E0 = -10.556012, S(L) = 0.370365  "
          "(equal-amplitude S = 0.297)\n")
    print(" walkers  forward   E_mix        err       S(L)      err")
    for nw in (256, 512, 1024, 2048):
        for T in (20, 40, 80, 160):
            res, lam = G.gfmc(lat, n_walkers=nw, n_steps=20000,
                              n_equil=2000, forward=T, qs=qL, t0=None)
            E, Ee = G.binned(res["E_mix"])
            S, Se = G.binned(res["S"][0])
            print(f"  {nw:5d}    {T:4d}   {E:10.5f}  {Ee:.5f}   "
                  f"{S:.6f}  {Se:.6f}", flush=True)
    print(f"\ntotal {time.perf_counter()-t0:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
