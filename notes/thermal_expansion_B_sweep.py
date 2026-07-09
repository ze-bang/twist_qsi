#!/usr/bin/env python3
r"""Thermal-expansion coefficient B(Jpm) vs lambda on the 16-site cubic cluster:
PBC (bare) vs the transported-dipole loop projection (clean) of
finite_size_loop_projection_notes.tex.

We build the Schrieffer-Wolff ice-manifold gauge Hamiltonian (order 2 = winding
four-loops, order 3 = hexagons) with recompute_finite_size_artifact.py, then turn
on the E_g source lambda, which softens each ring coupling at second order:
    order-2 (four-loop):  g4  -> g4 (1 - kappa4 lambda^2),  kappa4 = -6 /(Jpm Jzz)
    order-3 (hexagon):    ghex-> ghex(1 - kappa6 lambda^2), kappa6 = -15/(Jpm Jzz)
(the signed B_k = kappa_k |Jpm| Jzz = -B_k^0 sign(Jpm), B4^0=6, B6^0=15).

The lower (gauge) C(T) peak gives Tpk(lambda); B = [1-Tpk(l)/Tpk(0)]/l^2 * |Jpm|.

- PBC keeps H2+H3  -> four-loop-contaminated B (messy, sign-changing).
- clean keeps delta=0 rows only -> winding four-loops gone, hexagon-only
  -> B -> 15 (sign locked to flux).  This is the loop correction.
"""
from __future__ import annotations
import importlib.util, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("recomp", HERE / "recompute_finite_size_artifact.py")
R = importlib.util.module_from_spec(spec)
sys.modules["recomp"] = R          # needed so @dataclass can resolve __module__
spec.loader.exec_module(R)

JZZ = 1.0
B4_0, B6_0 = 6.0, 15.0
_T = np.logspace(-6.5, -0.6, 13000)


def assemble_src(cl, pt, jpm, lam, mode):
    """Effective ice-manifold H with the E_g source softening of each ring order."""
    kappa = {2: -B4_0 / (jpm * JZZ), 3: -B6_0 / (jpm * JZZ)}
    H = np.zeros((cl.n_ice, cl.n_ice))
    for key, power in (("H2", 2), ("H3", 3)):
        rows = pt[key]
        if len(rows["c"]) == 0:
            continue
        if mode == "all":
            keep = np.ones(len(rows["c"]), dtype=bool)
        else:  # delta0 : transported-dipole zero (loop projection)
            keep = (R.transport_delta(cl, rows) == 0).all(axis=1)
        soft = 1.0 - kappa[power] * lam**2
        vals = (jpm ** power) * soft * rows["c"][keep]
        np.add.at(H, (rows["t"][keep], rows["s"][keep]), vals)
    return 0.5 * (H + H.T)


def gauge_peak(E):
    C = R.specific_heat(E, _T)
    return R.refined_peak(_T, C)


def main():
    cl = R.build_cluster("cubic", (1, 1, 1))
    pt = R.sw_order23(cl, verbose=False)
    print(f"cluster: {cl.n_sites} sites, {cl.n_ice} ice states, "
          f"H2 rows={len(pt['H2']['c'])}, H3 rows={len(pt['H3']['c'])}")
    grid = [round(x, 2) for x in np.arange(-0.10, 0.051, 0.01)]
    grid = [j for j in grid if abs(j) > 1e-9]
    lam = 0.01
    print(f"\n{'Jpm':>6} {'Tpk0_bare':>10} {'B_bare':>9} | {'Tpk0_clean':>11} {'B_clean':>9}")
    out = {}
    for jpm in grid:
        res = {}
        for mode in ("all", "delta0"):
            tp0 = gauge_peak(np.linalg.eigvalsh(assemble_src(cl, pt, jpm, 0.0, mode)))
            tpl = gauge_peak(np.linalg.eigvalsh(assemble_src(cl, pt, jpm, lam, mode)))
            B = (1 - tpl / tp0) / lam**2 * abs(jpm) * JZZ
            res[mode] = (tp0, B)
        out[jpm] = res
        print(f"{jpm:>6.2f} {res['all'][0]:>10.6f} {res['all'][1]:>9.3f} | "
              f"{res['delta0'][0]:>11.6f} {res['delta0'][1]:>9.3f}", flush=True)
    np.savez_compressed(HERE / "thermal_expansion_B_sweep.npz",
                        jpms=np.array(grid),
                        B_bare=np.array([out[j]["all"][1] for j in grid]),
                        B_clean=np.array([out[j]["delta0"][1] for j in grid]),
                        tp0_bare=np.array([out[j]["all"][0] for j in grid]),
                        tp0_clean=np.array([out[j]["delta0"][0] for j in grid]))
    print("\nsaved thermal_expansion_B_sweep.npz")


if __name__ == "__main__":
    main()
