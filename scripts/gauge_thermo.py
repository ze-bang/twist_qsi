"""Gauge-sector thermodynamics of quantum spin ice from the transport-clean
effective Hamiltonian.

Build (or load) the Schrieffer-Wolff ring-exchange Hamiltonian in the ice
manifold once at Jpm=1 (couplings scale as Jpm^k exactly), then evaluate
exact full-spectrum thermodynamics -- C(T), S(T), the low-T (gauge) peak,
and the entropy plateau -- over a grid of couplings and both flux signs,
for plain periodic boundaries ('all') and the zero-transport projector
('delta0').

This is the workhorse for the 0-flux QMC-benchmark and the pi-flux payoff:
  * 16-site cubic : ice = 90    (validation vs exact ED)
  * 32-site FCC   : ice = 2970  (production; dense full spectrum ~ seconds)
  * 48-site FCC   : ice = 87546 (stretch; --lanczos for low spectrum only)

Usage
-----
  # one-time build (expensive part; ~4 min at 32 sites, ~hours at 48):
  python gauge_thermo.py build --basis fcc --shape 2 2 2 --order 4 --out rows_fcc222_o4.npz
  # sweep (cheap; reuses the row table):
  python gauge_thermo.py sweep --rows rows_fcc222_o4.npz \
        --jpm -0.03 -0.05 -0.08 0.03 0.046 0.05 --modes all delta0 --order 4 \
        --out thermo_fcc222.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh

import ice_pt_lib as ipl


# ----------------------------------------------------------------------------
def entropy_from_spectrum(E, T, n_sites):
    """S(T)/N from the full (or low) spectrum, per site, in units of k_B."""
    E = np.sort(np.asarray(E, float))
    E = E - E[0]
    beta = 1.0 / T[:, None]
    w = np.exp(-beta * E[None, :])
    Z = w.sum(axis=1)
    U = (w * E[None, :]).sum(axis=1) / Z
    F = -T * np.log(Z)
    return (U - F) / T / n_sites


def refined_peak(T, C):
    k = int(np.argmax(C))
    if 0 < k < len(T) - 1:
        x = np.log(T[k - 1:k + 2]); y = C[k - 1:k + 2]
        den = y[0] - 2 * y[1] + y[2]
        if abs(den) > 1e-30:
            return float(np.exp(x[1] + 0.5 * (y[0] - y[2]) / den * (x[1] - x[0]))), float(y.max())
    return float(T[k]), float(C[k])


# ----------------------------------------------------------------------------
def cmd_build(args):
    cl = ipl.build_cluster(args.basis, tuple(args.shape))
    print(f"cluster {args.basis}{tuple(args.shape)}: N={cl.n_sites} ice={cl.n_ice} "
          f"loops4={len(cl.loops4)} hex={len(cl.hexes)}", flush=True)
    t0 = time.time()
    pt = ipl.sw_effective(cl, 1.0, order=args.order, verbose=True)
    print(f"SW order-{args.order} build: {time.time()-t0:.0f}s "
          f"(H2 {len(pt['H2']['c'])}, H3 {len(pt['H3']['c'])}"
          f"{', H4 '+str(len(pt['H4']['c'])) if 'H4' in pt else ''} rows)", flush=True)
    ipl.save_rows(args.out, pt, cl, args.order)
    print(f"saved {args.out} ({Path(args.out).stat().st_size/1e6:.1f} MB)")


def cmd_sweep(args):
    pt, meta = ipl.load_rows(args.rows)
    cl = ipl.build_cluster(meta["basis"], meta["shape"])
    assert cl.n_ice == meta["n_ice"]
    T = np.geomspace(args.tmin, args.tmax, args.nt)
    order = args.order or meta["order"]
    lanczos = args.lanczos and cl.n_ice > args.lanczos_above

    results = {"meta": meta, "order": order, "T": T.tolist(),
               "n_sites": cl.n_sites, "curves": []}
    for J in args.jpm:
        for mode in args.modes:
            t0 = time.time()
            M = ipl.assemble(cl, pt, J, order, mode=mode)
            if lanczos:
                Msp = csr_matrix(np.real(M))
                E = np.sort(eigsh(Msp, k=args.lanczos_k, which="SA",
                                  return_eigenvectors=False, tol=1e-9))
            else:
                E = np.linalg.eigvalsh(M)
            E = E - E[0]
            C = ipl.C_of_T(E, T)
            Tpk, Cpk = refined_peak(T, C)
            S = None if lanczos else entropy_from_spectrum(E, T, cl.n_sites)
            ghex = 12 * abs(J) ** 3
            g4 = 4 * J ** 2
            rec = dict(Jpm=J, mode=mode, Tpk=Tpk, Cpk=Cpk,
                       Tpk_over_ghex=Tpk / ghex, ghex=ghex, g4=g4,
                       C=C.tolist(),
                       S=(S.tolist() if S is not None else None),
                       gap=float(E[1]) if len(E) > 1 else 0.0,
                       lanczos=bool(lanczos))
            results["curves"].append(rec)
            print(f"Jpm={J:+.3f} {mode:>7}: Tpk={Tpk:.5f} "
                  f"(Tpk/ghex={Tpk/ghex:.2f}) gap={E[1]:.2e} "
                  f"[{time.time()-t0:.0f}s]", flush=True)
    Path(args.out).write_text(json.dumps(results))
    print(f"saved {args.out}")


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build")
    b.add_argument("--basis", required=True, choices=["cubic", "fcc"])
    b.add_argument("--shape", type=int, nargs=3, required=True)
    b.add_argument("--order", type=int, default=4)
    b.add_argument("--out", required=True)
    b.set_defaults(func=cmd_build)

    s = sub.add_parser("sweep")
    s.add_argument("--rows", required=True)
    s.add_argument("--jpm", type=float, nargs="+", required=True)
    s.add_argument("--modes", nargs="+", default=["all", "delta0"])
    s.add_argument("--order", type=int, default=None)
    s.add_argument("--tmin", type=float, default=2e-4)
    s.add_argument("--tmax", type=float, default=0.3)
    s.add_argument("--nt", type=int, default=600)
    s.add_argument("--lanczos", action="store_true",
                   help="use Lanczos low-spectrum (needed for 48-site)")
    s.add_argument("--lanczos-above", type=int, default=10000)
    s.add_argument("--lanczos-k", type=int, default=1500)
    s.add_argument("--out", required=True)
    s.set_defaults(func=cmd_sweep)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
