#!/usr/bin/env python3
r"""Thermal-expansion B(Jpm) the QED way: full-microscopic 2-delta low-band
operator averaging WITH the E_g source, PBC (bare) vs loop-projected (clean).

For each Jpm and character point theta in {0,pi}^3 we build the 2-delta
character-deformed microscopic Hamiltonian (Eq. mic_dipole_twisted_new of the
notes) PLUS the uniform E_g source -lambda*sum_i(S^+_i+S^-_i) (rotated-real
representative), solve the lowest N_ice=90 states with QED, Loewdin-pull the band
back to the ice basis, and average the band operators over the eight corners
(clean) or take theta=0 (bare).  B = [1 - Tpk(lambda)/Tpk(0)]/lambda^2 * |Jpm|
from the lower C(T) peak of the (bare/clean) band spectrum.

This is the all-orders QED downfolding, not the leading Schrieffer-Wolff row
projection; B_clean is the dressed hexagon value (near, not exactly, 15).
"""
from __future__ import annotations
import argparse, json, sys, tempfile, time
from itertools import product
from pathlib import Path
import numpy as np, h5py

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import qed
import recompute_finite_size_artifact as R
import twist_resolved_qed_full_band as TQ

TEMPS = np.geomspace(1e-5, 0.12, 1400)


def solve_band(cl, jpm, phi, lam, ice_indices, sz_restrict, n_band=90):
    """Low-band operator H_B(theta,lambda) in the ice basis via QED + Loewdin."""
    op = TQ.build_qed_operator(cl, jpm, np.asarray(phi, float), "dipole2")
    if lam != 0.0:                      # uniform rotated-real E_g source (breaks Sz)
        for i in range(cl.n_sites):
            op.add_one_body(qed.OP_SPLUS, i, complex(-lam, 0.0))
            op.add_one_body(qed.OP_SMINUS, i, complex(-lam, 0.0))
    kw = dict(auto_sz=False, symmetry=None, spin_flip="off", time_reversal="off",
              num_eigenvalues=n_band, compute_eigenvectors=True, device="cpu",
              tolerance=1e-9, max_iterations=1500)
    if sz_restrict:
        kw.update(sz=cl.n_sites // 2, solver="full")
    else:
        kw.update(solver="krylov_schur")
    with tempfile.TemporaryDirectory() as tmp:
        t0 = time.time()
        res = qed.solve(op, output_dir=tmp, verbose=False, **kw)
        evals = np.asarray(res.eigenvalues, float)
        with h5py.File(res.eigenvectors_path, "r") as h5:
            vecs = [TQ.h5_vector_to_complex(h5[f"eigendata/eigenvector_{k}"])
                    for k in range(len(evals))]
        dt = time.time() - t0
    psi = np.column_stack(vecs)
    order = np.argsort(evals)
    evals, psi = evals[order], psi[:, order]
    if sz_restrict:
        idx = ice_indices          # rows already in the Sz=0 basis
    else:
        idx = np.asarray([int(s) for s in cl.ice_states], dtype=np.int64)  # full-basis id
    h_band, diag = TQ.band_operator_from_full_eigenvectors(evals, psi, idx)
    return h_band, dt, diag["ice_overlap_min"]


def tpk(h):
    ev = np.linalg.eigvalsh(0.5 * (h + h.conj().T))
    return R.refined_peak(TEMPS, R.specific_heat(ev, TEMPS))


def run_jpm(cl, jpm, ice_sz0, lam, corners):
    """Return (B_bare, B_clean, Tpk dict) for one Jpm."""
    out = {}
    for tag, lam_v in (("l0", 0.0), ("lam", lam)):
        sz = (lam_v == 0.0)
        bands = []
        for k, phi in enumerate(corners):
            idx = ice_sz0 if sz else None
            hb, dt, ov = solve_band(cl, jpm, phi, lam_v, ice_sz0, sz)
            bands.append(hb)
            print(f"    [{tag}] corner {k+1}/{len(corners)} phi/pi="
                  f"{tuple(round(x/np.pi,2) for x in phi)}  {dt:.0f}s ov={ov:.3f}", flush=True)
        out[tag] = dict(bare=bands[0], clean=sum(bands) / len(bands))
    Tpk = {m + "_" + t: tpk(out[t][m]) for t in ("l0", "lam") for m in ("bare", "clean")}
    def B(m):
        return (1 - Tpk[f"{m}_lam"] / Tpk[f"{m}_l0"]) / lam**2 * abs(jpm)
    return B("bare"), B("clean"), Tpk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jpms", type=float, nargs="+", default=[-0.05, 0.05])
    ap.add_argument("--lam", type=float, default=0.02)
    ap.add_argument("--ncorner", type=int, choices=[1, 8], default=8)
    ap.add_argument("--out", type=Path, default=HERE / "qed_thermal_expansion_B.json")
    args = ap.parse_args()

    cl = R.build_cluster("cubic", (1, 1, 1))
    sz_basis = TQ.fixed_sz_basis(cl.n_sites, cl.n_sites // 2)
    sz_index = {int(s): k for k, s in enumerate(sz_basis)}
    ice_sz0 = np.asarray([sz_index[int(s)] for s in cl.ice_states], dtype=np.int64)
    corners = [(0., 0., 0.)] if args.ncorner == 1 else \
        list(product([0.0, np.pi], repeat=3))

    results = {}
    if args.out.exists():
        results = json.loads(args.out.read_text())
    for jpm in args.jpms:
        key = f"{jpm:+.2f}"
        if key in results:
            print(f"skip {key} (cached)"); continue
        t0 = time.time()
        print(f"=== Jpm={key}  (g4={4*jpm**2:.5f}, ghex={12*abs(jpm)**3:.6f}) ===", flush=True)
        Bb, Bc, Tpk = run_jpm(cl, jpm, ice_sz0, args.lam, corners)
        results[key] = dict(jpm=jpm, B_bare=Bb, B_clean=Bc, Tpk=Tpk,
                            lam=args.lam, ncorner=args.ncorner)
        args.out.write_text(json.dumps(results, indent=2))
        print(f"  Jpm={key}: B_bare={Bb:+.2f}  B_clean={Bc:+.2f}   "
              f"({time.time()-t0:.0f}s)\n", flush=True)
    print("done ->", args.out)


if __name__ == "__main__":
    main()
