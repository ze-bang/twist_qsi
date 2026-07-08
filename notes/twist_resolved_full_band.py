#!/usr/bin/env python3
"""Twist-resolved low-energy band from the full 16-site XXZ Hamiltonian.

This is a proof-of-principle implementation of the full-Hamiltonian version
of the finite-size-loop removal idea:

    full H(phi) -> lowest ice-like band -> H_band(phi) in fixed ice basis
    -> operator average over smooth twists.

It intentionally reuses only the standalone geometry/ice helpers in
``recompute_finite_size_artifact.py``.
"""
from __future__ import annotations

import argparse
import json
import time
from itertools import combinations, product
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import eigsh, lobpcg

import recompute_finite_size_artifact as R

HERE = Path(__file__).resolve().parent


def sz0_basis(n_sites: int) -> np.ndarray:
    states = []
    for occ in combinations(range(n_sites), n_sites // 2):
        s = 0
        for i in occ:
            s |= 1 << i
        states.append(s)
    return np.array(states, dtype=np.uint64)


def full_hamiltonian_sparse(cl: R.Cluster, states: np.ndarray, jpm: float, phi) -> coo_matrix:
    """Full microscopic Hamiltonian in the fixed Sz=0 basis.

    The Ising energy is shifted so all ice states have E=0:
        H0 = 0.5 * sum_t (n_up_t - 2)^2 .
    Smooth twist: A = phi L^{-1}, coefficient for S_i^+ S_j^- is
        -Jpm exp[-i A . d_ij],
    where d_ij is the minimum-image displacement from i to j.
    """
    index = {int(s): i for i, s in enumerate(states)}
    dim = len(states)
    phi = np.asarray(phi, dtype=float)
    A = phi @ np.linalg.inv(cl.Lvecs)

    rows = []
    cols = []
    vals = []

    # diagonal Ising energy relative to the ice manifold
    diag = R.ising_energy(cl, states)
    rows.extend(range(dim))
    cols.extend(range(dim))
    vals.extend(diag.astype(complex))

    one = np.uint64(1)
    for col, st in enumerate(states):
        for (i, j), n in zip(cl.bonds, cl.bond_wrap):
            bi = one << np.uint64(i)
            bj = one << np.uint64(j)
            up_i = (st & bi) != 0
            up_j = (st & bj) != 0
            if up_i == up_j:
                continue
            d_ij = cl.positions[j] - cl.positions[i] - np.asarray(n) @ cl.Lvecs
            phase = np.exp(-1j * float(A @ d_ij))
            new = int(st ^ (bi | bj))
            row = index[new]
            if (not up_i) and up_j:
                # S_i^+ S_j^- : raise i, lower j
                amp = -jpm * phase
            else:
                # S_i^- S_j^+ : Hermitian conjugate process
                amp = -jpm * np.conjugate(phase)
            rows.append(row)
            cols.append(col)
            vals.append(amp)
    return coo_matrix((np.asarray(vals, complex), (rows, cols)), shape=(dim, dim)).tocsr()


def band_hamiltonian(cl, states, ice_indices, jpm, phi, n_band, tol, sigma, solver, maxiter):
    H = full_hamiltonian_sparse(cl, states, jpm, phi)
    t0 = time.time()
    if solver == "eigsh":
        E, Psi = eigsh(
            H,
            k=n_band,
            sigma=sigma,
            which="LM",
            tol=tol,
            ncv=max(2 * n_band + 20, 220),
        )
    elif solver == "lobpcg":
        X0 = np.zeros((H.shape[0], n_band), dtype=complex)
        for a, idx in enumerate(ice_indices[:n_band]):
            X0[idx, a] = 1.0
        rng = np.random.default_rng(12345)
        X0 += 1e-6 * (rng.standard_normal(X0.shape) + 1j * rng.standard_normal(X0.shape))
        E, Psi = lobpcg(H, X0, largest=False, tol=tol, maxiter=maxiter)
    else:
        raise ValueError(solver)
    order = np.argsort(E)
    E = E[order]
    Psi = Psi[:, order]
    elapsed = time.time() - t0

    X = Psi[ice_indices, :]  # fixed ice basis x low eigenstates
    S = X.conj().T @ X
    s_eval, U = np.linalg.eigh(0.5 * (S + S.conj().T))
    if np.min(s_eval) <= 1e-12:
        raise RuntimeError(f"low-band ice projection is singular: min eval {np.min(s_eval)}")
    Sinvhalf = U @ np.diag(1.0 / np.sqrt(s_eval)) @ U.conj().T
    Q = X @ Sinvhalf
    Hband = Q @ np.diag(E) @ Q.conj().T
    Hband = 0.5 * (Hband + Hband.conj().T)
    return Hband, E, {
        "phi": [float(x) for x in phi],
        "elapsed_s": elapsed,
        "E_min": float(E[0]),
        "E_max_band": float(E[-1]),
        "ice_overlap_min": float(np.min(s_eval)),
        "ice_overlap_max": float(np.max(s_eval)),
        "ice_overlap_mean": float(np.mean(s_eval)),
    }


def specific_heat_from_matrix(H, T):
    E = np.linalg.eigvalsh(H)
    return E, R.specific_heat(E, T)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jpm", type=float, default=-0.05)
    ap.add_argument("--grid", choices=["two", "one"], default="two",
                    help="'two' = {0,pi}^3; 'one' = phi=0 only")
    ap.add_argument("--tol", type=float, default=1e-9)
    ap.add_argument("--sigma", type=float, default=-1.0,
                    help="shift-invert target placed below the low spectrum")
    ap.add_argument("--solver", choices=["lobpcg", "eigsh"], default="lobpcg")
    ap.add_argument("--maxiter", type=int, default=250)
    ap.add_argument("--n-band", type=int, default=None)
    ap.add_argument("--out", type=Path, default=HERE / "twist_resolved_full_band_jm0p05.npz")
    args = ap.parse_args()

    cl = R.build_cluster("cubic", (1, 1, 1))
    states = sz0_basis(cl.n_sites)
    state_index = {int(s): i for i, s in enumerate(states)}
    ice_indices = np.array([state_index[int(s)] for s in cl.ice_states], dtype=np.int64)
    n_band = cl.n_ice if args.n_band is None else args.n_band
    T = np.geomspace(1e-4, 0.12, 900)

    print(f"cluster: N={cl.n_sites}, Sz0 dim={len(states)}, ice={cl.n_ice}", flush=True)
    print(f"Jpm={args.jpm:+.5f}, n_band={n_band}", flush=True)

    if args.grid == "one":
        twists = [(0.0, 0.0, 0.0)]
    else:
        twists = list(product([0.0, np.pi], repeat=3))

    Hbands = []
    E_low_columns = []
    diagnostics = []
    for k, phi in enumerate(twists, start=1):
        print(f"[{k}/{len(twists)}] phi/pi = {tuple(float(x / np.pi) for x in phi)}", flush=True)
        Hb, E_low, diag = band_hamiltonian(
            cl, states, ice_indices, args.jpm, np.array(phi), n_band,
            args.tol, args.sigma, args.solver, args.maxiter
        )
        Hbands.append(Hb)
        E_low_columns.append(E_low)
        diagnostics.append(diag)
        print(
            f"    eigsh {diag['elapsed_s']:.1f}s, E0={diag['E_min']:.8f}, "
            f"band top={diag['E_max_band']:.8f}, ice overlap min={diag['ice_overlap_min']:.6f}",
            flush=True,
        )

    H_phi0 = Hbands[0]
    H_avg = sum(Hbands) / len(Hbands)
    E_phi0, C_phi0 = specific_heat_from_matrix(H_phi0, T)
    E_avg, C_avg = specific_heat_from_matrix(H_avg, T)

    # Perturbative reference, built from the same standalone code.  If a
    # truncated band is requested, compare to the lowest n_band levels of the
    # perturbative matrices.
    pt = R.sw_order23(cl, verbose=False)
    H_pt_all = R.assemble(cl, pt, args.jpm, "all")
    H_pt_clean = R.assemble(cl, pt, args.jpm, "delta0")
    E_pt_all_full = np.linalg.eigvalsh(H_pt_all)
    E_pt_clean_full = np.linalg.eigvalsh(H_pt_clean)
    E_pt_all = E_pt_all_full[:n_band]
    E_pt_clean = E_pt_clean_full[:n_band]
    C_pt_all = R.specific_heat(E_pt_all, T)
    C_pt_clean = R.specific_heat(E_pt_clean, T)

    summary = {
        "jpm": args.jpm,
        "n_sites": cl.n_sites,
        "sz0_dim": int(len(states)),
        "ice_dim": int(cl.n_ice),
        "grid": args.grid,
        "n_twists": len(twists),
        "diagnostics": diagnostics,
        "Tpk_full_band_phi0": R.refined_peak(T, C_phi0),
        "Tpk_full_band_twist_avg_operator": R.refined_peak(T, C_avg),
        "Tpk_pt_all": R.refined_peak(T, C_pt_all),
        "Tpk_pt_delta0": R.refined_peak(T, C_pt_clean),
        "g4": 4 * args.jpm * args.jpm,
        "ghex": 12 * abs(args.jpm) ** 3,
    }
    print("\nsummary")
    print(json.dumps(summary, indent=2))

    np.savez_compressed(
        args.out,
        T=T,
        H_phi0=H_phi0,
        H_avg=H_avg,
        E_phi0=E_phi0,
        C_phi0=C_phi0,
        E_avg=E_avg,
        C_avg=C_avg,
        E_pt_all=E_pt_all,
        C_pt_all=C_pt_all,
        E_pt_clean=E_pt_clean,
        C_pt_clean=C_pt_clean,
        E_low_by_twist=np.array(E_low_columns),
        summary=json.dumps(summary),
    )
    (args.out.with_suffix(".json")).write_text(json.dumps(summary, indent=2))
    print(f"wrote {args.out}")
    print(f"wrote {args.out.with_suffix('.json')}")


if __name__ == "__main__":
    main()
