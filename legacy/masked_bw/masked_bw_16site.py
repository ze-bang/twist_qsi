#!/usr/bin/env python3
"""Masked Brillouin-Wigner downfolding on cubic-16: the all-Jpm validation ladder.

Production object (Stage A, M=2):

    F(z)       = P H P + P H Q (z - Q H Q)^{-1} Q H P      (exact, any Jpm)
    F_clean(z) = parity-mask of F(z)                        (Theorem B mask)
    levels:      E_n = eig_n F_clean(E_n)                   (self-consistent)

No eigenvectors of H, no band selection, no Gram matrix.  On 16 sites we can
afford exact ground truth, so per coupling this script also computes:

  - exact full spectrum of H (dense eigh, 12870)            [ground truth]
  - the Loewdin band operator of the lowest-90 band + mask  [current protocol]
  - eta_min = lambda_min of the ice Gram of that band       [old validity gate]

Validation targets (from twist_resolved_qed_dipole2_M2_*.npz):
  Jpm=-0.03: bare low band exact to ~1e-13, clean peak 0.0002996
  Jpm=-0.05: clean peak 0.0012227

Outputs: masked_bw_16site_results.json + masked_bw_16site_curves.npz
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import recompute_finite_size_artifact as R  # noqa: E402

JZZ = 1.0
JPM_LIST = [-0.03, -0.05, -0.08, -0.10, -0.18, -0.30, 0.05]
T_GRID = np.logspace(-5, 0.5, 2000)
MAX_IT = 400
TOL = 1e-12
DAMP = 0.5


# ---------------------------------------------------------------- basis / H

def fixed_sz_basis(n_sites: int, n_up: int) -> np.ndarray:
    basis = []
    state = (1 << n_up) - 1
    limit = 1 << n_sites
    while state < limit:
        basis.append(state)
        c = state & -state
        r = state + c
        state = (((r ^ state) >> 2) // c) | r
    return np.asarray(basis, dtype=np.int64)


def build_h(cl, jpm: float, basis: np.ndarray, index: dict[int, int]):
    """Sparse H = Jzz sum SzSz - Jpm sum (S+S- + S-S+) in the fixed-Sz basis."""
    dim = len(basis)
    diag = np.zeros(dim)
    rows, cols, vals = [], [], []
    for (i, j) in cl.bonds:
        i, j = int(i), int(j)
        bi = (basis >> i) & 1
        bj = (basis >> j) & 1
        diag += JZZ * (bi - 0.5) * (bj - 0.5)
        anti = np.nonzero(bi != bj)[0]
        flipped = basis[anti] ^ ((1 << i) | (1 << j))
        tgt = np.fromiter((index[int(s)] for s in flipped), dtype=np.int64,
                          count=len(anti))
        rows.append(anti)
        cols.append(tgt)
        vals.append(np.full(len(anti), -jpm))
    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    vals = np.concatenate(vals)
    h = sp.coo_matrix((vals, (rows, cols)), shape=(dim, dim)).tocsr()
    h = h + sp.diags(diag)
    return h


# ------------------------------------------------------------------- pieces

def ice_polarizations(cl) -> np.ndarray:
    ice = np.asarray(cl.ice_states, dtype=np.uint64)
    n = cl.n_sites
    sz = ((ice[:, None] >> np.arange(n, dtype=np.uint64)) & np.uint64(1)).astype(float) - 0.5
    return sz @ cl.positions


def parity_mask(x: np.ndarray, tol: float = 1e-9) -> np.ndarray:
    diff = 2.0 * (x[:, None, :] - x[None, :, :])
    frac = np.mod(diff, 2.0)
    return np.all((np.abs(frac) < tol) | (np.abs(frac - 2.0) < tol), axis=2)


def specific_heat(evals: np.ndarray, T: np.ndarray) -> np.ndarray:
    e = evals - evals.min()
    w = np.exp(-np.outer(1.0 / T, e))
    z = w.sum(axis=1)
    e1 = (w * e).sum(axis=1) / z
    e2 = (w * e * e).sum(axis=1) / z
    return (e2 - e1 * e1) / T**2


def peak_of(evals: np.ndarray, T: np.ndarray) -> float:
    return float(T[np.argmax(specific_heat(evals, T))])


def loewdin_band(psi_low: np.ndarray, evals_low: np.ndarray, ice_idx: np.ndarray):
    """Current protocol: Loewdin pullback of the lowest band onto the ice basis."""
    X = psi_low[ice_idx, :]
    S = X.T @ X
    svals = np.linalg.eigvalsh(S)
    eta = float(svals.min())
    if eta < 1e-8:
        return None, eta
    sv, sU = np.linalg.eigh(S)
    S_inv_sqrt = sU @ np.diag(1.0 / np.sqrt(sv)) @ sU.T
    Q = X @ S_inv_sqrt
    return Q @ np.diag(evals_low) @ Q.T, eta


# -------------------------------------------------------------- BW machinery

class BWDownfold:
    """F(z) = PHP + W diag(1/(z - lam)) W^T with QHQ = V lam V^T (dense, real)."""

    def __init__(self, h: sp.csr_matrix, ice_idx: np.ndarray):
        dim = h.shape[0]
        q_idx = np.setdiff1d(np.arange(dim), ice_idx)
        hd = h.toarray()
        self.php = hd[np.ix_(ice_idx, ice_idx)].copy()
        phq = hd[np.ix_(ice_idx, q_idx)].copy()
        qhq = hd[np.ix_(q_idx, q_idx)].copy()
        del hd
        t0 = time.time()
        self.lam, v = np.linalg.eigh(qhq)
        self.t_eigh = time.time() - t0
        self.w = phq @ v          # (n_ice, dim_q)

    def f(self, z: float) -> np.ndarray:
        denom = z - self.lam
        return self.php + (self.w / denom) @ self.w.T

    def solve_level(self, n: int, z0: float, mask: np.ndarray | None):
        """Self-consistent E = eig_n F_clean(E); returns (E, converged, iters,
        pole_gap, nonice_weight)."""
        z = z0
        for it in range(MAX_IT):
            fz = self.f(z)
            if mask is not None:
                fz = np.where(mask, fz, 0.0)
            evals, evecs = np.linalg.eigh(fz)
            e_new = evals[n]
            if abs(e_new - z) < TOL:
                u = evecs[:, n]
                y = self.w.T @ u / (z - self.lam)
                nonice = float(y @ y)
                return e_new, True, it + 1, float(np.min(np.abs(z - self.lam))), \
                    nonice / (1.0 + nonice)
            z = (1.0 - DAMP) * z + DAMP * e_new
        u = evecs[:, n]
        y = self.w.T @ u / (z - self.lam)
        nonice = float(y @ y)
        return z, False, MAX_IT, float(np.min(np.abs(z - self.lam))), \
            nonice / (1.0 + nonice)


# --------------------------------------------------------------------- main

def run_one(cl, jpm, basis, index, ice_idx, mask, x_ref: dict):
    n_ice = len(ice_idx)
    rec: dict = {"jpm": jpm}
    print(f"=== Jpm = {jpm:+.2f} ===", flush=True)

    h = build_h(cl, jpm, basis, index)

    # exact ground truth (16-site luxury)
    t0 = time.time()
    e_full, psi_full = np.linalg.eigh(h.toarray())
    rec["t_full_eigh"] = time.time() - t0
    e_low, psi_low = e_full[:n_ice], psi_full[:, :n_ice]

    # current protocol: Loewdin band + mask
    hb, eta = loewdin_band(psi_low, e_low, ice_idx)
    rec["eta_min_theta0"] = eta
    rec["bare_peak"] = peak_of(e_low, T_GRID)
    if hb is not None:
        e_loewdin_clean = np.linalg.eigvalsh(np.where(mask, hb, 0.0))
        rec["loewdin_clean_peak"] = peak_of(e_loewdin_clean, T_GRID)
    else:
        e_loewdin_clean = None
        rec["loewdin_clean_peak"] = None

    # BW downfold
    bw = BWDownfold(h, ice_idx)
    rec["t_qhq_eigh"] = bw.t_eigh
    rec["lam_min_QHQ"] = float(bw.lam.min())
    z0 = float(np.mean(np.diag(bw.php)))

    def solve_all(m):
        es, conv, pole, nonice = [], [], [], []
        for n in range(n_ice):
            zz = z0 if not es else es[-1]
            e, ok, _, pg, nw = bw.solve_level(n, zz, m)
            es.append(e)
            conv.append(ok)
            pole.append(pg)
            nonice.append(nw)
        return (np.asarray(es), np.asarray(conv), np.asarray(pole),
                np.asarray(nonice))

    e_bw, conv_b, pole_b, ni_b = solve_all(None)
    e_bwc, conv_c, pole_c, ni_c = solve_all(mask)

    rec["bw_bare_vs_exact_max_err"] = float(np.max(np.abs(np.sort(e_bw) - e_low)))
    rec["bw_bare_converged"] = int(conv_b.sum())
    rec["bw_clean_converged"] = int(conv_c.sum())
    rec["bw_clean_peak"] = peak_of(e_bwc, T_GRID)
    rec["bw_clean_min_pole_gap"] = float(pole_c.min())
    rec["bw_clean_max_nonice_weight"] = float(ni_c.max())
    rec["bw_clean_mean_nonice_weight"] = float(ni_c.mean())
    rec["ghex"] = 12.0 * abs(jpm) ** 3 / JZZ**2
    rec["g4"] = 4.0 * jpm**2 / JZZ
    rec["bare_peak_over_g4"] = rec["bare_peak"] / rec["g4"]
    rec["bw_clean_peak_over_ghex"] = rec["bw_clean_peak"] / rec["ghex"]
    if e_loewdin_clean is not None:
        rec["bw_vs_loewdin_clean_spec_dev"] = float(
            np.max(np.abs(np.sort(e_bwc) - np.sort(e_loewdin_clean))))

    # regression against saved eight-corner QED files where available
    tag = {-0.03: "jm0p03", -0.05: "jm0p05"}.get(jpm)
    if tag:
        d = np.load(HERE / f"twist_resolved_qed_dipole2_M2_{tag}.npz",
                    allow_pickle=True)
        rec["exact_low_vs_saved_max_err"] = float(
            np.max(np.abs(e_low - d["E_qed_phi0"])))
        e_saved_clean = np.linalg.eigvalsh(d["H_qed_twist_avg"])
        rec["saved_clean_peak"] = peak_of(e_saved_clean, T_GRID)
        rec["bw_clean_vs_saved_clean_spec_dev"] = float(
            np.max(np.abs(np.sort(e_bwc) - np.sort(e_saved_clean))))

    x_ref[f"{jpm:+.2f}"] = {
        "e_bare": e_low, "e_bw_clean": e_bwc,
        "e_loewdin_clean": e_loewdin_clean,
        "C_bare": specific_heat(e_low, T_GRID),
        "C_bw_clean": specific_heat(e_bwc, T_GRID),
        "pole_gap": pole_c, "nonice": ni_c,
    }
    for k, v in rec.items():
        print(f"  {k}: {v}", flush=True)
    return rec


def main():
    cl = R.build_cluster("cubic", (1, 1, 1))
    basis = fixed_sz_basis(cl.n_sites, cl.n_sites // 2)
    index = {int(s): k for k, s in enumerate(basis)}
    ice_sorted = np.sort(np.asarray(cl.ice_states, dtype=np.int64))
    # mask must follow the ice ordering used for P rows
    order = np.argsort(np.asarray(cl.ice_states, dtype=np.int64))
    x = ice_polarizations(cl)[order]
    mask = parity_mask(x)
    ice_idx = np.fromiter((index[int(s)] for s in ice_sorted), dtype=np.int64)

    curves: dict = {}
    records = [run_one(cl, jpm, basis, index, ice_idx, mask, curves)
               for jpm in JPM_LIST]

    with open(HERE / "masked_bw_16site_results.json", "w") as fh:
        json.dump(records, fh, indent=1)
    flat = {"T": T_GRID}
    for tag, d in curves.items():
        for k, v in d.items():
            if v is not None:
                flat[f"{tag}/{k}"] = v
    np.savez_compressed(HERE / "masked_bw_16site_curves.npz", **flat)
    print("done")


if __name__ == "__main__":
    main()
