#!/usr/bin/env python3
"""Production utilities for winding-free QSI ED: parity mask + graded BW.

Implements the two production objects of SIMULATION_PLAN.md on any cluster
that exposes ``positions``, ``bonds``, ``ice_states``, ``n_sites`` (the
interface of notes/recompute_finite_size_artifact.Cluster):

Stage A  mask_operator(H_B, mask)      -- Theorem-B parity mask of a band
                                          operator pulled back to the ice basis
                                          (equals the eight-corner character
                                          average exactly at M=2).
Stage B  BWDownfold / solve_masked_bw  -- graded Brillouin-Wigner downfolding,
                                          defined at any Jpm; the dense QHQ
                                          backend here serves clusters up to
                                          cubic-16; FCC-32 replaces
                                          ``BWDownfold`` with shifted-Krylov
                                          solves on the QED stack behind the
                                          same ``f(z)`` interface.

The mask needs only the ice-state polarizations x_alpha = sum_i r_i S_i^z in
the SAME coordinate frame as the cluster builder (FCC primitive frame for
FCC clusters) and the SAME ice-basis ordering as the pulled-back operator.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

__all__ = [
    "ice_polarizations", "parity_mask", "mask_operator",
    "fixed_sz_basis", "build_xxz_hamiltonian",
    "BWDownfold", "solve_masked_bw",
    "specific_heat", "entropy",
]


# ------------------------------------------------------------- mask (Stage A)

def ice_polarizations(cl, ice_order: np.ndarray | None = None) -> np.ndarray:
    """x_alpha = sum_i r_i S_i^z(alpha), rows ordered as cl.ice_states or as
    ice_order (indices into cl.ice_states)."""
    ice = np.asarray(cl.ice_states, dtype=np.uint64)
    if ice_order is not None:
        ice = ice[ice_order]
    n = cl.n_sites
    sz = ((ice[:, None] >> np.arange(n, dtype=np.uint64)) & np.uint64(1)).astype(float) - 0.5
    return sz @ cl.positions


def parity_mask(x: np.ndarray, tol: float = 1e-9) -> np.ndarray:
    """keep[a,b] iff 2(x_a - x_b) is componentwise even (Theorem B)."""
    diff = 2.0 * (x[:, None, :] - x[None, :, :])
    frac = np.mod(diff, 2.0)
    return np.all((np.abs(frac) < tol) | (np.abs(frac - 2.0) < tol), axis=2)


def mask_operator(h_band: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """The M=2 transported-dipole character average of a band operator."""
    return np.where(mask, h_band, 0.0)


# --------------------------------------------------- microscopic H (generic)

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


def build_xxz_hamiltonian(cl, jpm: float, basis: np.ndarray,
                          index: dict[int, int], jzz: float = 1.0):
    """Sparse H = Jzz sum SzSz - Jpm sum (S+S- + S-S+) in a fixed-Sz basis."""
    dim = len(basis)
    diag = np.zeros(dim)
    rows, cols, vals = [], [], []
    for (i, j) in cl.bonds:
        i, j = int(i), int(j)
        bi = (basis >> i) & 1
        bj = (basis >> j) & 1
        diag += jzz * (bi - 0.5) * (bj - 0.5)
        anti = np.nonzero(bi != bj)[0]
        flipped = basis[anti] ^ ((1 << i) | (1 << j))
        tgt = np.fromiter((index[int(s)] for s in flipped), dtype=np.int64,
                          count=len(anti))
        rows.append(anti)
        cols.append(tgt)
        vals.append(np.full(len(anti), -jpm))
    h = sp.coo_matrix((np.concatenate(vals),
                       (np.concatenate(rows), np.concatenate(cols))),
                      shape=(dim, dim)).tocsr()
    return h + sp.diags(diag)


# ------------------------------------------------------------- BW (Stage B)

class BWDownfold:
    """F(z) = PHP + W diag(1/(z-lam)) W^T via one dense eigh of QHQ.

    Dense backend: fine through cubic-16 (dim_Q ~ 1.3e4).  For FCC-32 keep
    the same interface but implement ``f(z)`` with block shifted-Krylov
    solves of (z - QHQ)^{-1} (QHP) on the QED stack.
    """

    def __init__(self, h: sp.csr_matrix, ice_idx: np.ndarray):
        dim = h.shape[0]
        q_idx = np.setdiff1d(np.arange(dim), ice_idx)
        hd = h.toarray()
        self.php = hd[np.ix_(ice_idx, ice_idx)].copy()
        phq = hd[np.ix_(ice_idx, q_idx)].copy()
        qhq = hd[np.ix_(q_idx, q_idx)].copy()
        del hd
        self.lam, v = np.linalg.eigh(qhq)
        self.w = phq @ v

    def f(self, z: float) -> np.ndarray:
        return self.php + (self.w / (z - self.lam)) @ self.w.T

    def nonice_weight(self, z: float, u: np.ndarray) -> float:
        y = self.w.T @ u / (z - self.lam)
        s = float(y @ y)
        return s / (1.0 + s)

    def pole_gap(self, z: float) -> float:
        return float(np.min(np.abs(z - self.lam)))


def solve_masked_bw(bw: BWDownfold, mask: np.ndarray | None, n_levels: int,
                    z0: float, max_it: int = 400, tol: float = 1e-12,
                    damp: float = 0.5):
    """Self-consistent E_n = eig_n F_clean(E_n) for n = 0..n_levels-1.

    Returns dict of arrays: energies, converged, pole_gap, nonice.
    """
    es, conv, pole, nonice = [], [], [], []
    for n in range(n_levels):
        z = es[-1] if es else z0
        ok = False
        for _ in range(max_it):
            fz = bw.f(z)
            if mask is not None:
                fz = np.where(mask, fz, 0.0)
            evals, evecs = np.linalg.eigh(fz)
            e_new = evals[n]
            if abs(e_new - z) < tol:
                ok = True
                break
            z = (1.0 - damp) * z + damp * e_new
        es.append(e_new if ok else z)
        conv.append(ok)
        pole.append(bw.pole_gap(es[-1]))
        nonice.append(bw.nonice_weight(es[-1], evecs[:, n]))
    return {"energies": np.asarray(es), "converged": np.asarray(conv),
            "pole_gap": np.asarray(pole), "nonice": np.asarray(nonice)}


# --------------------------------------------------------------- thermo

def specific_heat(evals: np.ndarray, T: np.ndarray) -> np.ndarray:
    e = np.asarray(evals, float)
    e = e - e.min()
    w = np.exp(-np.outer(1.0 / T, e))
    z = w.sum(axis=1)
    e1 = (w * e).sum(axis=1) / z
    e2 = (w * e * e).sum(axis=1) / z
    return (e2 - e1 * e1) / T**2


def entropy(evals: np.ndarray, T: np.ndarray) -> np.ndarray:
    e = np.asarray(evals, float)
    e = e - e.min()
    w = np.exp(-np.outer(1.0 / T, e))
    z = w.sum(axis=1)
    e1 = (w * e).sum(axis=1) / z
    return np.log(z) + e1 / T
