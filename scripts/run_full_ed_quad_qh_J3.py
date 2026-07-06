#!/usr/bin/env python3
"""Exact full diagonalization → thermal expectation ⟨Q⟩(T), ⟨Q H⟩(T)
and the connected correlator ⟨Q H⟩_c = ⟨Q H⟩ − ⟨Q⟩⟨H⟩ for the five
quadrupole operators (Eg_Q1, Eg_Q2, T2g_Q_xy, T2g_Q_xz, T2g_Q_yz).

The Hamiltonian and quadrupole operators are all invariant under the
fcc translation subgroup :math:`\\mathbb{Z}_2 \\times \\mathbb{Z}_2`
of the 1×1×1 conventional cubic cluster, so we project both into the
four momentum sectors of the cluster and diagonalise each block.  For
each sector we keep the eigenvectors so we can evaluate the diagonal
matrix elements

    Q_n^{(\\alpha,a)} = ⟨n_\\alpha | Q^a_\\alpha | n_\\alpha⟩

and aggregate the global thermal averages.

Memory at 16 sites: each complex block carries the eigenvector matrix
(~4.3 GB), plus one Q · V workspace, plus the residual sparse Q
matrices.  Peak is approximately 12 GB per sector during the
Q-diagonal evaluation, with at most one sector resident at a time.
"""
from __future__ import annotations

import argparse
import functools
import multiprocessing as mp
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp

print = functools.partial(print, flush=True)


def _ts() -> str:
    return time.strftime("%H:%M:%S")

N_SITES = 16

QUAD_OPS = [
    ("Eg_Q1", "Eg_Q1"),
    ("Eg_Q2", "Eg_Q2"),
    ("T2g_Q_xy", "T2g_Q_xy"),
    ("T2g_Q_xz", "T2g_Q_xz"),
    ("T2g_Q_yz", "T2g_Q_yz"),
]
QUAD_OP_NAMES = {name for name, _ in QUAD_OPS}


# ---------------------------------------------------------------------------
# Deck parsers
# ---------------------------------------------------------------------------

def _parse_quad_trans(path: Path):
    """Parse a quadrupole ``.Trans.dat`` (legacy block format).

    Rows look like ``op site re im``.  The file has a small header
    ``num <N>`` and three separator lines.
    """
    rows = []
    if not path.exists():
        return rows
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) == 4:
                try:
                    op, site = int(parts[0]), int(parts[1])
                    cr, ci = float(parts[2]), float(parts[3])
                except ValueError:
                    continue
                c = complex(cr, ci)
                if abs(c) > 1e-15:
                    rows.append((op, site, c))
    return rows


def _parse_trans_simple(path: Path):
    """Legacy 4-column ``Trans.dat`` files (used by the H deck)."""
    rows = []
    if not path.exists():
        return rows
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) == 4:
                try:
                    op, site = int(parts[0]), int(parts[1])
                    cr, ci = float(parts[2]), float(parts[3])
                except ValueError:
                    continue
                c = complex(cr, ci)
                if abs(c) > 1e-15:
                    rows.append((op, site, c))
    return rows


def _parse_interall(path: Path):
    rows = []
    if not path.exists():
        return rows
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) == 6:
                try:
                    op1, s1, op2, s2 = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
                    cr, ci = float(parts[4]), float(parts[5])
                except ValueError:
                    continue
                c = complex(cr, ci)
                if abs(c) > 1e-15:
                    rows.append((op1, s1, op2, s2, c))
    return rows


def _parse_three(path: Path):
    rows = []
    if not path.exists():
        return rows
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) == 8:
                try:
                    op1, s1, op2, s2, op3, s3 = (int(parts[0]), int(parts[1]),
                                                  int(parts[2]), int(parts[3]),
                                                  int(parts[4]), int(parts[5]))
                    cr, ci = float(parts[6]), float(parts[7])
                except ValueError:
                    continue
                c = complex(cr, ci)
                if abs(c) > 1e-15:
                    rows.append((op1, s1, op2, s2, op3, s3, c))
    return rows


# ---------------------------------------------------------------------------
# Vectorised single-site action and operator construction
# ---------------------------------------------------------------------------

def _apply_op_vec(states: np.ndarray, op: int, site: int):
    bits = ((states >> np.uint64(site)) & np.uint64(1)).astype(np.int8)
    if op == 2:
        amps = np.where(bits == 0, 0.5, -0.5).astype(np.float64)
        return states, amps, np.ones(states.shape, dtype=bool)
    if op == 0:  # S+: down(1) → up(0)
        valid = bits == 1
        new = states ^ (np.uint64(1) << np.uint64(site))
        new = np.where(valid, new, states)
        return new, valid.astype(np.float64), valid
    if op == 1:  # S-: up(0) → down(1)
        valid = bits == 0
        new = states ^ (np.uint64(1) << np.uint64(site))
        new = np.where(valid, new, states)
        return new, valid.astype(np.float64), valid
    raise ValueError(f"unknown op {op}")


def _add_term(rows_list, cols_list, vals_list, out_states, states, amps, valid, coeff):
    v = coeff * amps
    mask = valid & (np.abs(v) > 1e-15)
    if not mask.any():
        return
    rows_list.append(out_states[mask].astype(np.int64, copy=False))
    cols_list.append(states[mask].astype(np.int64, copy=False))
    vals_list.append(v[mask])


def build_H_components(
    ham_dir: Path, n_sites: int = N_SITES
) -> Dict[str, sp.csr_matrix]:
    """Build H decomposed into Ising, transverse (pm), and J3 (three-body) parts.

    Classification of 2-body InterAll terms by operator codes:
      Ising  : op1=2 (Sz) and op2=2 (Sz)   -> SzSz diagonal
      pm     : (op1,op2) in {(0,1),(1,0)}   -> S+S-/S-S+ transverse hopping
    Three-body terms (ThreeBodyG.dat) go into the J3 component.
    """
    dim = 1 << n_sites
    states = np.arange(dim, dtype=np.uint64)
    inter = _parse_interall(ham_dir / "InterAll.dat")
    three = _parse_three(ham_dir / "ThreeBodyG.dat")
    components: Dict[str, Tuple[list, list, list]] = {
        "H_Ising": ([], [], []),
        "H_pm":    ([], [], []),
    }
    for op1, s1, op2, s2, c in inter:
        if op1 == 2 and op2 == 2:
            key = "H_Ising"
        else:
            key = "H_pm"
        rs_list, cs_list, vs_list = components[key]
        sa, a1, v1 = _apply_op_vec(states, op1, s1)
        sb, a2, v2 = _apply_op_vec(sa, op2, s2)
        _add_term(rs_list, cs_list, vs_list, sb, states, a1 * a2, v1 & v2, c)

    result: Dict[str, sp.csr_matrix] = {}
    for key, (rs, cs, vs) in components.items():
        if not vs:
            result[key] = sp.csr_matrix((dim, dim), dtype=np.float64)
        else:
            M = sp.coo_matrix(
                (np.concatenate(vs), (np.concatenate(rs), np.concatenate(cs))),
                shape=(dim, dim),
            ).tocsr()
            M = 0.5 * (M + M.conj().T)
            result[key] = M

    if three:
        rs3, cs3, vs3 = [], [], []
        for op1, s1, op2, s2, op3, s3, c in three:
            sa, a1, v1 = _apply_op_vec(states, op1, s1)
            sb, a2, v2 = _apply_op_vec(sa, op2, s2)
            sc, a3, v3 = _apply_op_vec(sb, op3, s3)
            _add_term(rs3, cs3, vs3, sc, states, a1 * a2 * a3, v1 & v2 & v3, c)
        M3 = sp.coo_matrix(
            (np.concatenate(vs3), (np.concatenate(rs3), np.concatenate(cs3))),
            shape=(dim, dim),
        ).tocsr()
        result["H_J3"] = 0.5 * (M3 + M3.conj().T)

    return result


def build_H_sparse(ham_dir: Path, n_sites: int = N_SITES) -> sp.csr_matrix:
    dim = 1 << n_sites
    states = np.arange(dim, dtype=np.uint64)
    trans = _parse_trans_simple(ham_dir / "Trans.dat")
    inter = _parse_interall(ham_dir / "InterAll.dat")
    three = _parse_three(ham_dir / "ThreeBodyG.dat")
    print(f"  H deck: {len(trans)} 1-body, {len(inter)} 2-body, {len(three)} 3-body")

    rows_list, cols_list, vals_list = [], [], []
    for op, site, c in trans:
        new, a, v = _apply_op_vec(states, op, site)
        _add_term(rows_list, cols_list, vals_list, new, states, a, v, c)
    for op1, s1, op2, s2, c in inter:
        sa, a1, v1 = _apply_op_vec(states, op1, s1)
        sb, a2, v2 = _apply_op_vec(sa, op2, s2)
        _add_term(rows_list, cols_list, vals_list, sb, states, a1 * a2, v1 & v2, c)
    for op1, s1, op2, s2, op3, s3, c in three:
        sa, a1, v1 = _apply_op_vec(states, op1, s1)
        sb, a2, v2 = _apply_op_vec(sa, op2, s2)
        sc, a3, v3 = _apply_op_vec(sb, op3, s3)
        _add_term(rows_list, cols_list, vals_list, sc, states,
                  a1 * a2 * a3, v1 & v2 & v3, c)

    rows = np.concatenate(rows_list)
    cols = np.concatenate(cols_list)
    vals = np.concatenate(vals_list)
    H = sp.coo_matrix((vals, (rows, cols)), shape=(dim, dim)).tocsr()
    H = 0.5 * (H + H.conj().T)  # scrub roundoff
    return H


def build_quad_sparse(quad_dir: Path, name: str, n_sites: int = N_SITES) -> sp.csr_matrix:
    """Build the quadrupole operator as a sparse matrix.

    The quadrupole decks use the same legacy 4-column format for
    one-body terms (``op site re im``) plus an empty/optional
    ``InterAll`` companion.  At the moment, the five quadrupole files
    in this repository only carry one-body terms.
    """
    dim = 1 << n_sites
    states = np.arange(dim, dtype=np.uint64)
    trans = _parse_quad_trans(quad_dir / f"{name}.Trans.dat")
    inter = _parse_interall(quad_dir / f"{name}.InterAll.dat")
    rows_list, cols_list, vals_list = [], [], []
    for op, site, c in trans:
        new, a, v = _apply_op_vec(states, op, site)
        _add_term(rows_list, cols_list, vals_list, new, states, a, v, c)
    for op1, s1, op2, s2, c in inter:
        sa, a1, v1 = _apply_op_vec(states, op1, s1)
        sb, a2, v2 = _apply_op_vec(sa, op2, s2)
        _add_term(rows_list, cols_list, vals_list, sb, states, a1 * a2, v1 & v2, c)
    if not vals_list:
        raise RuntimeError(f"No matrix elements assembled for {name} from {quad_dir}")
    rows = np.concatenate(rows_list)
    cols = np.concatenate(cols_list)
    vals = np.concatenate(vals_list)
    Q = sp.coo_matrix((vals, (rows, cols)), shape=(dim, dim)).tocsr()
    Q = 0.5 * (Q + Q.conj().T)
    return Q


def parse_source_specs(specs: List[str]) -> Dict[str, float]:
    """Parse repeated NAME=VALUE source terms.

    The source enters as H -> H - sum_a lambda_a Q_a.
    """
    out: Dict[str, float] = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"invalid --source '{spec}' (expected NAME=VALUE)")
        name, value = spec.split("=", 1)
        name = name.strip()
        if name not in QUAD_OP_NAMES:
            allowed = ", ".join(sorted(QUAD_OP_NAMES))
            raise ValueError(f"unknown source operator '{name}' (allowed: {allowed})")
        try:
            coeff = float(value)
        except ValueError as exc:
            raise ValueError(f"invalid source strength in '{spec}'") from exc
        out[name] = out.get(name, 0.0) + coeff
    return out


# ---------------------------------------------------------------------------
# Translation Z2 x Z2 sector basis (same as in run_full_ed_thermo_J3.py)
# ---------------------------------------------------------------------------

def _fcc_perms() -> Tuple[np.ndarray, np.ndarray]:
    T1 = np.array([12,13,14,15, 8, 9,10,11, 4, 5, 6, 7, 0, 1, 2, 3], dtype=np.int64)
    T2 = np.array([ 8, 9,10,11,12,13,14,15, 0, 1, 2, 3, 4, 5, 6, 7], dtype=np.int64)
    return T1, T2


def _bit_perm_vec(states: np.ndarray, perm: np.ndarray, n_sites: int) -> np.ndarray:
    out = np.zeros_like(states)
    one = np.int64(1)
    for i in range(n_sites):
        bit = (states >> np.int64(i)) & one
        out |= bit << np.int64(int(perm[i]))
    return out


def build_sector_projectors(n_sites: int = N_SITES) -> Dict[Tuple[int, int], sp.csr_matrix]:
    T1_perm, T2_perm = _fcc_perms()
    dim = 1 << n_sites
    states = np.arange(dim, dtype=np.int64)
    T1_states = _bit_perm_vec(states, T1_perm, n_sites)
    T2_states = _bit_perm_vec(states, T2_perm, n_sites)
    T12_states = _bit_perm_vec(T1_states, T2_perm, n_sites)
    orbit_members = np.stack([states, T1_states, T2_states, T12_states])
    rep_of = orbit_members.min(axis=0)
    unique_reps = np.unique(rep_of)
    rep_T1 = T1_states[unique_reps]
    rep_T2 = T2_states[unique_reps]
    rep_T12 = T12_states[unique_reps]
    stab_T1 = (rep_T1 == unique_reps)
    stab_T2 = (rep_T2 == unique_reps)
    stab_T12 = (rep_T12 == unique_reps)
    stab_size = 1 + stab_T1.astype(int) + stab_T2.astype(int) + stab_T12.astype(int)
    orbit_size = 4 // stab_size
    image_at_rep = [unique_reps, rep_T1, rep_T2, rep_T12]
    projectors: Dict[Tuple[int, int], sp.csr_matrix] = {}
    for s1 in (+1, -1):
        for s2 in (+1, -1):
            chi = np.array([1, s1, s2, s1 * s2], dtype=np.int64)
            lives = np.ones(len(unique_reps), dtype=bool)
            lives &= ~(stab_T1 & (chi[1] != 1))
            lives &= ~(stab_T2 & (chi[2] != 1))
            lives &= ~(stab_T12 & (chi[3] != 1))
            rows = []
            cols = []
            vals = []
            row_idx = 0
            for o_idx in np.flatnonzero(lives):
                norm = 1.0 / np.sqrt(orbit_size[o_idx])
                seen: dict = {}
                for g in range(4):
                    n = int(image_at_rep[g][o_idx])
                    if n not in seen:
                        seen[n] = g
                for n, g in seen.items():
                    rows.append(row_idx)
                    cols.append(n)
                    vals.append(float(chi[g]) * norm)
                row_idx += 1
            P = sp.csr_matrix(
                (np.asarray(vals, dtype=np.float64),
                 (np.asarray(rows, dtype=np.int64),
                  np.asarray(cols, dtype=np.int64))),
                shape=(row_idx, dim),
            )
            projectors[(s1, s2)] = P
    return projectors


# ---------------------------------------------------------------------------
# Thermodynamics + connected ⟨QH⟩
# ---------------------------------------------------------------------------

def evaluate_thermal_curves(
    evals: np.ndarray,
    q_diags: Dict[str, np.ndarray],
    t_grid: np.ndarray,
    h_diags: Dict[str, np.ndarray] | None = None,
) -> Dict[str, np.ndarray]:
    """Compute thermal observables on a temperature grid using the full
    spectrum and the diagonals of each Q operator.

    If h_diags is provided (dict name -> diagonal array), also compute
    per-H-component connected correlators:
        alpha_{Q,X}(T) = (<Q H_X>_conn) / T^2
    stored as  f"{q_name}_x_{h_name}_alpha".
    """
    h_diags = h_diags or {}
    out: Dict[str, np.ndarray] = {
        "temperatures": t_grid,
        "energy": np.zeros_like(t_grid),
        "energy_var": np.zeros_like(t_grid),
        "specific_heat": np.zeros_like(t_grid),
        "free_energy": np.zeros_like(t_grid),
        "entropy": np.zeros_like(t_grid),
    }
    for name in q_diags:
        out[f"{name}_expect"] = np.zeros_like(t_grid)
        out[f"{name}_QH_expect"] = np.zeros_like(t_grid)
        out[f"{name}_QH_connected"] = np.zeros_like(t_grid)
        out[f"{name}_alpha"] = np.zeros_like(t_grid)
        out[f"{name}_Q2_expect"] = np.zeros_like(t_grid)
        out[f"{name}_chi"] = np.zeros_like(t_grid)
    for hname in h_diags:
        out[f"{hname}_expect"] = np.zeros_like(t_grid)
        for qname in q_diags:
            out[f"{qname}_x_{hname}_alpha"] = np.zeros_like(t_grid)

    e_min = float(evals.min())
    shifted = evals - e_min
    for i, T in enumerate(t_grid):
        beta = 1.0 / T
        w = np.exp(-beta * shifted)
        Z = w.sum()
        p = w / Z
        E = float((evals * p).sum())
        E2 = float((evals * evals * p).sum())
        out["energy"][i] = E
        out["energy_var"][i] = E2 - E * E
        out["specific_heat"][i] = (E2 - E * E) / (T * T)
        out["free_energy"][i] = e_min - T * np.log(Z)
        out["entropy"][i] = (E - out["free_energy"][i]) / T

        # H-component expectations (for decomposition)
        HX_exp: Dict[str, float] = {}
        for hname, hd in h_diags.items():
            HX_exp[hname] = float((hd * p).sum())
            out[f"{hname}_expect"][i] = HX_exp[hname]

        for name, qd in q_diags.items():
            Qexp = float((qd * p).sum())
            QHexp = float((qd * evals * p).sum())
            Q2exp = float((qd * qd * p).sum())
            out[f"{name}_expect"][i] = Qexp
            out[f"{name}_QH_expect"][i] = QHexp
            out[f"{name}_QH_connected"][i] = QHexp - Qexp * E
            out[f"{name}_alpha"][i] = (QHexp - Qexp * E) / (T * T)
            out[f"{name}_Q2_expect"][i] = Q2exp
            out[f"{name}_chi"][i] = (Q2exp - Qexp * Qexp) / T
            # per-component contributions
            for hname, hd in h_diags.items():
                QHXexp = float((qd * hd * p).sum())
                out[f"{name}_x_{hname}_alpha"][i] = (
                    QHXexp - Qexp * HX_exp[hname]
                ) / (T * T)
    return out


# ---------------------------------------------------------------------------
# Per-sector diagonalisation (checkpointable / parallelisable)
# ---------------------------------------------------------------------------

def _sector_ckpt_path(ckpt_dir: Path, label: Tuple[int, int]) -> Path:
    s1, s2 = label
    return ckpt_dir / f"sector_{s1:+d}_{s2:+d}.npz"


def _load_sector_ckpt(path: Path) -> Tuple[np.ndarray, Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    data = np.load(path)
    ev = data["eigenvalues"]
    q_diags = {name: data[f"{name}_diagonal"] for name, _ in QUAD_OPS}
    h_diags = {k[:-9]: data[k] for k in data.files
               if k.endswith("_diagonal") and k not in {f"{n}_diagonal" for n, _ in QUAD_OPS}}
    return ev, q_diags, h_diags


def process_one_sector(
    label: Tuple[int, int],
    P: sp.csr_matrix,
    H: sp.csr_matrix,
    Qs: Dict[str, sp.csr_matrix],
    H_components: Dict[str, sp.csr_matrix] | None = None,
    *,
    use_real_H: bool,
    driver: str,
    ckpt_dir: Path | None = None,
) -> Tuple[np.ndarray, Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Diagonalise one translation sector and return (evals, q_diags, h_diags).

    h_diags contains diagonal matrix elements of each H component in the
    eigenbasis.  Empty dict when H_components is None.
    """
    tag = f"({label[0]:+d},{label[1]:+d})"
    if ckpt_dir is not None:
        ckpt = _sector_ckpt_path(ckpt_dir, label)
        if ckpt.exists():
            ev, q_diags, h_diags_ckpt = _load_sector_ckpt(ckpt)
            # If decomp requested but not in checkpoint, fall through to recompute
            if H_components is None or all(k in h_diags_ckpt for k in H_components):
                print(f"[{_ts()}] sector {tag}: loading checkpoint {ckpt}")
                h_diags = {k: h_diags_ckpt[k] for k in H_components} if H_components else {}
                return ev, q_diags, h_diags

    print(f"[{_ts()}] sector {tag}: projecting H ...")
    t0 = time.time()
    Halpha = (P @ H @ P.T).toarray()
    Halpha = 0.5 * (Halpha + Halpha.conj().T)
    if use_real_H:
        Halpha = Halpha.real.astype(np.float64, copy=False)
    else:
        Halpha = Halpha.astype(np.complex128, copy=False)
    print(f"[{_ts()}] sector {tag}: H block ready in {time.time()-t0:.1f}s "
          f"(dim={Halpha.shape[0]}, mem={Halpha.nbytes/1e9:.2f} GB)")

    print(f"[{_ts()}] sector {tag}: scipy.linalg.eigh(driver='{driver}') ...")
    t0 = time.time()
    ev, V = la.eigh(Halpha, driver=driver, overwrite_a=True)
    print(f"[{_ts()}] sector {tag}: diagonalised in {time.time()-t0:.1f}s; "
          f"E_min={ev[0]:.8f} E_max={ev[-1]:.8f}")
    del Halpha

    q_diags: Dict[str, np.ndarray] = {}
    for name, _ in QUAD_OPS:
        t0 = time.time()
        Qalpha = P @ Qs[name] @ P.T
        QV = Qalpha @ V
        qd = np.einsum("kn,kn->n", V.conj(), QV).real.astype(np.float64)
        q_diags[name] = qd
        print(f"[{_ts()}] sector {tag}: {name} diag in {time.time()-t0:.1f}s; "
              f"min={qd.min():.4f} max={qd.max():.4f}")
        del QV, Qalpha

    h_diags: Dict[str, np.ndarray] = {}
    for hname, Hc in (H_components or {}).items():
        t0 = time.time()
        Hc_alpha = (P @ Hc @ P.T).toarray()
        if use_real_H:
            Hc_alpha = Hc_alpha.real.astype(np.float64, copy=False)
        HcV = Hc_alpha @ V
        hd = np.einsum("kn,kn->n", V.conj(), HcV).real.astype(np.float64)
        h_diags[hname] = hd
        print(f"[{_ts()}] sector {tag}: {hname} diag in {time.time()-t0:.1f}s; "
              f"min={hd.min():.4f} max={hd.max():.4f}")
        del HcV, Hc_alpha
    del V

    if ckpt_dir is not None:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt = _sector_ckpt_path(ckpt_dir, label)
        payload = {"eigenvalues": ev}
        for name, qd in q_diags.items():
            payload[f"{name}_diagonal"] = qd
        for hname, hd in h_diags.items():
            payload[f"{hname}_diagonal"] = hd
        np.savez_compressed(ckpt, **payload)
        print(f"[{_ts()}] sector {tag}: wrote checkpoint {ckpt}")
    return ev, q_diags, h_diags


# Fork-pool globals (Linux copy-on-write after parent builds H/Q/P).
_POOL_H: sp.csr_matrix | None = None
_POOL_QS: Dict[str, sp.csr_matrix] | None = None
_POOL_PS: Dict[Tuple[int, int], sp.csr_matrix] | None = None
_POOL_H_COMPONENTS: Dict[str, sp.csr_matrix] | None = None
_POOL_USE_REAL_H = False
_POOL_DRIVER = "evr"
_POOL_CKPT_DIR: Path | None = None


def _pool_sector_task(label: Tuple[int, int]) -> Tuple[Tuple[int, int], np.ndarray, Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    assert _POOL_H is not None and _POOL_QS is not None and _POOL_PS is not None
    ev, q_diags, h_diags = process_one_sector(
        label, _POOL_PS[label], _POOL_H, _POOL_QS, _POOL_H_COMPONENTS,
        use_real_H=_POOL_USE_REAL_H,
        driver=_POOL_DRIVER,
        ckpt_dir=_POOL_CKPT_DIR,
    )
    return label, ev, q_diags, h_diags


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ham", required=True, type=Path)
    ap.add_argument("--quad", required=True, type=Path,
                    help="directory containing Eg_Q1.Trans.dat etc.")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--n-sites", type=int, default=N_SITES)
    ap.add_argument("--temp-min", type=float, default=0.005)
    ap.add_argument("--temp-max", type=float, default=5.0)
    ap.add_argument("--temp-points", type=int, default=80)
    ap.add_argument("--driver", default="evr", choices=["evd", "evr", "ev"],
                    help="LAPACK driver; evr (MRRR) is ~2-3× faster than evd for eigenvectors")
    ap.add_argument("--checkpoint-dir", type=Path, default=None,
                    help="save/load per-sector NPZ checkpoints (resume-safe)")
    ap.add_argument("--parallel-sectors", type=int, default=1, choices=[1, 2, 3, 4],
                    help="diagonalise up to 4 sectors concurrently (Linux fork pool)")
    ap.add_argument("--source", action="append", default=[],
                    help="uniform source term NAME=VALUE, applied as H <- H - VALUE * Q_NAME")
    ap.add_argument("--decompose-H", action="store_true", default=False,
                    help="also compute per-H-term alpha_{Q,X} contributions "
                         "(H_Ising, H_pm, H_J3 if present); requires re-diagonalisation "
                         "if checkpoints lack the H-diagonal arrays")
    args = ap.parse_args()

    if args.out.exists():
        print(f"output {args.out} already exists; aborting")
        return

    n_threads = os.environ.get("OMP_NUM_THREADS", "?")
    print(f"=== Full ED connected ⟨QH⟩ for ham={args.ham}")
    print(f"  driver={args.driver}  parallel_sectors={args.parallel_sectors}  "
          f"OMP_NUM_THREADS={n_threads}")
    if args.parallel_sectors > 1:
        print(f"  tip: set OMP_NUM_THREADS≈{max(1, (os.cpu_count() or 24)//args.parallel_sectors)} "
              f"per worker to avoid oversubscription")
    t0 = time.time()
    H = build_H_sparse(args.ham, args.n_sites)
    print(f"  H sparse built in {time.time()-t0:.1f}s; nnz={H.nnz}")

    max_imag = abs(H.imag).max() if H.imag.nnz else 0.0
    print(f"  H max|Im| = {max_imag:.3e}")
    use_real_H = max_imag < 1e-10

    H_components: Dict[str, sp.csr_matrix] | None = None
    if args.decompose_H:
        print("Building H-component matrices (Ising / pm / J3) ...")
        H_components = build_H_components(args.ham, args.n_sites)
        for cname, Hc in H_components.items():
            print(f"  {cname}: nnz={Hc.nnz}")

    print(f"Loading quadrupole operators from {args.quad}")
    Qs: Dict[str, sp.csr_matrix] = {}
    for name, _ in QUAD_OPS:
        Q = build_quad_sparse(args.quad, name, args.n_sites)
        Qs[name] = Q
        print(f"  {name}: nnz={Q.nnz} max|Im|={abs(Q.imag).max() if Q.imag.nnz else 0.0:.3e}")

    source_terms = parse_source_specs(args.source)
    if source_terms:
        print("Adding explicit symmetry-breaking source terms:")
        for name in sorted(source_terms):
            coeff = source_terms[name]
            print(f"  H <- H - ({coeff:+.6e}) * {name}")
            H = H - coeff * Qs[name]
        H = 0.5 * (H + H.conj().T)

    print("Building Z2xZ2 translation projectors...")
    Ps = build_sector_projectors(args.n_sites)
    total_dim = sum(P.shape[0] for P in Ps.values())
    assert total_dim == (1 << args.n_sites), f"sector dims sum to {total_dim}"
    print("  sectors: " + ", ".join(f"({k[0]:+d},{k[1]:+d}) dim={v.shape[0]}"
                                     for k, v in Ps.items()))

    labels = sorted(Ps.keys(), reverse=True)
    sector_results: Dict[Tuple[int, int], Tuple[np.ndarray, Dict[str, np.ndarray], Dict[str, np.ndarray]]] = {}

    if args.parallel_sectors > 1:
        global _POOL_H, _POOL_QS, _POOL_PS, _POOL_H_COMPONENTS, _POOL_USE_REAL_H, _POOL_DRIVER, _POOL_CKPT_DIR
        _POOL_H, _POOL_QS, _POOL_PS = H, Qs, Ps
        _POOL_H_COMPONENTS = H_components
        _POOL_USE_REAL_H, _POOL_DRIVER, _POOL_CKPT_DIR = use_real_H, args.driver, args.checkpoint_dir
        print(f"[{_ts()}] launching {args.parallel_sectors} sector workers (fork pool) ...")
        ctx = mp.get_context("fork")
        with ctx.Pool(processes=args.parallel_sectors) as pool:
            for label, ev, q_diags, h_diags in pool.imap_unordered(_pool_sector_task, labels):
                sector_results[label] = (ev, q_diags, h_diags)
                print(f"[{_ts()}] sector ({label[0]:+d},{label[1]:+d}) worker done "
                      f"({len(sector_results)}/{len(labels)})")
    else:
        for label in labels:
            ev, q_diags, h_diags = process_one_sector(
                label, Ps[label], H, Qs, H_components,
                use_real_H=use_real_H,
                driver=args.driver,
                ckpt_dir=args.checkpoint_dir,
            )
            sector_results[label] = (ev, q_diags, h_diags)

    all_evals = [sector_results[label][0] for label in labels]
    q_diags_pieces: Dict[str, List[np.ndarray]] = {
        name: [sector_results[label][1][name] for label in labels]
        for name, _ in QUAD_OPS
    }
    h_diags_pieces: Dict[str, List[np.ndarray]] = {}
    if H_components:
        for hname in H_components:
            h_diags_pieces[hname] = [sector_results[label][2][hname] for label in labels]

    evals = np.concatenate(all_evals)
    q_diags: Dict[str, np.ndarray] = {
        name: np.concatenate(q_diags_pieces[name]) for name, _ in QUAD_OPS
    }
    h_diags: Dict[str, np.ndarray] = {
        hname: np.concatenate(h_diags_pieces[hname]) for hname in h_diags_pieces
    }
    order = np.argsort(evals)
    evals = evals[order]
    for name in q_diags:
        q_diags[name] = q_diags[name][order]
    for hname in h_diags:
        h_diags[hname] = h_diags[hname][order]
    print(f"Full spectrum: n={len(evals)} E_min={evals[0]:.8f} E_max={evals[-1]:.8f}")

    t_grid = np.logspace(np.log10(args.temp_min), np.log10(args.temp_max), args.temp_points)
    curves = evaluate_thermal_curves(evals, q_diags, t_grid, h_diags if h_diags else None)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(curves)
    payload["eigenvalues"] = evals
    for name, qd in q_diags.items():
        payload[f"{name}_diagonal"] = qd
    for hname, hd in h_diags.items():
        payload[f"{hname}_diagonal"] = hd
    if source_terms:
        for name, coeff in source_terms.items():
            payload[f"source_{name}"] = np.array(coeff, dtype=np.float64)
    np.savez_compressed(args.out, **payload)
    print(f"Wrote {args.out} ({args.out.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
