#!/usr/bin/env python3
"""Full 16-site exact diagonalization → thermodynamics from the spectrum.

Builds H from a legacy QED deck (``InterAll.dat``, ``Trans.dat``,
``ThreeBodyG.dat``) using the same convention the C++ matvec uses
(bit 0 ↔ up / +½, bit 1 ↔ down / −½; ``op_type`` 0=S+, 1=S-, 2=Sz)
and diagonalises in the four momentum sectors of the fcc translation
subgroup of the 1×1×1 conventional cubic cluster.

The cluster has three primitive-fcc translation generators

    T1 = (½, ½, 0),  T2 = (½, 0, ½),  T3 = (0, ½, ½) = T1·T2,

each an involution under PBC.  The independent translation group is
therefore :math:`\\mathbb{Z}_2 \\times \\mathbb{Z}_2` of order four,
labelled by four 1-D irreps :math:`(s_1, s_2)\\in\\{\\pm1\\}^2`.  These
characters are **exact symmetries of the deck** (verified numerically
against ``InterAll.dat`` and ``ThreeBodyG.dat``), so projecting onto
each sector gives a complete block decomposition into ~16384-dim
sub-blocks.

Each block is densified (complex Hermitian if ``J3≠0``, real symmetric
otherwise) and diagonalised with ``scipy.linalg.eigh`` (driver ``evd``,
``eigvals_only=True``).  Memory is at most ≈4 GB per block in complex
storage, and the LAPACK 'N' workspace is small.
"""
from __future__ import annotations

import argparse
import functools
import sys
import time
from pathlib import Path
from typing import Tuple

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp

print = functools.partial(print, flush=True)  # so logs survive pipe buffering

N_SITES_DEFAULT = 16


# ---------------------------------------------------------------------------
# Legacy deck parsing
# ---------------------------------------------------------------------------

def parse_trans(path: Path):
    rows = []
    if not path.exists():
        return rows
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) == 4:
                op, site, re, im = parts
                c = complex(float(re), float(im))
                rows.append((int(op), int(site), c))
    return rows


def parse_interall(path: Path):
    rows = []
    if not path.exists():
        return rows
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) == 6:
                op1, s1, op2, s2, re, im = parts
                c = complex(float(re), float(im))
                rows.append((int(op1), int(s1), int(op2), int(s2), c))
    return rows


def parse_threebody(path: Path):
    rows = []
    if not path.exists():
        return rows
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) == 8:
                op1, s1, op2, s2, op3, s3, re, im = parts
                c = complex(float(re), float(im))
                rows.append((int(op1), int(s1), int(op2), int(s2), int(op3), int(s3), c))
    return rows


# ---------------------------------------------------------------------------
# Vectorised ladder/diagonal action on the integer basis
# ---------------------------------------------------------------------------

def apply_op_vec(states: np.ndarray, op: int, site: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply a single-site operator to a vector of basis indices.

    Returns (new_states, amplitudes, valid_mask).
    """
    bits = ((states >> np.uint64(site)) & np.uint64(1)).astype(np.int8)
    if op == 2:  # Sz
        amps = np.where(bits == 0, 0.5, -0.5).astype(np.float64)
        return states, amps, np.ones(states.shape, dtype=bool)
    if op == 0:  # S+: down(1) -> up(0)
        valid = bits == 1
        new = states ^ (np.uint64(1) << np.uint64(site))
        new = np.where(valid, new, states)
        return new, valid.astype(np.float64), valid
    if op == 1:  # S-: up(0) -> down(1)
        valid = bits == 0
        new = states ^ (np.uint64(1) << np.uint64(site))
        new = np.where(valid, new, states)
        return new, valid.astype(np.float64), valid
    raise ValueError(f"unknown op code {op}")


def build_h_sparse(ham_dir: Path, n_sites: int) -> sp.csr_matrix:
    dim = 1 << n_sites
    states = np.arange(dim, dtype=np.uint64)

    trans = parse_trans(ham_dir / "Trans.dat")
    inter = parse_interall(ham_dir / "InterAll.dat")
    three = parse_threebody(ham_dir / "ThreeBodyG.dat")
    print(f"  parsed: {len(trans)} one-body, {len(inter)} two-body, {len(three)} three-body terms")

    rows_list, cols_list, vals_list = [], [], []

    def add_term(out_states: np.ndarray, amps: np.ndarray, valid: np.ndarray, coeff: complex) -> None:
        v = coeff * amps
        mask = valid & (np.abs(v) > 1e-15)
        if not mask.any():
            return
        rows_list.append(out_states[mask].astype(np.int64, copy=False))
        cols_list.append(states[mask].astype(np.int64, copy=False))
        vals_list.append(v[mask])

    for op, site, c in trans:
        new, amps, valid = apply_op_vec(states, op, site)
        add_term(new, amps, valid, c)

    for op1, s1, op2, s2, c in inter:
        sa, a1, v1 = apply_op_vec(states, op1, s1)
        sb, a2, v2 = apply_op_vec(sa, op2, s2)
        add_term(sb, a1 * a2, v1 & v2, c)

    for op1, s1, op2, s2, op3, s3, c in three:
        sa, a1, v1 = apply_op_vec(states, op1, s1)
        sb, a2, v2 = apply_op_vec(sa, op2, s2)
        sc, a3, v3 = apply_op_vec(sb, op3, s3)
        add_term(sc, a1 * a2 * a3, v1 & v2 & v3, c)

    if not vals_list:
        raise RuntimeError(f"No matrix elements assembled from {ham_dir}")

    rows = np.concatenate(rows_list)
    cols = np.concatenate(cols_list)
    vals = np.concatenate(vals_list)
    print(f"  COO triplets: {len(vals)}")

    H = sp.coo_matrix((vals, (rows, cols)), shape=(dim, dim)).tocsr()
    print(f"  CSR nnz: {H.nnz}")
    return H


# ---------------------------------------------------------------------------
# Thermodynamics from a complete spectrum
# ---------------------------------------------------------------------------

def fcc_site_permutations() -> Tuple[np.ndarray, np.ndarray]:
    """Return T1, T2 as 16-site permutations (site i -> perm[i])."""
    T1 = np.array([12,13,14,15, 8, 9,10,11, 4, 5, 6, 7, 0, 1, 2, 3],
                  dtype=np.int64)
    T2 = np.array([ 8, 9,10,11,12,13,14,15, 0, 1, 2, 3, 4, 5, 6, 7],
                  dtype=np.int64)
    return T1, T2


def _bit_perm_vec(states: np.ndarray, perm: np.ndarray, n_sites: int) -> np.ndarray:
    """Apply a site permutation to a vector of basis indices.

    For each bit position ``i`` of the original state, route that bit
    to position ``perm[i]`` of the new state.
    """
    out = np.zeros_like(states)
    one = np.int64(1)
    for i in range(n_sites):
        bit = (states >> np.int64(i)) & one
        out |= bit << np.int64(int(perm[i]))
    return out


def build_translation_sectors(n_sites: int):
    """Build orbit structure and sparse projectors for the 4 sectors of
    Z_2 x Z_2 (fcc translations).

    Returns dict ``{(s1, s2): (P_alpha, dim_alpha)}`` where ``P_alpha``
    is a sparse ``(dim_alpha, D)`` matrix with rows that are
    orthonormal basis vectors in sector ``α``.
    """
    if n_sites != 16:
        raise NotImplementedError("translation sectors are hard-coded for 16-site pyrochlore")
    T1_perm, T2_perm = fcc_site_permutations()
    dim = 1 << n_sites
    states = np.arange(dim, dtype=np.int64)

    T1_states = _bit_perm_vec(states, T1_perm, n_sites)
    T2_states = _bit_perm_vec(states, T2_perm, n_sites)
    T12_states = _bit_perm_vec(T1_states, T2_perm, n_sites)

    # Sanity: T1, T2 are involutions on the basis.
    assert np.array_equal(_bit_perm_vec(T1_states, T1_perm, n_sites), states)
    assert np.array_equal(_bit_perm_vec(T2_states, T2_perm, n_sites), states)

    # Orbit representative = min over the 4 group images.
    orbit_members = np.stack([states, T1_states, T2_states, T12_states])
    rep_of = orbit_members.min(axis=0)

    unique_reps = np.unique(rep_of)
    # Stabiliser of each rep (which non-identity group elements fix it).
    rep_T1 = T1_states[unique_reps]
    rep_T2 = T2_states[unique_reps]
    rep_T12 = T12_states[unique_reps]
    stab_T1 = (rep_T1 == unique_reps)
    stab_T2 = (rep_T2 == unique_reps)
    stab_T12 = (rep_T12 == unique_reps)
    stab_size = 1 + stab_T1.astype(int) + stab_T2.astype(int) + stab_T12.astype(int)
    orbit_size = 4 // stab_size

    size_counts = {int(s): int((orbit_size == s).sum()) for s in [4, 2, 1]}
    print(f"  {len(unique_reps)} orbits: " +
          ", ".join(f"{v} of size {k}" for k, v in size_counts.items() if v))

    sectors: dict = {}
    # Precompute the four image arrays at reps for convenience.
    image_at_rep = [unique_reps, rep_T1, rep_T2, rep_T12]

    for s1 in (+1, -1):
        for s2 in (+1, -1):
            chi = np.array([1, s1, s2, s1 * s2], dtype=np.int64)
            # A sector lives on an orbit iff its character is trivial on the stabiliser.
            lives = np.ones(len(unique_reps), dtype=bool)
            lives &= ~(stab_T1 & (chi[1] != 1))
            lives &= ~(stab_T2 & (chi[2] != 1))
            lives &= ~(stab_T12 & (chi[3] != 1))

            rows = []
            cols = []
            vals = []
            row_idx = 0
            o_idx_arr = np.flatnonzero(lives)
            for o_idx in o_idx_arr:
                norm = 1.0 / np.sqrt(orbit_size[o_idx])
                # Build unique (state, group_label) map, with smallest g.
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
            sectors[(s1, s2)] = (P, row_idx)
            print(f"  sector (s1={s1:+d}, s2={s2:+d}): dim={row_idx}")
    return sectors


def block_diagonalize(H: sp.csr_matrix, n_sites: int):
    """Return ``{(s1, s2): H_sector_dense}`` projecting H onto each
    translation sector.
    """
    sectors = build_translation_sectors(n_sites)
    blocks = {}
    for label, (P, dim) in sectors.items():
        # H_alpha = P H P^T (P is real)
        # For complex H the projection picks up complex entries through H.
        Hp = (P @ H @ P.T).toarray()
        Hp = 0.5 * (Hp + Hp.conj().T)
        blocks[label] = Hp
    return blocks


def compute_thermo(evals: np.ndarray, t_grid: np.ndarray) -> dict:
    evals = np.asarray(evals, dtype=np.float64)
    e_min = float(evals.min())
    e_max = float(evals.max())
    energy = np.zeros_like(t_grid)
    heat = np.zeros_like(t_grid)
    entropy = np.zeros_like(t_grid)
    free_energy = np.zeros_like(t_grid)
    shifted = evals - e_min
    for i, T in enumerate(t_grid):
        beta = 1.0 / T
        w = np.exp(-beta * shifted)
        Z = w.sum()
        E = (evals * w).sum() / Z
        E2 = (evals * evals * w).sum() / Z
        energy[i] = E
        heat[i] = (E2 - E * E) / (T * T)
        # Z_shifted = sum exp(-beta(E_n - e_min))  ⇒  log Z = log Z_shifted - beta e_min
        free_energy[i] = e_min - T * np.log(Z)
        entropy[i] = (E - free_energy[i]) / T
    return {
        "temperatures": t_grid,
        "energy": energy,
        "specific_heat": heat,
        "entropy": entropy,
        "free_energy": free_energy,
        "eigenvalue_min": e_min,
        "eigenvalue_max": e_max,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ham", required=True, type=Path, help="Hamiltonian directory")
    ap.add_argument("--out", required=True, type=Path, help="output NPZ path")
    ap.add_argument("--n-sites", type=int, default=N_SITES_DEFAULT)
    ap.add_argument("--temp-min", type=float, default=0.005)
    ap.add_argument("--temp-max", type=float, default=5.0)
    ap.add_argument("--temp-points", type=int, default=80)
    ap.add_argument("--force-complex", action="store_true",
                    help="use complex Hermitian path even if H is detectably real")
    ap.add_argument("--no-translation", action="store_true",
                    help="diagonalise the full D=2^N space (no Z2xZ2 translation block)")
    ap.add_argument("--driver", default="evd", choices=["evd", "evr", "ev"],
                    help="LAPACK driver to use in scipy.linalg.eigh")
    args = ap.parse_args()

    if args.out.exists():
        print(f"output {args.out} already exists; refusing to overwrite")
        return

    print(f"=== Full ED for {args.ham} (n_sites={args.n_sites})")
    t0 = time.time()
    H = build_h_sparse(args.ham, args.n_sites)
    print(f"  sparse build: {time.time() - t0:.1f}s")

    diff = H - H.conj().T
    diff_abs = abs(diff).max() if diff.nnz else 0.0
    print(f"  Hermiticity |H-H†|_max = {diff_abs:.3e}")

    Hh = 0.5 * (H + H.conj().T)
    imag_csr = Hh.imag
    max_imag = abs(imag_csr).max() if imag_csr.nnz else 0.0
    print(f"  max |Im H| = {max_imag:.3e}")
    use_real = (not args.force_complex) and max_imag < 1e-10

    extra = {}
    if args.no_translation:
        print(f"Densifying full {'real symmetric' if use_real else 'complex Hermitian'}...")
        t0 = time.time()
        if use_real:
            Hd = Hh.real.toarray().astype(np.float64, copy=False)
        else:
            Hd = Hh.toarray().astype(np.complex128, copy=False)
        del H, Hh
        print(f"  dense shape={Hd.shape} dtype={Hd.dtype} "
              f"mem={Hd.nbytes / 1e9:.2f} GB in {time.time() - t0:.1f}s")
        print(f"scipy.linalg.eigh(eigvals_only=True, driver='{args.driver}')...")
        t0 = time.time()
        evals = la.eigh(Hd, eigvals_only=True, driver=args.driver)
        print(f"  diagonalised in {time.time() - t0:.1f}s; "
              f"E_min={evals[0]:.8f} E_max={evals[-1]:.8f}")
        del Hd
    else:
        t0 = time.time()
        # For real H we can stay in float64 to save memory; complex H
        # gives complex blocks.
        H_use = Hh.real if use_real else Hh
        blocks = block_diagonalize(H_use, args.n_sites)
        del H, Hh, H_use
        total_dim = sum(b.shape[0] for b in blocks.values())
        if total_dim != (1 << args.n_sites):
            raise RuntimeError(f"sector dims sum to {total_dim}, expected {1<<args.n_sites}")
        total_gb = sum(b.nbytes for b in blocks.values()) / 1e9
        print(f"  4 blocks built in {time.time() - t0:.1f}s; total mem ≈ {total_gb:.2f} GB")
        ev_pieces = []
        for label in sorted(blocks.keys(), reverse=True):
            block = blocks[label]
            tag = f"({label[0]:+d},{label[1]:+d})"
            print(f"Diagonalising sector {tag} dim={block.shape[0]} dtype={block.dtype} ...")
            t0 = time.time()
            ev = la.eigh(block, eigvals_only=True, driver=args.driver)
            print(f"  sector {tag} done in {time.time() - t0:.1f}s; "
                  f"E_min={ev[0]:.8f} E_max={ev[-1]:.8f}")
            ev_pieces.append(ev)
            extra[f"eigenvalues_sector_{label[0]:+d}_{label[1]:+d}"] = ev
            blocks[label] = None  # free immediately
        evals = np.concatenate(ev_pieces)
        evals.sort()
        print(f"  combined spectrum: E_min={evals[0]:.8f} E_max={evals[-1]:.8f} "
              f"(n_total={len(evals)})")

    t_grid = np.logspace(np.log10(args.temp_min), np.log10(args.temp_max), args.temp_points)
    print(f"Thermodynamics on {len(t_grid)} log-spaced T points...")
    thermo = compute_thermo(evals, t_grid)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, eigenvalues=evals, **extra, **thermo)
    print(f"Wrote {args.out}  ({args.out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
