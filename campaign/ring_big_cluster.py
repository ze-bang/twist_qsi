"""Push the roton calculation past 64 sites.

The 64-site pipeline materializes an (n_states x n_sites) float array to
assign transport sectors; that is 8.6 GB at 72 sites and 55 GB at 80, so
everything here is chunked and the sector is extracted before any
per-site array is built.

Steps: enumerate the ice manifold -> label transport sectors in chunks ->
locate the sector holding the ground state by a sparse Lanczos on the
full manifold -> diagonalize inside that sector -> S^zz(q,omega) at the
L point by Lanczos continued fraction, giving the roton energy and its
share of the neutron weight at a third system size.

Usage:  python ring_big_cluster.py 2 3 3
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "notes"))
sys.path.insert(0, str(ROOT / "campaign"))

import recompute_finite_size_artifact as geometry               # noqa: E402
from run_momentum_resolved_spectrum import star                 # noqa: E402
from run_ring48_dssf import contractible                        # noqa: E402
from ring48_momenta_fix import fcc_momenta                      # noqa: E402
from ring64_pipeline import hex_masks_patterns                  # noqa: E402

OUT = ROOT / "campaign" / "outputs" / "ring_model_dssf"
CHUNK = 1 << 21


def log(msg, t0):
    print(f"[{time.perf_counter()-t0:7.1f}s] {msg}", flush=True)


def sector_labels_chunked(states, positions):
    """Transport-sector label of every ice state, without ever holding an
    (n_states x n_sites) array."""
    n = positions.shape[0]
    shifts = np.arange(n, dtype=np.uint64)
    pol = np.empty((len(states), 3), dtype=np.float64)
    for a in range(0, len(states), CHUNK):
        b = min(a + CHUNK, len(states))
        bits = ((states[a:b, None] >> shifts) & np.uint64(1)).astype(
            np.float64)
        pol[a:b] = (bits - 0.5) @ positions
    uniq, labels = np.unique(np.round(pol, 6), axis=0,
                            return_inverse=True)
    return labels.astype(np.int32), uniq


def ring_matrix(states, hmp, index_of):
    """Sparse -sum_p (W_p + W_p^dag) on the given (sorted) state list."""
    rows, cols = [], []
    for mask, p1, p2 in hmp:
        sel = states & mask
        idx = np.where((sel == p1) | (sel == p2))[0]
        if not len(idx):
            continue
        tgt = states[idx] ^ mask
        pos = index_of(tgt)
        ok = pos >= 0
        rows.append(idx[ok])
        cols.append(pos[ok])
    if not rows:
        return sp.csr_matrix((len(states), len(states)))
    r = np.concatenate(rows)
    c = np.concatenate(cols)
    d = -np.ones(len(r))
    return sp.csr_matrix((d, (r, c)), shape=(len(states), len(states)))


def searcher(sorted_states):
    def index_of(vals):
        pos = np.searchsorted(sorted_states, vals)
        pos = np.clip(pos, 0, len(sorted_states) - 1)
        bad = sorted_states[pos] != vals
        pos = pos.astype(np.int64)
        pos[bad] = -1
        return pos
    return index_of


def probe_cols_chunked(sec_states, positions, q, gs, n):
    """S^z_mu(q)|gs> for mu = 0..3, chunked over states."""
    phase = np.exp(-2j * np.pi * (positions @ q))
    sub = np.arange(n) % 4
    shifts = np.arange(n, dtype=np.uint64)
    cols = [np.zeros(len(sec_states), dtype=complex) for _ in range(4)]
    for a in range(0, len(sec_states), CHUNK):
        b = min(a + CHUNK, len(sec_states))
        spins = ((sec_states[a:b, None] >> shifts) & np.uint64(1)
                 ).astype(np.float64) - 0.5
        for mu in range(4):
            cols[mu][a:b] = (spins @ (phase * (sub == mu))) * gs[a:b]
    return cols


def lanczos_cf(H, seed, e0, n_iter=300):
    """Ritz energies and weights from one seed (full reorthogonalization)."""
    nrm = float(np.linalg.norm(seed))
    if nrm < 1e-14:
        return np.array([]), np.array([]), 0.0
    Q = np.zeros((len(seed), n_iter), dtype=complex)
    q = seed / nrm
    Q[:, 0] = q
    alpha, beta = [], []
    r = H @ q
    a = float(np.vdot(q, r).real)
    alpha.append(a)
    r = r - a * q
    for k in range(1, n_iter):
        r -= Q[:, :k] @ (Q[:, :k].conj().T @ r)
        bb = float(np.linalg.norm(r))
        if bb < 1e-11:
            break
        beta.append(bb)
        q = r / bb
        Q[:, k] = q
        r = H @ q
        a = float(np.vdot(q, r).real)
        alpha.append(a)
        r = r - a * q - bb * Q[:, k - 1]
    from scipy.linalg import eigh_tridiagonal
    ev, evec = eigh_tridiagonal(np.array(alpha), np.array(beta))
    return ev - e0, (nrm ** 2) * np.abs(evec[0, :]) ** 2, nrm ** 2


def main() -> int:
    t0 = time.perf_counter()
    shape = tuple(int(x) for x in sys.argv[1:4]) if len(sys.argv) >= 4 \
        else (2, 3, 3)
    cl = geometry.build_cluster("fcc", shape)
    n = int(cl.n_sites)
    states = np.sort(np.asarray(cl.ice_states, dtype=np.uint64))
    positions = np.asarray(cl.positions, dtype=float)
    hexes = contractible(cl)
    hmp = hex_masks_patterns(hexes)
    log(f"fcc{shape}: {n} sites, {len(states):,} ice states, "
        f"{len(hexes)} contractible hexagons", t0)

    labels, uniq = sector_labels_chunked(states, positions)
    sizes = np.bincount(labels)
    log(f"{len(sizes)} transport sectors, largest {sizes.max():,}", t0)

    idx_all = searcher(states)
    H = ring_matrix(states, hmp, idx_all)
    log(f"full ring matrix: {H.nnz:,} nonzeros", t0)
    vals, vecs = eigsh(H, k=1, which="SA", tol=1e-10, ncv=40)
    e_full = float(vals[0])
    occ = np.bincount(labels, weights=np.abs(vecs[:, 0]) ** 2,
                      minlength=len(sizes))
    sector = int(np.argmax(occ))
    log(f"E0(full) = {e_full:.6f} K, ground sector {sector} "
        f"(dim {sizes[sector]:,}, weight {occ[sector]:.6f})", t0)

    rows = np.where(labels == sector)[0]
    sec = states[rows]
    Hs = ring_matrix(sec, hmp, searcher(sec))
    log(f"sector matrix: {Hs.nnz:,} nonzeros", t0)
    v0, w0 = eigsh(Hs, k=1, which="SA", tol=1e-11, ncv=60)
    e0 = float(v0[0])
    gs = w0[:, 0].astype(complex)
    log(f"E0(sector) = {e0:.6f} K  (matches full: "
        f"{abs(e0 - e_full):.2e})", t0)

    momenta, _ = fcc_momenta(cl)
    rec = {"shape": list(shape), "n_sites": n,
           "n_ice": int(len(states)), "sector": sector,
           "sector_dim": int(len(rows)), "E0": e0, "momenta": []}
    for qi, q in enumerate(momenta):
        lab = star(q)
        if lab != "L":
            continue
        cols = probe_cols_chunked(sec, positions, np.asarray(q), gs, n)
        tot, lines = 0.0, []
        for mu, col in enumerate(cols):
            ev, wt, nn = lanczos_cf(Hs, col, e0)
            if nn < 1e-12:
                continue
            tot += nn / n
            for e, w in zip(ev, wt):
                if w / n > 1e-4:
                    lines.append((float(e), float(w / n)))
        lines.sort(key=lambda p: -p[1])
        log(f"L momentum {np.round(q,3)}: S(q) = {tot:.5f}", t0)
        for e, w in lines[:6]:
            log(f"    omega = {e:8.4f} K   share = {w/max(tot,1e-30):.3f}",
                t0)
        rec["momenta"].append({"q": [float(x) for x in q], "S": tot,
                               "lines": lines[:12]})
        if len(rec["momenta"]) >= 2:
            break

    tag = "".join(str(s) for s in shape)
    with open(OUT / f"ring_big_fcc{tag}.json", "w") as fh:
        json.dump(rec, fh, indent=1, default=float)
    log(f"wrote ring_big_fcc{tag}.json", t0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
