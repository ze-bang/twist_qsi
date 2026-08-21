"""The 96-site cluster: fcc(2,3,4), the size that closes the roton test.

Three things make 96 sites different from 72, and each is handled here.

ENUMERATION.  The depth-first enumerator is far too slow at this size,
so the manifold is built with the vectorized tetrahedron-by-tetrahedron
frontier of ice_enumerate_fast.

SECTOR.  We never sort the whole manifold to label every winding
sector -- that would be a lexsort of billions of rows.  The winding
number is an exact integer vector (positions live on a quarter-integer
lattice, so 4 * polarization is integral), so we pack it into one int64
key, estimate the largest sectors from a subsample, and then extract
each candidate exactly with a single masked pass.  The ground sector is
identified by diagonalizing the candidates and comparing, as at 72
sites, rather than assumed.

ARITHMETIC.  The ring Hamiltonian is real symmetric, so its
eigenvectors can be taken real and a complex probe v = a + ib
contributes |<n|v>|^2 = <n|a>^2 + <n|b>^2.  We therefore run REAL
Lanczos on the real and imaginary parts separately and add the
weights.  This halves the Krylov basis, which is what makes the
continued fraction fit in memory at a sector dimension of order 10^8.

Invoke as:  ring96_pipeline.py 2 3 4 [n_iter]
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
from scipy.linalg import eigh_tridiagonal

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "notes"))
sys.path.insert(0, str(ROOT / "campaign"))

import recompute_finite_size_artifact as geometry               # noqa: E402
from run_momentum_resolved_spectrum import star                 # noqa: E402
from run_ring48_dssf import contractible                        # noqa: E402
from ring48_momenta_fix import fcc_momenta                      # noqa: E402
from ice_enumerate_fast import (enumerate_ice_fast, bit_of,     # noqa: E402
                                site_mask)

OUT = ROOT / "campaign" / "outputs" / "ring_model_dssf"
U64 = np.uint64
CHUNK = 1 << 21
SUBSAMPLE = 4_000_000
N_CAND = 3


def log(m, t0):
    print(f"[{time.perf_counter()-t0:8.1f}s] {m}", flush=True)


class Table128:
    """Sorted (lo,hi) table with exact vectorized lookup."""

    def __init__(self, words):
        order = np.lexsort((words[:, 0], words[:, 1]))
        self.w = np.ascontiguousarray(words[order])
        self.uhi, self.start = np.unique(self.w[:, 1], return_index=True)
        self.end = np.append(self.start[1:], len(self.w))

    def __len__(self):
        return len(self.w)

    def index_of(self, q):
        out = np.full(len(q), -1, dtype=np.int64)
        bpos = np.searchsorted(self.uhi, q[:, 1])
        good = bpos < len(self.uhi)
        good[good] &= (self.uhi[bpos[good]] == q[good, 1])
        for b in np.unique(bpos[good]):
            sel = np.where(good & (bpos == b))[0]
            s, e = self.start[b], self.end[b]
            sub = self.w[s:e, 0]
            p = np.searchsorted(sub, q[sel, 0])
            p = np.clip(p, 0, len(sub) - 1)
            hit = sub[p] == q[sel, 0]
            out[sel[hit]] = s + p[hit]
        return out


def winding_keys(states, ipos, chunk=CHUNK):
    """Exact integer winding key per state, packed into one int64.

    ipos is 4*positions as integers, so the polarization
    sum_i (2 n_i - 1) * ipos_i is an integer vector; it is packed with a
    stride wide enough that distinct vectors never collide.
    """
    nn = ipos.shape[0]
    keys = np.empty(len(states), dtype=np.int64)
    span = int(np.abs(ipos).sum(axis=0).max()) * 2 + 3
    for a in range(0, len(states), chunk):
        b = min(a + chunk, len(states))
        acc = np.zeros((b - a, 3), dtype=np.int64)
        for s in range(nn):
            bit = bit_of(states[a:b], s).astype(np.int64)
            acc += np.outer(2 * bit - 1, ipos[s])
        keys[a:b] = ((acc[:, 0] * span) + acc[:, 1]) * span + acc[:, 2]
    return keys


def hex_words(hexes):
    out = []
    for path in hexes:
        sites = [int(x) for x in path]
        mask = site_mask(sites)
        p1 = site_mask([s for j, s in enumerate(sites) if j % 2 == 0])
        p2 = np.array([mask[0] ^ p1[0], mask[1] ^ p1[1]], dtype=U64)
        out.append((mask, p1, p2))
    return out


def ring_matrix(tab, hw, t0):
    rows, cols = [], []
    W = tab.w
    for i, (mask, p1, p2) in enumerate(hw):
        s0 = W[:, 0] & mask[0]
        s1 = W[:, 1] & mask[1]
        hit = np.where(((s0 == p1[0]) & (s1 == p1[1]))
                       | ((s0 == p2[0]) & (s1 == p2[1])))[0]
        if not len(hit):
            continue
        tgt = np.empty((len(hit), 2), dtype=U64)
        tgt[:, 0] = W[hit, 0] ^ mask[0]
        tgt[:, 1] = W[hit, 1] ^ mask[1]
        pos = tab.index_of(tgt)
        ok = pos >= 0
        rows.append(hit[ok])
        cols.append(pos[ok])
        if i % 24 == 0:
            log(f"    hexagon {i+1}/{len(hw)}", t0)
    r = np.concatenate(rows)
    c = np.concatenate(cols)
    del rows, cols
    return sp.csr_matrix((-np.ones(len(r)), (r, c)),
                         shape=(len(tab), len(tab)))


def nflip_counts(W, hw):
    v = np.zeros(len(W), dtype=np.float64)
    for mask, p1, p2 in hw:
        s0 = W[:, 0] & mask[0]
        s1 = W[:, 1] & mask[1]
        v += (((s0 == p1[0]) & (s1 == p1[1]))
              | ((s0 == p2[0]) & (s1 == p2[1]))).astype(np.float64)
    return v


def lanczos_real(H, seed, e0, n_iter):
    """Real Lanczos with full reorthogonalization.  The Krylov basis is
    real, which halves memory against a complex run and is exact here
    because H is real symmetric."""
    nrm = float(np.linalg.norm(seed))
    if nrm < 1e-14:
        return np.array([]), np.array([]), 0.0
    Q = np.zeros((len(seed), n_iter), dtype=np.float64)
    q = seed / nrm
    Q[:, 0] = q
    alpha, beta = [], []
    r = H @ q
    a = float(q @ r)
    alpha.append(a)
    r -= a * q
    for k in range(1, n_iter):
        r -= Q[:, :k] @ (Q[:, :k].T @ r)
        bb = float(np.linalg.norm(r))
        if bb < 1e-10:
            break
        beta.append(bb)
        q = r / bb
        Q[:, k] = q
        r = H @ q
        a = float(q @ r)
        alpha.append(a)
        r -= a * q + bb * Q[:, k - 1]
    del Q
    ev, evec = eigh_tridiagonal(np.array(alpha), np.array(beta))
    return ev - e0, (nrm ** 2) * evec[0, :] ** 2, nrm ** 2


def main() -> int:
    t0 = time.perf_counter()
    shape = tuple(int(x) for x in sys.argv[1:4]) if len(sys.argv) >= 4 \
        else (2, 3, 4)
    n_iter = int(sys.argv[4]) if len(sys.argv) >= 5 else 120

    orig = geometry.enumerate_ice_states
    geometry.enumerate_ice_states = lambda a, b: np.array([0], dtype=U64)
    cl = geometry.build_cluster("fcc", shape)
    geometry.enumerate_ice_states = orig
    n = int(cl.n_sites)
    positions = np.asarray(cl.positions, dtype=float)
    tets = [tuple(int(s) for s in t) for t in cl.tets]
    hexes = contractible(cl)
    log(f"fcc{shape}: {n} sites, {len(tets)} tetrahedra, "
        f"{len(hexes)} contractible hexagons, n_iter={n_iter}", t0)
    if len(tets) != n // 2 or len(hexes) != n:
        log("  cluster fails the admissibility gate -- aborting", t0)
        return 1

    states = enumerate_ice_fast(n, tets, budget_gb=250.0,
                                log=lambda m: log(m.strip(), t0))
    log(f"enumerated {len(states):,} ice configurations "
        f"({states.nbytes/2**30:.2f} GB)", t0)

    ipos = np.rint(positions * 4).astype(np.int64)
    if not np.allclose(ipos / 4.0, positions):
        log("  positions are not on the quarter-integer lattice", t0)
        return 1

    rng = np.random.default_rng(12345)
    sub = rng.integers(0, len(states),
                       size=min(SUBSAMPLE, len(states)))
    ksub = winding_keys(states[sub], ipos)
    uk, cnt = np.unique(ksub, return_counts=True)
    top = uk[np.argsort(cnt)[::-1][:N_CAND]]
    cand = list(dict.fromkeys([int(x) for x in top] + [0]))
    log(f"{len(uk)} sectors seen in a {len(sub):,} sample (with replacement); "
        f"testing {len(cand)} candidates", t0)
    del ksub, sub, uk, cnt

    keys = winding_keys(states, ipos)
    log("winding keys computed for the full manifold", t0)

    hw = hex_words(hexes)
    best = None
    for c in cand:
        rows = np.where(keys == c)[0]
        if len(rows) < 2:
            log(f"  sector {c}: dimension {len(rows)} -- skipped", t0)
            continue
        tab = Table128(states[rows])
        H = ring_matrix(tab, hw, t0)
        e = float(eigsh(H, k=1, which="SA", tol=1e-9, ncv=30,
                        return_eigenvectors=False)[0])
        log(f"  sector {c}: dim {len(rows):,}, nnz {H.nnz:,}, "
            f"E0 = {e:.8f} K", t0)
        if best is None or e < best[0] - 1e-9:
            best = (e, c, rows, tab, H)
        else:
            del tab, H
    if best is None:
        log("no usable sector found", t0)
        return 1
    e_ref, key, rows, tab, H = best
    n_ice = len(states)
    del states, keys
    log(f"ground sector {key} (dim {len(rows):,}), E0 = {e_ref:.8f} K, "
        f"E0/site = {e_ref/n:.8f}", t0)

    val, vec = eigsh(H, k=1, which="SA", tol=1e-11, ncv=40)
    e0 = float(val[0])
    gs = np.ascontiguousarray(vec[:, 0].astype(np.float64))
    del val, vec

    nf = nflip_counts(tab.w, hw)
    nf_gs = float(np.sum(gs ** 2 * nf))
    del nf
    log(f"<n_flip> ground = {nf_gs:.4f} of {len(hexes)} hexagons "
        f"({nf_gs/len(hexes):.4f} per hexagon)", t0)

    momenta, _ = fcc_momenta(cl)
    rec = {"shape": list(shape), "n_sites": n, "n_ice": int(n_ice),
           "n_hex": len(hexes), "sector_key": int(key),
           "sector_dim": int(len(rows)), "E0": e0, "E0_per_site": e0 / n,
           "nflip_ground": nf_gs, "nflip_per_hex": nf_gs / len(hexes),
           "n_iter": n_iter, "momenta": []}

    subl = np.arange(n) % 4

    def probe_col(q, mu):
        """S^z_mu(q)|0>, built chunked so no (states x sites) array forms."""
        phase = np.exp(-2j * np.pi * (positions @ np.asarray(q)))
        sites = np.where(subl == mu)[0]
        col = np.zeros(len(tab.w), dtype=complex)
        for a in range(0, len(tab.w), CHUNK):
            b = min(a + CHUNK, len(tab.w))
            acc = np.zeros(b - a, dtype=complex)
            for s in sites:
                acc += (bit_of(tab.w[a:b], int(s)) - 0.5) * phase[s]
            col[a:b] = acc * gs[a:b]
        return col

    # ---- pass 1: the ground-state prediction --------------------------
    # S(q) is an equal-time quantity and needs no excited state at all,
    # so it is computed FIRST.  If the L star splits into inequivalent
    # classes, the f_1/S mechanism predicts here, before any spectrum is
    # computed, which class should reconstruct.
    Ls = [q for q in momenta if star(q) == "L"]
    Sq = []
    for q in Ls:
        s = 0.0
        for mu in range(4):
            c = probe_col(q, mu)
            s += float(np.vdot(c, c).real) / n
            del c
        Sq.append(s)
        log(f"  S(L) at {np.round(q, 3)} = {s:.6f}", t0)
    rec["L_structure_factors"] = [{"q": [float(x) for x in q], "S": s}
                                  for q, s in zip(Ls, Sq)]
    classes = []
    for q, s in zip(Ls, Sq):
        if not any(abs(s - s2) < 1e-6 for _, s2 in classes):
            classes.append((q, s))
    log(f"{len(Ls)} L momenta in {len(classes)} distinct class(es) by S; "
        f"running the continued fraction on one representative of each",
        t0)

    # ---- pass 2: the spectrum ----------------------------------------
    for q, s_pred in classes:
        S_q, lines = 0.0, []
        for mu in range(4):
            col = probe_col(q, mu)
            for part in (col.real, col.imag):
                ev, wt, nn = lanczos_real(H, np.ascontiguousarray(part),
                                          e0, n_iter)
                if nn < 1e-14:
                    continue
                S_q += nn / n
                for e, w in zip(ev, wt):
                    if w / n > 1e-6:
                        lines.append((float(e), float(w / n)))
            del col
            log(f"    sublattice {mu} done", t0)
        lines.sort()
        agg = []
        for e, w in lines:
            if agg and abs(agg[-1][0] - e) < 1e-6:
                agg[-1][1] += w
            else:
                agg.append([e, w])
        agg.sort(key=lambda r: -r[1])
        log(f"\nL at {np.round(q, 3)}:  S(L) = {S_q:.5f} "
            f"(equal-time value {s_pred:.5f}, ratio "
            f"{S_q/max(s_pred,1e-30):.4f} -- Lanczos sum-rule gate)", t0)
        for e, w in sorted(agg[:6]):
            log(f"    omega = {e:8.4f} K   share = {w/max(S_q,1e-30):.4f}",
                t0)
        rec["momenta"].append({"q": [float(x) for x in q], "S": S_q,
                               "S_equal_time": s_pred,
                               "lines": [list(x) for x in agg[:12]]})


    tag = "".join(str(s) for s in shape)
    with open(OUT / f"ring96_fcc{tag}.json", "w") as fh:
        json.dump(rec, fh, indent=1, default=float)
    log(f"wrote ring96_fcc{tag}.json", t0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
