"""The 72-site cluster: exact-diagonalization spectroscopy past 64 spins.

Every existing pipeline here stores an ice configuration as a single
uint64 bit mask, which caps the cluster at 64 spins.  The (2,3,3)
torus has 72 sites and is the next cluster that is genuinely
three-dimensional in all three directions -- the alternative 48- and
64-site shapes are one cell thick in one direction and lose half their
contractible hexagons, so they change the Hamiltonian rather than
testing it.

Here a configuration is a PAIR of uint64 words, (lo, hi), and every
operation is redefined on that pair: masking, flipping, sorting, and
lookup.  Lookup is the only nontrivial one.  We sort by (hi, lo), and
because hi < 2^8 for 72 sites there are few distinct hi values; queries
are grouped by hi and each group resolved with one vectorized binary
search on the lo word within that block.  This is exact, not hashed.

Reported at L, for direct comparison with 48 and 64 sites: the ground
energy, the two transverse branch energies and their weights, the
equal-time structure factor S(L), and the Hellmann-Feynman flippability
census -- the four quantities the finite-size argument rests on.
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

OUT = ROOT / "campaign" / "outputs" / "ring_model_dssf"
CHUNK = 1 << 20
U64 = np.uint64


def log(m, t0):
    print(f"[{time.perf_counter()-t0:8.1f}s] {m}", flush=True)


# ---------------------------------------------------------------- enum
def enumerate_ice_python(n_sites, tets):
    """Same recursion as the 64-bit version but keeping Python ints, so
    configurations wider than 64 bits survive."""
    tets_of_site = {}
    for ti, tet in enumerate(tets):
        for s in tet:
            tets_of_site.setdefault(s, []).append(ti)
    assigned = [0] * len(tets)
    ups = [0] * len(tets)
    states = []

    def rec(site, state):
        if site == n_sites:
            states.append(state)
            return
        for spin in (0, 1):
            ok = True
            for tid in tets_of_site[site]:
                u = ups[tid] + spin
                d = assigned[tid] + 1 - u
                if u > 2 or d > 2:
                    ok = False
                    break
            if not ok:
                continue
            for tid in tets_of_site[site]:
                assigned[tid] += 1
                ups[tid] += spin
            rec(site + 1, state | (spin << site))
            for tid in tets_of_site[site]:
                assigned[tid] -= 1
                ups[tid] -= spin

    rec(0, 0)
    return states


def split_words(pyints):
    """List of Python ints -> (M,2) uint64 array [lo, hi]."""
    m = len(pyints)
    lo = np.empty(m, dtype=U64)
    hi = np.empty(m, dtype=U64)
    mask64 = (1 << 64) - 1
    for i, v in enumerate(pyints):
        lo[i] = v & mask64
        hi[i] = (v >> 64) & mask64
    return np.stack([lo, hi], axis=1)


class Table128:
    """Sorted (lo,hi) state table with exact vectorized lookup."""

    def __init__(self, words):
        order = np.lexsort((words[:, 0], words[:, 1]))
        self.w = words[order]
        self.uhi, self.start = np.unique(self.w[:, 1], return_index=True)
        self.end = np.append(self.start[1:], len(self.w))

    def __len__(self):
        return len(self.w)

    def index_of(self, q):
        """q: (M,2) uint64 -> int64 positions, -1 where absent."""
        out = np.full(len(q), -1, dtype=np.int64)
        bpos = np.searchsorted(self.uhi, q[:, 1])
        good = (bpos < len(self.uhi))
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


def hex_words(hexes, n_sites):
    """(mask, pattern1, pattern2) per hexagon, each as a (2,) uint64."""
    out = []
    for path in hexes:
        mask = 0
        p1 = 0
        for j, site in enumerate(path):
            mask |= (1 << site)
            if j % 2 == 0:
                p1 |= (1 << site)
        p2 = mask ^ p1
        out.append(tuple(split_words([mask, p1, p2])))
    return out


def bits_of(words, sites):
    """(M, len(sites)) float array of the requested site occupations."""
    m = len(words)
    out = np.empty((m, len(sites)), dtype=np.float64)
    for j, s in enumerate(sites):
        col = 0 if s < 64 else 1
        sh = U64(s if s < 64 else s - 64)
        out[:, j] = ((words[:, col] >> sh) & U64(1)).astype(np.float64)
    return out


def ring_matrix(tab, hw):
    rows, cols = [], []
    W = tab.w
    for mask, p1, p2 in hw:
        sel0 = W[:, 0] & mask[0]
        sel1 = W[:, 1] & mask[1]
        hit = np.where(((sel0 == p1[0]) & (sel1 == p1[1]))
                       | ((sel0 == p2[0]) & (sel1 == p2[1])))[0]
        if not len(hit):
            continue
        tgt = np.empty((len(hit), 2), dtype=U64)
        tgt[:, 0] = W[hit, 0] ^ mask[0]
        tgt[:, 1] = W[hit, 1] ^ mask[1]
        pos = tab.index_of(tgt)
        ok = pos >= 0
        rows.append(hit[ok])
        cols.append(pos[ok])
    r = np.concatenate(rows)
    c = np.concatenate(cols)
    return sp.csr_matrix((-np.ones(len(r)), (r, c)),
                         shape=(len(tab), len(tab)))


def nflip_counts(W, hw):
    v = np.zeros(len(W))
    for mask, p1, p2 in hw:
        sel0 = W[:, 0] & mask[0]
        sel1 = W[:, 1] & mask[1]
        v += (((sel0 == p1[0]) & (sel1 == p1[1]))
              | ((sel0 == p2[0]) & (sel1 == p2[1]))).astype(float)
    return v


def lanczos_cf(H, seed, e0, n_iter=250):
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

    # build geometry without tripping the 64-bit enumeration
    orig = geometry.enumerate_ice_states
    geometry.enumerate_ice_states = lambda n, t: np.array([0], dtype=U64)
    cl = geometry.build_cluster("fcc", shape)
    geometry.enumerate_ice_states = orig
    n = int(cl.n_sites)
    positions = np.asarray(cl.positions, dtype=float)
    tets = [tuple(int(s) for s in t) for t in cl.tets]
    hexes = contractible(cl)
    log(f"fcc{shape}: {n} sites, {len(tets)} tetrahedra, "
        f"{len(hexes)} contractible hexagons", t0)
    if len(tets) != n // 2:
        log("  SPURIOUS TETRAHEDRA -- aborting", t0)
        return 1

    py = enumerate_ice_python(n, tets)
    log(f"enumerated {len(py):,} ice configurations", t0)
    words = split_words(py)
    del py
    tab = Table128(words)
    del words
    log(f"state table built ({len(tab):,} entries)", t0)

    # --- transport sectors, chunked -----------------------------------
    pol = np.empty((len(tab), 3))
    for a in range(0, len(tab), CHUNK):
        b = min(a + CHUNK, len(tab))
        bits = bits_of(tab.w[a:b], range(n))
        pol[a:b] = (bits - 0.5) @ positions
    uniq, labels = np.unique(np.round(pol, 6), axis=0, return_inverse=True)
    del pol
    sizes = np.bincount(labels)
    zero = int(np.argmin(np.abs(uniq).sum(axis=1)))
    log(f"{len(sizes)} transport sectors; zero-polarization sector "
        f"{zero} has dimension {sizes[zero]:,} "
        f"(largest is {sizes.max():,})", t0)

    # --- diagonalize the three largest sectors as a ground-sector gate --
    cand = list(np.argsort(sizes)[::-1][:3])
    if zero not in cand:
        cand.append(zero)
    best = None
    for sc in cand:
        rows = np.where(labels == sc)[0]
        sub = Table128(tab.w[rows])
        hw = hex_words(hexes, n)
        H = ring_matrix(sub, hw)
        val = eigsh(H, k=1, which="SA", tol=1e-10, ncv=60,
                    return_eigenvectors=False)
        e = float(val[0])
        log(f"  sector {sc}: dim {len(rows):,}, nnz {H.nnz:,}, "
            f"E0 = {e:.8f} K", t0)
        if best is None or e < best[1]:
            best = (sc, e, rows, sub, H)
    sector, e_ref, rows, sub, H = best
    log(f"ground sector {sector} (dim {len(rows):,}), E0 = {e_ref:.8f} K, "
        f"E0/site = {e_ref/n:.8f}", t0)

    val, vec = eigsh(H, k=1, which="SA", tol=1e-11, ncv=80)
    e0 = float(val[0])
    gs = vec[:, 0].astype(complex)

    hw = hex_words(hexes, n)
    nf = nflip_counts(sub.w, hw)
    nf_gs = float(np.sum(np.abs(gs) ** 2 * nf))
    log(f"<n_flip> ground = {nf_gs:.4f} of {len(hexes)} hexagons "
        f"({nf_gs/len(hexes):.4f} per hexagon)", t0)

    momenta, _ = fcc_momenta(cl)
    rec = {"shape": list(shape), "n_sites": n, "n_ice": int(len(tab)),
           "n_hex": len(hexes), "sector": int(sector),
           "sector_dim": int(len(rows)), "E0": e0, "E0_per_site": e0 / n,
           "nflip_ground": nf_gs, "nflip_per_hex": nf_gs / len(hexes),
           "momenta": []}

    subl = np.arange(n) % 4
    done = 0
    for q in momenta:
        if star(q) != "L":
            continue
        phase = np.exp(-2j * np.pi * (positions @ np.asarray(q)))
        S_q, lines = 0.0, []
        for mu in range(4):
            sites = np.where(subl == mu)[0]
            col = np.zeros(len(sub.w), dtype=complex)
            for a in range(0, len(sub.w), CHUNK):
                b = min(a + CHUNK, len(sub.w))
                spins = bits_of(sub.w[a:b], sites) - 0.5
                col[a:b] = (spins @ phase[sites]) * gs[a:b]
            ev, wt, nn = lanczos_cf(H, col, e0)
            if nn < 1e-14:
                continue
            S_q += nn / n
            for e, w in zip(ev, wt):
                if w / n > 1e-5:
                    lines.append((float(e), float(w / n)))
        lines.sort(key=lambda p: -p[1])
        top = lines[:2]
        top.sort()
        log(f"\nL at {np.round(q,3)}:  S(L) = {S_q:.5f}", t0)
        for e, w in lines[:6]:
            log(f"    omega = {e:8.4f} K   share = {w/max(S_q,1e-30):.4f}",
                t0)
        rec["momenta"].append({"q": [float(x) for x in q], "S": S_q,
                               "lines": lines[:12],
                               "branches": top})
        done += 1
        if done >= 2:
            break

    tag = "".join(str(s) for s in shape)
    with open(OUT / f"ring128_fcc{tag}.json", "w") as fh:
        json.dump(rec, fh, indent=1, default=float)
    log(f"wrote ring128_fcc{tag}.json", t0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
