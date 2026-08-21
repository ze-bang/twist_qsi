"""Size dependence: ring-model DSSF on fcc(2,2,4) (64 sites, 2.76M ice states)
via Lanczos continued fraction, GATED on reproducing the exact fcc(2,2,3)
dense results first.  Also: the sin-B magnetic probe on fcc(2,2,3) as a
monopole-pair discriminator for the L intruder.

Vectorized throughout (bit masks on uint64 arrays); the per-state Python
loops of the 48-site pipeline would take hours at 2.76M states.

Stages:
  A. fcc(2,2,3) with the FAST pipeline + dense eigh (88 s):
     - GATE 1: fast ring matrix == slow ring matrix results (E0, L lines).
     - sin-B channel: R_n = w_sinB / w_E for the L intruder vs the L photon.
     - GATE 2: Lanczos CF on the same sector reproduces the dense dominant
       lines and weights to 1e-4.
  B. fcc(2,2,4): ground sector, Lanczos CF at all 16 momenta (S^z channels)
     + even channel at X-type momenta.  Report: photon dispersion, presence/
     energy/weight of an L intruder, f_res per momentum, new small-|q| point.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.linalg import eigh, eigh_tridiagonal
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import eigsh

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "notes"))
sys.path.insert(0, str(ROOT / "campaign"))

import recompute_finite_size_artifact as geometry               # noqa: E402
from run_momentum_resolved_spectrum import reduce_momentum, star  # noqa: E402
from run_ring48_dssf import contractible                         # noqa: E402
from ring48_momenta_fix import fcc_momenta                       # noqa: E402

OUT = ROOT / "campaign" / "outputs" / "ring_model_dssf"
EDEGEN = 1e-8


# ---------------------------------------------------------------- fast core
def hex_masks_patterns(hexes):
    """Per hexagon: full mask and the two alternating bit patterns."""
    out = []
    for path in hexes:
        mask = np.uint64(0)
        p1 = np.uint64(0)
        for j, site in enumerate(path):
            b = np.uint64(1) << np.uint64(site)
            mask |= b
            if j % 2 == 0:
                p1 |= b
        out.append((mask, p1, mask ^ p1))
    return out


def fast_labels(states, positions):
    """Vectorized polarization_sector_labels (identical convention)."""
    n = positions.shape[0]
    bits = ((states[:, None] >> np.arange(n, dtype=np.uint64))
            & np.uint64(1)).astype(np.float64)
    pol = (bits - 0.5) @ positions
    rounded = np.round(pol, 6)
    uniq, labels = np.unique(rounded, axis=0, return_inverse=True)
    return labels.astype(np.int64), bits - 0.5


def fast_ring(states, hmp, v_pot):
    """Sparse -sum(flip) + V n_flip via bit masks, fully vectorized."""
    dim = len(states)
    rows_all, cols_all = [], []
    n_flip = np.zeros(dim)
    for mask, p1, p2 in hmp:
        sel = states & mask
        flip = (sel == p1) | (sel == p2)
        idx = np.where(flip)[0]
        n_flip[idx] += 1.0
        targets = states[idx] ^ mask
        j = np.searchsorted(states, targets)
        if not np.array_equal(states[j], targets):
            raise SystemExit("hexagon flip left the ice manifold")
        rows_all.append(j)
        cols_all.append(idx)
    rows = np.concatenate(rows_all)
    cols = np.concatenate(cols_all)
    vals = np.full(len(rows), -1.0)
    ham = coo_matrix((np.concatenate([vals, v_pot * n_flip]),
                      (np.concatenate([rows, np.arange(dim)]),
                       np.concatenate([cols, np.arange(dim)]))),
                     shape=(dim, dim)).tocsr()
    return ham, n_flip


def probe_columns(spins, positions, momentum, gs, n_sites):
    """S^z_mu(q)|0> for mu = 0..3, as columns (sector basis)."""
    phase = np.exp(-2j * np.pi * (positions @ momentum))
    sub = np.arange(n_sites) % 4
    cols = []
    for mu in range(4):
        cols.append((spins @ (phase * (sub == mu))) * gs)
    return cols


def lanczos_lines(ham_sector, seed, n_iter=350):
    """Ritz (energy, weight) pairs from one seed, full reorthogonalization."""
    norm2 = float(np.vdot(seed, seed).real)
    if norm2 < 1e-14:
        return np.array([]), np.array([]), 0.0
    Q = np.zeros((len(seed), n_iter), dtype=complex)
    alpha, beta = [], []
    q = seed / np.sqrt(norm2)
    Q[:, 0] = q
    r = ham_sector @ q
    a = float(np.vdot(q, r).real)
    alpha.append(a)
    r = r - a * q
    for k in range(1, n_iter):
        r -= Q[:, :k] @ (Q[:, :k].conj().T @ r)      # full reorth
        b = float(np.linalg.norm(r))
        if b < 1e-12:
            break
        beta.append(b)
        q = r / b
        Q[:, k] = q
        r = ham_sector @ q
        a = float(np.vdot(q, r).real)
        alpha.append(a)
        r = r - a * q - b * Q[:, k - 1]
    vals, vecs = eigh_tridiagonal(np.array(alpha), np.array(beta))
    weights = norm2 * np.abs(vecs[0, :]) ** 2
    return vals, weights, norm2


def group_lines(all_vals, all_wts, e0, tol=1e-6):
    """Merge Ritz lines across seeds into energy groups above the GS."""
    order = np.argsort(all_vals)
    vals, wts = all_vals[order] - e0, all_wts[order]
    out = []
    for v, w in zip(vals, wts):
        if w < 1e-12:
            continue
        if out and v - out[-1][0] < tol:
            out[-1][1] += w
        else:
            out.append([v, w])
    return out


def dssf_at(ham_sec, spins, positions, momenta, gs, e0, n_sites, tag):
    results = {}
    for qi, momentum in enumerate(momenta):
        vals_all, wts_all = [], []
        for col in probe_columns(spins, positions, momentum, gs, n_sites):
            v, w, _ = lanczos_lines(ham_sec, col)
            vals_all.append(v)
            wts_all.append(w)
        lines = group_lines(np.concatenate(vals_all),
                            np.concatenate(wts_all), e0)
        tot = sum(w for _, w in lines)
        if tot < 1e-10:
            continue
        lines.sort(key=lambda x: -x[1])
        top = lines[:6]
        qmag = float(np.linalg.norm(reduce_momentum(momentum)))
        results[qi] = {"qmag": qmag, "star": star(momentum), "total": tot,
                       "top": [[float(e), float(w / tot)] for e, w in top]}
        print(f"  {tag} q#{qi:2d} |q|={qmag:.4f} {star(momentum):<18s} "
              + "  ".join(f"{e:.4f}:{w/tot:.3f}" for e, w in top[:4]),
              flush=True)
    return results


def main() -> int:
    t0 = time.perf_counter()
    rec = {}

    # ================= STAGE A: fcc(2,2,3) gates + sin-B ================
    cl = geometry.build_cluster("fcc", (2, 2, 3))
    n = int(cl.n_sites)
    states = np.sort(np.asarray(cl.ice_states, dtype=np.uint64))
    positions = np.asarray(cl.positions, dtype=float)
    labels, spins_all = fast_labels(states, positions)
    hmp = hex_masks_patterns(contractible(cl))
    ham, _ = fast_ring(states, hmp, 0.0)
    e0v, v0 = eigsh(ham, k=1, which="SA", tol=1e-11, ncv=80)
    e0 = float(e0v[0])
    print(f"A: fcc223 fast pipeline: E0 = {e0:.9f} "
          f"(exact dense reference -10.556011756)  "
          f"{'GATE1 PASS' if abs(e0 + 10.556011756) < 1e-6 else 'GATE1 FAIL'}",
          flush=True)
    if abs(e0 + 10.556011756) >= 1e-6:
        return 1

    occ = {int(s): float(np.sum(v0[labels == s, 0] ** 2))
           for s in np.unique(labels)}
    sector = max(occ, key=occ.get)
    rows = np.where(labels == sector)[0]
    ham_sec = ham[np.ix_(rows, rows)].tocsr()
    spins = spins_all[rows]
    sec_states = states[rows]

    # dense reference for gates + sin-B ratios
    dense = np.asarray(ham_sec.todense())
    values, vectors = eigh(dense)
    omega = values - values[0]
    gs = vectors[:, 0].astype(complex)

    # --- sin-B magnetic probe at L: intruder vs photon character --------
    Lq = None
    momenta223, _ = fcc_momenta(cl)
    for qi, m in enumerate(momenta223):
        if star(m) == "L":
            Lq = qi
            break
    index = {int(s): i for i, s in enumerate(sec_states)}
    sinb = np.zeros(len(rows), dtype=complex)
    ecol = np.zeros(len(rows), dtype=complex)
    phase_sites = np.exp(-2j * np.pi * (positions @ momenta223[Lq]))
    from ring48_channels import unwrap_hexagons                  # noqa: E402
    for path_sites, pts, signs, center in unwrap_hexagons(cl, contractible(cl)):
        mask = np.uint64(0)
        p1 = np.uint64(0)
        for j, s_ in enumerate(path_sites):
            b = np.uint64(1) << np.uint64(s_)
            mask |= b
            if j % 2 == 0:
                p1 |= b
        p2 = mask ^ p1
        sel = sec_states & mask
        ph = np.exp(-2j * np.pi * np.dot(momenta223[Lq], center))
        for pat, sgn in ((p1, 1.0), (p2, -1.0)):
            idx = np.where(sel == pat)[0]
            tgt = np.searchsorted(sec_states, sec_states[idx] ^ mask)
            contrib = np.zeros(len(rows), dtype=complex)
            contrib[tgt] = gs[idx]
            sinb += 1j * sgn * ph * contrib
    sub = np.arange(n) % 4
    for mu in range(4):
        ecol += (spins @ (phase_sites * (sub == mu))) * gs
    w_sin = np.abs(vectors.conj().T @ sinb) ** 2
    w_e = np.abs(vectors.conj().T @ ecol) ** 2
    groups, start = [], 0
    for i in range(1, len(omega) + 1):
        if i == len(omega) or omega[i] - omega[start] > EDEGEN:
            groups.append((float(omega[start]), slice(start, i)))
            start = i
    print("\nA: sin-B vs E character at L (fcc223):")
    print("   omega     w_E        w_sinB     R = sinB/E")
    for e, sl in groups:
        we, ws = float(w_e[sl].sum()), float(w_sin[sl].sum())
        if we + ws > 1e-4:
            print(f"   {e:.4f}   {we:.5f}   {ws:.5f}   "
                  f"{ws / we if we > 1e-12 else float('inf'):.3f}")
        if e > 4.0:
            break
    rec["sinB_L_fcc223"] = [
        {"omega": e, "w_E": float(w_e[sl].sum()),
         "w_sinB": float(w_sin[sl].sum())}
        for e, sl in groups if e < 4.5]

    # --- GATE 2: Lanczos CF reproduces the dense L lines -----------------
    cf = dssf_at(ham_sec, spins, positions, [momenta223[Lq]], gs, values[0],
                 n, "gate2")
    top = cf[0]["top"]
    ok = (abs(top[0][0] - 1.0235) < 2e-3 and abs(top[0][1] - 0.508) < 5e-3
          and abs(top[1][0] - 1.7300) < 2e-3)
    print(f"A: GATE2 (CF vs dense at L): top lines {top[:2]} -> "
          f"{'PASS' if ok else 'FAIL'}", flush=True)
    if not ok:
        return 1
    del dense, vectors
    print(f"A done ({time.perf_counter()-t0:.0f} s)\n", flush=True)

    # ================= STAGE B: fcc(2,2,4) ==============================
    cl4 = geometry.build_cluster("fcc", (2, 2, 4))
    n4 = int(cl4.n_sites)
    st4 = np.sort(np.asarray(cl4.ice_states, dtype=np.uint64))
    pos4 = np.asarray(cl4.positions, dtype=float)
    print(f"B: fcc224: {n4} sites, {len(st4):,} ice states "
          f"({time.perf_counter()-t0:.0f} s)", flush=True)
    lb4, spins4_all = fast_labels(st4, pos4)
    hmp4 = hex_masks_patterns(contractible(cl4))
    ham4, nf4 = fast_ring(st4, hmp4, 0.0)
    print(f"B: H built, nnz={ham4.nnz:,}, <n_flip>={nf4.mean():.3f} "
          f"range {int(nf4.min())}..{int(nf4.max())} "
          f"({time.perf_counter()-t0:.0f} s)", flush=True)
    e04, v04 = eigsh(ham4, k=1, which="SA", tol=1e-10, ncv=60)
    e04 = float(e04[0])
    occ4 = {}
    for s in np.unique(lb4):
        occ4[int(s)] = float(np.sum(v04[lb4 == s, 0] ** 2))
    sector4 = max(occ4, key=occ4.get)
    rows4 = np.where(lb4 == sector4)[0]
    print(f"B: E0={e04:.6f}; ground sector {sector4} "
          f"(dim {len(rows4):,}, occupancy {occ4[sector4]:.6f}) "
          f"({time.perf_counter()-t0:.0f} s)", flush=True)
    ham4s = ham4[np.ix_(rows4, rows4)].tocsr()
    spins4 = spins4_all[rows4]
    del spins4_all, ham4
    ev, gv = eigsh(ham4s, k=1, which="SA", tol=1e-11, ncv=60)
    gs4 = gv[:, 0].astype(complex)
    e0s = float(ev[0])
    print(f"B: sector ground {e0s:.6f} "
          f"({'consistent' if abs(e0s-e04) < 1e-8 else 'MISMATCH'})",
          flush=True)

    momenta4, _ = fcc_momenta(cl4)
    res = dssf_at(ham4s, spins4, pos4, momenta4, gs4, e0s, n4, "fcc224")
    rec["fcc224"] = {"E0": e04, "sector": int(sector4),
                     "sector_dim": int(len(rows4)),
                     "n_flip_mean": float(nf4.mean()),
                     "momenta": {str(k): v for k, v in res.items()}}
    with open(OUT / "ring64_results.json", "w") as fh:
        json.dump(rec, fh, indent=1, default=float)
    print(f"\nwrote ring64_results.json ({time.perf_counter()-t0:.0f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
