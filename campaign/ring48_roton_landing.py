"""Land the roton: operator content, analytic f-sum rule, Krylov improvement.

Questions answered here, in order:

(1) OPERATOR.  In Feynman-Bijl theory the collective mode is created by the
    density operator itself.  Here that operator is S^z(q) = E(q), the
    emergent electric field.  The single-mode (SMA) state is
        |SMA(q)> = S^z(q)|0> / ||S^z(q)|0>||
    and the exact overlap |<n|SMA>|^2 IS the normalized spectral weight of
    level n.  So the "share" column already measured is the SMA overlap.

(2) ANALYTIC f-SUM RULE.  For H = -K sum_p (W_p + W_p^dag),
        [S^z_mu(q), W_p]      = +g^mu_p(q) W_p
        [S^z_mu(q), W_p^dag]  = -g^mu_p(q) W_p^dag
    with g^mu_p(q) = sum_{i in p, i in mu} e^{-i q.r_i} delta_i, delta = +-1
    alternating around the hexagon.  Hence
        f_1(q) = (w K / 2N) sum_mu sum_p |g^mu_p(q)|^2 ,
        w = <W_p + W_p^dag> = -E_0/(K N_p).
    This is PURE LATTICE GEOMETRY times one ground-state number: it can be
    evaluated at ANY q, not just cluster momenta.  Gate: compare with the
    exact spectral f_1 = sum_n omega_n |<n|S^z(q)|0>|^2 / N.

(3) IS THE ICE-MANIFOLD S(q) THE CLASSICAL ONE?  Compare the quantum S(q)
    with the equal-amplitude (classical ice) S(q) on the same sector.  If
    they agree, S(q) over the full BZ may be taken from classical sampling,
    which together with (2) gives omega_SMA(q) everywhere.

(4) FEYNMAN -> FEYNMAN-COHEN.  Lanczos started FROM the SMA state: the
    k-th Ritz value is the best variational energy in the Krylov space
    span{H^j S^z(q)|0>}, j<k.  k=1 is exactly the SMA (omega_SMA); higher k
    is the systematic analogue of backflow.  How fast it converges to the
    exact line measures how collective the mode is.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.linalg import eigh, eigh_tridiagonal

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "notes"))
sys.path.insert(0, str(ROOT / "campaign"))

import recompute_finite_size_artifact as geometry               # noqa: E402
from run_momentum_resolved_spectrum import star                 # noqa: E402
from run_ring48_dssf import contractible                        # noqa: E402
from ring48_momenta_fix import fcc_momenta                      # noqa: E402
from ring64_pipeline import (fast_labels, fast_ring,            # noqa: E402
                             hex_masks_patterns, probe_columns)

OUT = ROOT / "campaign" / "outputs" / "ring_model_dssf"
NKRYLOV = 24


def hex_signed(hexes, n_sites):
    """delta_i = +-1 alternating around each hexagon -> (sites, signs)."""
    out = []
    for path in hexes:
        sites = np.asarray(path, dtype=int)
        signs = np.where(np.arange(len(sites)) % 2 == 0, 1.0, -1.0)
        out.append((sites, signs))
    return out


def f1_analytic(qs, hexsign, positions, wp, n_sites):
    """f_1(q) = (1/2N) sum_p w_p sum_mu |g^mu_p(q)|^2, K = 1.

    w_p = <W_p + W_p^dag> per plaquette.  On an anisotropic cluster the
    hexagon orientations are inequivalent, so w_p is NOT uniform and the
    per-plaquette value must be used.
    """
    sub = np.arange(n_sites) % 4
    vals = []
    for q in qs:
        phase = np.exp(-2j * np.pi * (positions @ q))
        tot = 0.0
        for (sites, signs), w in zip(hexsign, wp):
            z = signs * phase[sites]
            for mu in range(4):
                sel = sub[sites] == mu
                if sel.any():
                    tot += w * abs(z[sel].sum()) ** 2
        vals.append(tot / (2.0 * n_sites))
    return np.array(vals)


def plaquette_w(sec_states, hmp, gs):
    """<W_p + W_p^dag> for each plaquette, on the sector ground state."""
    out = []
    for mask, p1, p2 in hmp:
        sel = sec_states & mask
        flip = (sel == p1) | (sel == p2)
        idx = np.where(flip)[0]
        if len(idx) == 0:
            out.append(0.0)
            continue
        tgt = sec_states[idx] ^ mask
        pos = np.searchsorted(sec_states, tgt)
        ok = (pos < len(sec_states)) & (sec_states[np.minimum(
            pos, len(sec_states) - 1)] == tgt)
        out.append(float(np.sum(gs[idx[ok]].conj() * gs[pos[ok]]).real))
    return np.array(out)


def block_krylov_ritz(ham, seeds, e0, n_iter):
    """Lowest Ritz value in the block Krylov space span{H^j S^z_mu(q)|0>}.

    k=1 is the full four-channel single-mode subspace (a variational bound
    at or below omega_SMA); higher k is the systematic analogue of the
    Feynman-Cohen backflow correction.
    """
    V = np.column_stack(seeds)
    q, r = np.linalg.qr(V)
    keep = np.abs(np.diag(r)) > 1e-10
    if not keep.any():
        return []
    basis = q[:, keep]
    cur = basis
    ritz = []
    for _ in range(n_iter):
        Hb = basis.conj().T @ (ham @ basis)
        ev = np.linalg.eigvalsh((Hb + Hb.conj().T) / 2.0)
        ritz.append(float(ev[0]) - e0)
        cur = ham @ cur
        for _ in range(2):
            cur = cur - basis @ (basis.conj().T @ cur)
        q, r = np.linalg.qr(cur)
        keep = np.abs(np.diag(r)) > 1e-9
        if not keep.any():
            break
        cur = q[:, keep]
        basis = np.column_stack([basis, cur])
    return ritz


def krylov_ritz(ham, seed, e0, n_iter):
    """Lowest Ritz value vs Krylov dimension, seeded by the probe state."""
    nrm = float(np.linalg.norm(seed))
    if nrm < 1e-14:
        return []
    Q = np.zeros((len(seed), n_iter), dtype=complex)
    q = seed / nrm
    Q[:, 0] = q
    alpha, beta, ritz = [], [], []
    r = ham @ q
    a = float(np.vdot(q, r).real)
    alpha.append(a)
    ritz.append(a - e0)
    r = r - a * q
    for k in range(1, n_iter):
        r -= Q[:, :k] @ (Q[:, :k].conj().T @ r)
        b = float(np.linalg.norm(r))
        if b < 1e-11:
            break
        beta.append(b)
        q = r / b
        Q[:, k] = q
        r = ham @ q
        a = float(np.vdot(q, r).real)
        alpha.append(a)
        r = r - a * q - b * Q[:, k - 1]
        ev = eigh_tridiagonal(np.array(alpha), np.array(beta),
                              eigvals_only=True)
        ritz.append(float(ev[0]) - e0)
    return ritz


def main() -> int:
    t0 = time.perf_counter()
    rec = {}

    cl = geometry.build_cluster("fcc", (2, 2, 3))
    n = int(cl.n_sites)
    states = np.sort(np.asarray(cl.ice_states, dtype=np.uint64))
    positions = np.asarray(cl.positions, dtype=float)
    labels, spins_all = fast_labels(states, positions)
    hexes = contractible(cl)
    hmp = hex_masks_patterns(hexes)
    hexsign = hex_signed(hexes, n)
    momenta, _ = fcc_momenta(cl)

    ham, _ = fast_ring(states, hmp, 0.0)
    # ground sector
    from scipy.sparse.linalg import eigsh
    vals0, vecs0 = eigsh(ham, k=1, which="SA", tol=1e-11, ncv=60)
    per = {int(s): float(np.sum(vecs0[labels == s, 0] ** 2))
           for s in np.unique(labels)}
    sector = max(per, key=per.get)
    rows = np.where(labels == sector)[0]
    hs = ham[np.ix_(rows, rows)]
    values, vectors = eigh(np.asarray(hs.todense()))
    e0 = float(values[0])
    gs = vectors[:, 0].astype(complex)
    omega = values - e0
    print(f"sector {sector} dim {len(rows)}  E0 = {e0:.6f} K", flush=True)

    # <W_p + W_p^dag>: per plaquette (NOT uniform on an anisotropic box)
    npl = len(hexes)
    sec_states = states[rows]
    wp = plaquette_w(sec_states, hmp, gs)
    print(f"<W_p+W_p^dag>: mean {wp.mean():.6f}  min {wp.min():.6f}  "
          f"max {wp.max():.6f}  spread {(wp.max()-wp.min()):.2e}")
    print(f"   GATE0  sum_p w_p = {wp.sum():.8f}   -E0 = {-e0:.8f}   "
          f"ratio {wp.sum()/(-e0):.8f}")
    rec["w_plaquette_mean"] = float(wp.mean())
    rec["w_plaquette_min"] = float(wp.min())
    rec["w_plaquette_max"] = float(wp.max())
    rec["gate0_sum_over_E0"] = float(wp.sum() / (-e0))

    # ---------------- per-momentum analysis ----------------
    print("\nq                     S(q)     f1(exact)  f1(analytic)  ratio"
          "    om_SMA   om_exact  ratio   SMAshare")
    seen, table = {}, []
    hs_dense = np.asarray(hs.todense())
    for qi, q in enumerate(momenta):
        qr = np.asarray(q) % 1.0
        qr = np.minimum(qr, 1.0 - qr)
        key = tuple(np.round(np.sort(qr), 4))
        if key in seen:
            continue
        cols = probe_columns(spins_all[rows], positions, q, gs, n)
        wq = np.zeros(len(rows))
        seeds = []
        for col in cols:
            wq += np.abs(vectors.conj().T @ col) ** 2 / n
            seeds.append(col)
        S = float(wq.sum())
        if S < 1e-10:
            continue
        seen[key] = True
        f1_ex = float(np.dot(wq, omega))
        f1_an = float(f1_analytic([np.asarray(q)], hexsign, positions,
                                  wp, n)[0])
        om_sma = f1_ex / S
        jl = int(np.argmax(wq))
        om_ex = float(omega[jl])
        lab = star(q)
        print(f"{lab:<20s} {S:.5f}  {f1_ex:.5f}   {f1_an:.5f}    "
              f"{f1_an/f1_ex:.4f}  {om_sma:.4f}  {om_ex:.4f}  "
              f"{om_ex/om_sma:.4f}  {wq[jl]/S:.3f}", flush=True)
        table.append({"star": lab, "q": [float(x) for x in q], "S": S,
                      "f1_exact": f1_ex, "f1_analytic": f1_an,
                      "omega_sma": om_sma, "omega_exact": om_ex,
                      "ratio_exact_sma": om_ex / om_sma,
                      "sma_share": float(wq[jl] / S)})
        # Krylov improvement from the SMA state
        ritz = block_krylov_ritz(hs_dense, seeds, e0, NKRYLOV)
        table[-1]["krylov"] = [float(x) for x in ritz]
        show = "  ".join(f"{ritz[k]:.3f}" for k in
                         [0, 1, 2, 3, 5, 8, min(15, len(ritz) - 1)]
                         if 0 <= k < len(ritz))
        print(f"    Krylov(k=1,2,3,4,6,9,16): {show}", flush=True)

        # classical (equal-amplitude ice) S(q) on the same sector
        flat = np.ones(len(rows)) / np.sqrt(len(rows))
        ccols = probe_columns(spins_all[rows], positions, q,
                              flat.astype(complex), n)
        Sc = float(sum(float(np.vdot(c, c).real) for c in ccols) / n)
        table[-1]["S_classical"] = Sc
        print(f"    S quantum {S:.5f}   S classical(ice) {Sc:.5f}   "
              f"ratio {Sc/S:.4f}", flush=True)
    rec["momenta"] = table

    with open(OUT / "ring48_roton_landing.json", "w") as fh:
        json.dump(rec, fh, indent=1, default=float)
    print(f"\nwrote ring48_roton_landing.json "
          f"({time.perf_counter()-t0:.0f} s)")
    print("GATE A: f1_analytic/f1_exact must be 1.0000 at every q.")
    print("GATE B: S_classical/S_quantum near 1 licenses classical-ice S(q)")
    print("        over the full BZ; combined with GATE A this gives")
    print("        omega_SMA(q) everywhere.")
    return 0



if __name__ == "__main__":
    raise SystemExit(main())
