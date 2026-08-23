"""The backflow-like decomposition on clusters beyond 48 sites.

The 48-site result -- 71% of the variational relaxation of the soft L
mode carried by the hexagon family that is geometrically dark to the
probe -- was established by complete diagonalization.  If the
decomposition is part of the mechanism, its size dependence must be
tested.  This repeats it on the 64- and 72-site clusters:

  * ground state and the soft L eigenstate from Lanczos (eigsh);
  * the bare single-mode state = lowest eigenvector of H in the
    4-dimensional span of the sublattice-resolved probes
    S^z_mu(L)|0>;
  * per-hexagon-family <W_p + W_p^dag> in all three states; the
    energy decomposition E_n = -K sum_p [<W>_n - <W>_0] is exact with
    no remainder, so the per-family relief and its total are gated by
    the states' energies.

Families are labeled by the omitted sublattice (wrapping-immune).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.sparse.linalg import eigsh

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "notes"))
sys.path.insert(0, str(ROOT / "campaign"))

import recompute_finite_size_artifact as geometry               # noqa: E402
from run_momentum_resolved_spectrum import star                 # noqa: E402
from run_ring48_dssf import contractible                        # noqa: E402
from ring48_momenta_fix import fcc_momenta                      # noqa: E402
from ice_enumerate_fast import enumerate_ice_branched, bit_of   # noqa: E402
from ring96_pipeline import (Table128, winding_keys, hex_words,  # noqa: E402
                             ring_matrix)

OUT = ROOT / "campaign" / "outputs" / "ring_model_dssf"
U64 = np.uint64


def log(m, t0):
    print(f"[{time.perf_counter()-t0:7.1f}s] {m}", flush=True)


def family_of(hexes, n):
    sub = np.arange(n) % 4
    fam = []
    for path in hexes:
        present = set(sub[np.asarray(path, dtype=int)])
        fam.append([mu for mu in range(4) if mu not in present][0])
    return np.array(fam)


def wp_per_hex(tab, hw, v):
    """<W_p + W_p^dag> per hexagon in state v (real)."""
    W = tab.w
    out = []
    for mask, p1, p2 in hw:
        s0 = W[:, 0] & mask[0]
        s1 = W[:, 1] & mask[1]
        hit = np.where(((s0 == p1[0]) & (s1 == p1[1]))
                       | ((s0 == p2[0]) & (s1 == p2[1])))[0]
        tgt = np.empty((len(hit), 2), dtype=U64)
        tgt[:, 0] = W[hit, 0] ^ mask[0]
        tgt[:, 1] = W[hit, 1] ^ mask[1]
        pos = tab.index_of(tgt)
        ok = pos >= 0
        out.append(float(np.sum(v[hit[ok]] * v[pos[ok]])))
    return np.array(out)


def analyse(shape, t0):
    orig = geometry.enumerate_ice_states
    geometry.enumerate_ice_states = lambda a, b: np.array([0], dtype=U64)
    cl = geometry.build_cluster("fcc", shape)
    geometry.enumerate_ice_states = orig
    n = int(cl.n_sites)
    positions = np.asarray(cl.positions, dtype=float)
    tets = [tuple(int(s) for s in t) for t in cl.tets]
    hexes = contractible(cl)
    fam = family_of(hexes, n)
    states = enumerate_ice_branched(n, tets, log=lambda m: None)
    ipos = np.rint(positions * 4).astype(np.int64)
    keys = winding_keys(states, ipos)
    uk, cnt = np.unique(keys, return_counts=True)
    hw = hex_words(hexes)
    best = None
    for c in uk[np.argsort(cnt)[::-1][:3]]:
        rows = np.where(keys == c)[0]
        tab = Table128(states[rows])
        H = ring_matrix(tab, hw, t0)
        e = float(eigsh(H, k=1, which="SA", tol=1e-10, ncv=30,
                        return_eigenvectors=False)[0])
        if best is None or e < best[0] - 1e-9:
            best = (e, tab, H)
    _, tab, H = best
    del states, keys
    log(f"fcc{shape}: sector dim {len(tab):,}", t0)

    vals, vecs = eigsh(H, k=6, which="SA", tol=1e-12, ncv=48)
    order = np.argsort(vals)
    vals, vecs = vals[order], vecs[:, order]
    e0 = float(vals[0])
    gs = np.ascontiguousarray(vecs[:, 0])

    # the rotonizing L momentum = the one with the largest S
    subl = np.arange(n) % 4
    Ls = [q for q in fcc_momenta(cl)[0] if star(q) == "L"]
    cols_by_q = {}
    for qi, q in enumerate(Ls):
        phase = np.exp(-2j * np.pi * (positions @ np.asarray(q)))
        cols = []
        for mu in range(4):
            sites = np.where(subl == mu)[0]
            acc = np.zeros(len(tab.w), dtype=complex)
            for s in sites:
                acc += (bit_of(tab.w, int(s)) - 0.5) * phase[s]
            cols.append(acc * gs)
        S = sum(float(np.vdot(c, c).real) for c in cols) / n
        cols_by_q[qi] = (q, S, cols)
        log(f"  S(L) at {np.round(q,3)} = {S:.5f}", t0)
    qi = max(cols_by_q, key=lambda k: cols_by_q[k][1])
    q, S, cols = cols_by_q[qi]

    # --- soft eigenstate: the PROBE-ALIGNED member of the multiplet ---
    # The soft level is degenerate (two L momenta at 48, more at 64), so
    # an arbitrary eigsh vector is basis-dependent and its per-family
    # coherences are meaningless.  The physical state is the projection
    # of the probe onto the multiplet: rho = P_m (sum_mu c_mu c_mu^+) P_m,
    # normalized -- the state S^z(q)|0> actually creates at this q.
    # Diagnostics printed: probe-gs overlaps (must vanish), multiplet
    # capture of the probe, and the projected-state energy (must equal
    # the soft eigenvalue).
    for mu in range(4):
        ov = abs(np.vdot(gs, cols[mu])) / np.linalg.norm(cols[mu])
        log(f"    |<0|probe_{mu}>| / |probe_{mu}| = {ov:.2e}", t0)
    wts = []
    for j in range(1, len(vals)):
        v = vecs[:, j]
        w = sum(abs(np.vdot(v, c)) ** 2 for c in cols) / n
        wts.append((w, j))
    wts.sort(reverse=True)
    j_soft = wts[0][1]
    mult = [j for j in range(1, len(vals))
            if abs(vals[j] - vals[j_soft]) < 1e-7]
    log(f"  soft multiplet: levels {mult}, omega = "
        f"{vals[j_soft]-e0:.4f} K", t0)
    # rho = sum_mu |P c_mu|^2-weighted pure states; equivalent single
    # decomposition: w_p(rho) = sum_mu w_p(P c_mu) / sum_mu |P c_mu|^2
    proj = []
    for mu in range(4):
        a = np.zeros(len(gs), dtype=complex)
        for j in mult:
            a += vecs[:, j] * np.vdot(vecs[:, j], cols[mu])
        proj.append(a)
    cap = sum(float(np.vdot(a, a).real) for a in proj) \
        / sum(float(np.vdot(c, c).real) for c in cols)
    log(f"  multiplet captures {cap:.4f} of the probe norm "
        f"(= this level's share of S at this q)", t0)

    def famsum_complex(v):
        return (np.array([wp_per_hex(tab, hw, np.ascontiguousarray(
            v.real))[fam == mu].sum() for mu in range(4)])
            + np.array([wp_per_hex(tab, hw, np.ascontiguousarray(
                v.imag))[fam == mu].sum() for mu in range(4)]))

    nrm2 = sum(float(np.vdot(a, a).real) for a in proj)
    w_soft = sum(famsum_complex(a) for a in proj) / nrm2
    # energy gate on the projected state
    e_proj = sum(float(np.vdot(a, H @ a).real) for a in proj) / nrm2
    log(f"  projected-state energy = {e_proj - e0:.6f} K "
        f"(soft eigenvalue {vals[j_soft]-e0:.6f})", t0)

    # bare single-mode state: lowest eigenvector of H in span(cols),
    # with the ground state PROJECTED OUT first.  On fcc(2,3,3) this is
    # not optional: probe channel 3 at L is exactly proportional to the
    # ground state (the weighted sublattice-3 sum is a conserved
    # quantity of the ring dynamics on that torus -- family-3 hexagons
    # contain no sublattice-3 site, and the two sublattice-3 sites of
    # every other hexagon are L-commensurate), so the raw span contains
    # |0> and its lowest eigenvalue is E0.  The elastic share this
    # invariant contributes to S(q) is printed.
    elastic = sum(abs(np.vdot(gs, c)) ** 2 for c in cols) / n
    log(f"    elastic (ground-state) share of S at this q: "
        f"{elastic / S:.4f}", t0)
    B = np.stack([c - gs * np.vdot(gs, c) for c in cols], axis=1)
    keep = [k for k in range(4)
            if np.linalg.norm(B[:, k]) > 1e-8 * np.linalg.norm(cols[k])]
    B = B[:, keep]
    Q, R = np.linalg.qr(B)
    good = np.abs(np.diag(R)) > 1e-8
    Q = Q[:, good]
    log(f"    span dimension after gs projection: {Q.shape[1]}", t0)
    log(f"    max |<0|Q_k>| = "
        f"{max(abs(np.vdot(gs, Q[:, k])) for k in range(Q.shape[1])):.2e}",
        t0)
    Hq = Q.conj().T @ (H @ Q)
    ev, evec = np.linalg.eigh(Hq)
    log(f"    span spectrum - E0 = {np.round(ev - e0, 4)}", t0)
    sma = (Q @ evec[:, 0])
    e_sma = float(ev[0]) - e0
    log(f"  single-mode energy = {e_sma:.4f} K "
        f"(exact soft level {vals[j_soft]-e0:.4f})", t0)
    # robustness: the SMA-aligned member of the multiplet, for
    # comparison with the probe-aligned one used above
    b = np.zeros(len(gs), dtype=complex)
    for j in mult:
        b += vecs[:, j] * np.vdot(vecs[:, j], sma)
    nb = float(np.vdot(b, b).real)
    log(f"    multiplet capture of the SMA state: {nb:.4f}", t0)
    w_soft_sma = famsum_complex(b) / nb
    sma_r = np.ascontiguousarray(sma.real)
    sma_i = np.ascontiguousarray(sma.imag)

    def famsum(v):
        w = wp_per_hex(tab, hw, v)
        return np.array([w[fam == mu].sum() for mu in range(4)])

    w_gs = famsum(gs)
    w_sma = famsum(sma_r) + famsum(sma_i)
    gate = w_gs.sum() / (-e0)
    log(f"  GATE sum_p w_p / (-E0/K) = {gate:.8f}", t0)

    d_sma = -(w_sma - w_gs)     # energy contribution per family, K=1
    d_soft = -(w_soft - w_gs)
    relief = d_sma - d_soft
    log("  family   E_sma     E_exact   relief", t0)
    for mu in range(4):
        log(f"    {mu}    {d_sma[mu]:+8.4f}  {d_soft[mu]:+8.4f}  "
            f"{relief[mu]:+8.4f}", t0)
    log(f"  totals  {d_sma.sum():+8.4f}  {d_soft.sum():+8.4f}  "
        f"{relief.sum():+8.4f}   (gate vs energies: "
        f"{d_sma.sum():.4f} ?= {e_sma:.4f}, "
        f"{d_soft.sum():.4f} ?= {vals[j_soft]-e0:.4f})", t0)
    d_soft2 = -(w_soft_sma - w_gs)
    relief2 = d_sma - d_soft2
    log(f"  SMA-aligned variant relief: "
        f"{np.round(relief2, 4)}   (probe-aligned above: "
        f"{np.round(relief, 4)})", t0)
    # which family is dark to the probe at this L?
    hsg = [(np.asarray(p, dtype=int),
            np.where(np.arange(6) % 2 == 0, 1.0, -1.0)) for p in hexes]
    phase = np.exp(-2j * np.pi * (positions @ np.asarray(q)))
    g2 = []
    for sites, signs in hsg:
        z = signs * phase[sites]
        tot = 0.0
        for mu in range(4):
            sel = subl[sites] == mu
            if sel.any():
                tot += abs(z[sel].sum()) ** 2
        g2.append(tot)
    g2 = np.array(g2)
    gfam = np.array([g2[fam == mu].sum() for mu in range(4)])
    dark = int(np.argmin(gfam))
    log(f"  geometric factor by family: {np.round(gfam,1)}  "
        f"-> dark family {dark}", t0)
    log(f"  dark-family share of relief: "
        f"{relief[dark]/relief.sum():.3f}", t0)
    return {"shape": list(shape), "e_sma": e_sma,
            "e_soft": float(vals[j_soft] - e0),
            "E_sma_fam": [float(x) for x in d_sma],
            "E_soft_fam": [float(x) for x in d_soft],
            "relief_fam": [float(x) for x in relief],
            "dark_family": dark,
            "dark_share": float(relief[dark] / relief.sum()),
            "relief_sma_aligned": [float(x) for x in relief2],
            "elastic_share": float(elastic / S),
            "gate": float(gate)}


def main() -> int:
    t0 = time.perf_counter()
    out = []
    for shape in [(2, 2, 3), (2, 2, 4), (2, 3, 3)]:
        print(f"\n######## fcc{shape}", flush=True)
        out.append(analyse(shape, t0))
    with open(OUT / "backflow_sizes.json", "w") as fh:
        json.dump(out, fh, indent=1)
    log("wrote backflow_sizes.json", t0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
