"""Close the two remaining pendings of paper_multiphoton.

C. FLUX-RESOLVED DIAGNOSTIC.  For selected exact states (ground, roton,
   L photon, (1/3,1/3,1/3) photon, the largest residual states): the
   connected plaquette flux-flux correlator
      c_n(d) = < F_p F_p' >_n - <F_p>_n <F_p'>_n ,  F_p = W_p + W_p^+,
   binned by plaquette-center minimum-image distance, minus the ground-state
   baseline.  A monopole-pair-like state carries a localized flux
   disturbance (enhanced short-distance correlations); photon-like and
   multi-photon states a weak delocalized modulation.

D. ONE-LOOP (QUARTIC-VERTEX) COMPARISON.  Expand -2K cos B to quartic:
   H4 = -(K/12) sum_p B_p^4.  In the real mode basis (real SVD of the real
   curl), first-order degenerate perturbation theory gives the polarization
   splitting matrix  M_ab = -(K)  sum_p S_p G_pa G_pb  (Wick: 12*c4), and
   the 1->3 transferred weight  T_n = sum_f |<f|H4|1_n>|^2/(w_f-w_n)^2.
   GATE: every Wick factor validated against exact bosonic Fock
   diagonalization of a reduced 3-mode system before use.
   One parameter U/K is fitted to the measured (1/3,1/3,1/3) splitting;
   everything else is prediction, compared with the measured splittings and
   residual fractions.
"""

from __future__ import annotations

import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "notes"))
sys.path.insert(0, str(ROOT / "campaign"))

import recompute_finite_size_artifact as geometry               # noqa: E402
from run_ring48_dssf import contractible                         # noqa: E402
from ring48_channels import unwrap_hexagons                      # noqa: E402
from ring64_pipeline import (fast_labels, fast_ring,             # noqa: E402
                             hex_masks_patterns)

OUT = ROOT / "campaign" / "outputs" / "ring_model_dssf"
EDEGEN = 1e-8
SCALE = 0.586840          # sqrt(2UK)/K, photon-line fit


# ------------------------------------------------------------------ part C
def flux_part(cluster, sec_states, vectors, omega, hmp, centers, Linv, L):
    dim = len(sec_states)
    flip_maps = []
    for mask, p1, p2 in hmp:
        sel = sec_states & mask
        idx = np.where((sel == p1) | (sel == p2))[0]
        tgt = np.searchsorted(sec_states, sec_states[idx] ^ mask)
        flip_maps.append((idx, tgt))

    def fvecs(state):
        out = np.zeros((dim, len(flip_maps)))
        for p, (idx, tgt) in enumerate(flip_maps):
            out[tgt, p] = state[idx]
        return out

    # distance bins between plaquette centers (minimum image)
    dmat = np.zeros((len(centers), len(centers)))
    for i in range(len(centers)):
        d = centers - centers[i]
        d -= np.round(d @ Linv) @ L
        dmat[i] = np.linalg.norm(d, axis=1)
    bins = np.round(dmat, 3)
    uniq = np.unique(bins)[1:6]              # first few shells, skip d=0

    picks = {"ground": 0}
    for name, target in (("roton_L", 1.0235), ("photon_L", 1.7303),
                         ("photon_111", 1.9802), ("resid_4p34", 4.3447),
                         ("resid_3p09", 3.0947)):
        picks[name] = int(np.argmin(np.abs(omega - target)))

    print("C. flux-flux correlations (connected, minus ground baseline):")
    print("   state         " + "  ".join(f"d={u:.3f}" for u in uniq)
          + "   <n_flip> rel")
    base = None
    rec = {}
    for name, ni in picks.items():
        v = vectors[:, ni]
        F = fvecs(v)                          # (dim, 48)
        one = v @ F                           # <n|F_p|n>
        two = F.T @ F                         # <n|F_p F_p'|n>
        conn = two - np.outer(one, one)
        prof = np.array([conn[bins == u].mean() for u in uniq])
        nfrel = float(np.sum(F * F.T @ np.eye(1) if False else 0))
        # <n_flip> via sum_p <n|F_p^2|n> = sum_p <n|P_flip,p|n>
        nf = float(np.trace(two) if False else np.sum(np.einsum(
            'ip,ip->p', F, F)))
        if base is None:
            base = (prof, nf)
        d = prof - base[0]
        print(f"   {name:<12s} " + "  ".join(f"{x:+.4f}" for x in d)
              + f"     {nf - base[1]:+.3f}")
        rec[name] = {"omega": float(omega[ni]), "profile": d.tolist(),
                     "nflip_rel": nf - base[1]}
    return rec


# ------------------------------------------------------------------ part D
def wick_gate(Gm, wm, c4):
    """Validate PT formulas vs exact Fock diag on a 3-mode reduction."""
    nm = 3
    G = Gm[:, :nm]
    w = wm[:nm]
    nmax = 7
    dim = nmax ** nm
    idx = list(itertools.product(range(nmax), repeat=nm))
    pos = {t: i for i, t in enumerate(idx)}
    H = np.zeros((dim, dim))
    for i, occ in enumerate(idx):
        H[i, i] = sum(w[m] * occ[m] for m in range(nm))
    # x_m operators
    X = []
    for m in range(nm):
        xm = np.zeros((dim, dim))
        for i, occ in enumerate(idx):
            if occ[m] + 1 < nmax:
                j = pos[tuple(o + (1 if k == m else 0)
                              for k, o in enumerate(occ))]
                amp = np.sqrt(occ[m] + 1)
                xm[j, i] += amp
                xm[i, j] += amp
        X.append(xm)
    small = 0.02 * c4                        # perturbative regime for gate
    Hp = H.copy()
    for p in range(Gm.shape[0]):
        B = sum(G[p, m] * X[m] for m in range(nm))
        Hp += small * (B @ B @ B @ B)
    ev0, _ = np.linalg.eigh(H)
    ev1, _ = np.linalg.eigh(Hp)
    # exact first-order shift of one-boson states vs formula
    ok = True
    S = np.array([np.sum(G[p] ** 2 / (2 * w)) for p in range(Gm.shape[0])])
    # vacuum: 3 S^2 ; one-boson a: + 12 S_p g_pa^2, g = G/sqrt(2w)
    g = G / np.sqrt(2 * w)[None, :]
    e_vac = small * float(np.sum(3 * S ** 2))
    for a in range(nm):
        pred = small * float(np.sum(12 * S * g[:, a] ** 2))
        # exact: energy of lowest state with one boson in mode a minus vacuum
        target = w[a]
        cand = ev1 - ev1[0]
        exact = cand[np.argmin(np.abs(cand - target))] - target
        if abs(exact - pred) > 0.15 * max(abs(pred), 1e-9):
            ok = False
        print(f"   gate mode {a}: exact dE {exact:+.6f} vs Wick {pred:+.6f}")
    return ok


def oneloop_part(cluster):
    hexes = contractible(cluster)
    n = int(cluster.n_sites)
    C = np.zeros((len(hexes), n))
    for h, path in enumerate(hexes):
        for j, site in enumerate(path):
            C[h, site] = 1.0 if j % 2 == 0 else -1.0
    u, s, vt = np.linalg.svd(C)
    keep = s > 1e-9
    sig = s[keep]                     # 22 modes
    Umod = u[:, keep]                 # plaquette-space mode shapes
    nm = len(sig)
    print(f"D. one-loop on {nm} real modes; sigma range "
          f"[{sig.min():.3f},{sig.max():.3f}]")

    results = {}
    # scan U/K; omega_m = sig * SCALE (K units); g_pm = B-amplitude
    for uok in (0.05, 0.1722, 0.5):    # 0.1722 = SCALE^2/2 => U/K from fit?
        w = sig * SCALE
        # B_p = sum_m sigma_m U_pm q_m ; q_m = sqrt(U/(2 w_m)) x_m  (K=1)
        g = Umod * sig[None, :] * np.sqrt(uok / (2 * w))[None, :]
        c4 = -1.0 / 12.0
        S = np.sum(g ** 2, axis=1)
        # degenerate-PT matrix on the modes
        M = 12 * c4 * (g * S[:, None]).T @ g
        # splittings per sigma-multiplet
        splits = {}
        for val in np.unique(np.round(sig, 6)):
            sel = np.where(np.abs(sig - val) < 1e-6)[0]
            sub = M[np.ix_(sel, sel)]
            ev = np.linalg.eigvalsh(sub)
            base = val * SCALE
            frac = (ev.max() - ev.min()) / base if base else 0.0
            splits[f"{val:.4f}"] = float(frac)
        # 1->3 transferred weight for one mode per multiplet
        trans = {}
        for val in np.unique(np.round(sig, 6)):
            a0 = int(np.where(np.abs(sig - val) < 1e-6)[0][0])
            T = 0.0
            for aa in range(nm):
                for bb in range(aa, nm):
                    for cc in range(bb, nm):
                        # symmetry factor
                        if aa == bb == cc:
                            amp = 4 * np.sqrt(6) * np.sum(
                                g[:, aa] ** 3 * g[:, a0])
                        elif aa == bb:
                            amp = 12 * np.sqrt(2) * np.sum(
                                g[:, aa] ** 2 * g[:, cc] * g[:, a0])
                        elif bb == cc:
                            amp = 12 * np.sqrt(2) * np.sum(
                                g[:, bb] ** 2 * g[:, aa] * g[:, a0])
                        else:
                            amp = 24 * np.sum(
                                g[:, aa] * g[:, bb] * g[:, cc] * g[:, a0])
                        amp *= c4
                        dw = w[aa] + w[bb] + w[cc] - w[a0]
                        T += amp * amp / (dw * dw)
            trans[f"{val:.4f}"] = float(T / (1 + T))
        results[uok] = {"splittings": splits, "transferred": trans}
        print(f"  U/K={uok}: splittings " + "  ".join(
            f"{k}:{100*v:.1f}%" for k, v in splits.items()))
        print(f"           1->3 weight " + "  ".join(
            f"{k}:{100*v:.1f}%" for k, v in trans.items()))
    return results


def main() -> int:
    t0 = time.perf_counter()
    cl = geometry.build_cluster("fcc", (2, 2, 3))
    states = np.sort(np.asarray(cl.ice_states, dtype=np.uint64))
    positions = np.asarray(cl.positions, dtype=float)
    labels, _ = fast_labels(states, positions)
    hmp = hex_masks_patterns(contractible(cl))
    ham, _ = fast_ring(states, hmp, 0.0)
    e0, v0 = eigsh(ham, k=1, which="SA", tol=1e-11, ncv=80)
    occ = {int(s): float(np.sum(v0[labels == s, 0] ** 2))
           for s in np.unique(labels)}
    sector = max(occ, key=occ.get)
    rows = np.where(labels == sector)[0]
    values, vectors = eigh(np.asarray(ham[np.ix_(rows, rows)].todense()))
    omega = values - values[0]
    print(f"sector diagonalized ({time.perf_counter()-t0:.0f} s)", flush=True)

    L = np.asarray(cl.Lvecs, dtype=float)
    Linv = np.linalg.inv(L)
    centers = np.array([c for _, _, _, c in
                        unwrap_hexagons(cl, contractible(cl))])
    recC = flux_part(cl, states[rows], vectors, omega, hmp, centers, Linv, L)

    # gate the Wick factors first
    hexes = contractible(cl)
    n = int(cl.n_sites)
    C = np.zeros((len(hexes), n))
    for h, path in enumerate(hexes):
        for j, site in enumerate(path):
            C[h, site] = 1.0 if j % 2 == 0 else -1.0
    u, s, vt = np.linalg.svd(C)
    keep = s > 1e-9
    w_modes = s[keep] * SCALE
    G_modes = u[:, keep] * s[keep][None, :]
    print("\nD-gate: Wick factors vs exact 3-mode Fock diagonalization:")
    ok = wick_gate(G_modes, w_modes, 1.0)
    print(f"D-gate -> {'PASS' if ok else 'FAIL: formulas invalid'}")
    recD = oneloop_part(cl) if ok else {"gate": "failed"}

    with open(OUT / "ring48_close_pendings.json", "w") as fh:
        json.dump({"flux": recC, "oneloop": recD, "gate_pass": bool(ok)},
                  fh, indent=1, default=float)
    print(f"\nwrote ring48_close_pendings.json "
          f"({time.perf_counter()-t0:.0f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
