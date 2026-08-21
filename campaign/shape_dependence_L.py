"""Shape dependence of the L-point anomaly at fixed system size.

The referee's concern is that the L behaviour might be a property of
one particular anisotropic box rather than of the model.  The cleanest
control is to change the SHAPE of the torus while holding the number
of spins fixed, so that the Hilbert space is the same size and only
the boundary identifications differ.

Four inequivalent 48-site tori contain an L-type momentum:
(2,2,3), (1,3,4), (1,2,6) and (3,4,1).  For each we report every
quantity the anomaly is built from:

    E_0,  omega(L) of the two transverse branches,  S(L),
    the lower branch's share of the weight,  and
    d omega/dV, the Hellmann-Feynman flippability census,

together with the gates that must hold on any valid cluster
(sum_p <W_p + W_p^dag> = -E_0/K, vanishing longitudinal residue,
polarization purity).  Shapes that fail a gate are reported as failing
rather than quietly dropped.
"""

from __future__ import annotations

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
from run_momentum_resolved_spectrum import star                 # noqa: E402
from run_ring48_dssf import contractible                        # noqa: E402
from ring48_momenta_fix import fcc_momenta                      # noqa: E402
from ring48_channels import unwrap_hexagons, analytic_blocks    # noqa: E402
from ring64_pipeline import (fast_labels, fast_ring,            # noqa: E402
                             hex_masks_patterns)

OUT = ROOT / "campaign" / "outputs" / "ring_model_dssf"
EDEGEN = 1e-8
SHAPES = [(2, 2, 3), (1, 3, 4), (1, 2, 6), (3, 4, 1)]


def log(m, t0):
    print(f"[{time.perf_counter()-t0:7.1f}s] {m}", flush=True)


def nflip_diag(states, hmp):
    v = np.zeros(len(states))
    for mask, p1, p2 in hmp:
        sel = states & mask
        v += ((sel == p1) | (sel == p2)).astype(float)
    return v


def analyse(shape, t0):
    out = {"shape": list(shape)}
    cl = geometry.build_cluster("fcc", shape)
    n = int(cl.n_sites)
    states = np.sort(np.asarray(cl.ice_states, dtype=np.uint64))
    positions = np.asarray(cl.positions, dtype=float)
    hexes = contractible(cl)
    hmp = hex_masks_patterns(hexes)
    log(f"shape {shape}: {n} sites, {len(states):,} ice states, "
        f"{len(hexes)} contractible hexagons", t0)
    out.update({"n_sites": n, "n_ice": int(len(states)),
                "n_hex": len(hexes)})

    labels, spins_all = fast_labels(states, positions)
    ham, _ = fast_ring(states, hmp, 0.0)
    _, v0 = eigsh(ham, k=1, which="SA", tol=1e-11, ncv=80)
    occ = {int(s): float(np.sum(v0[labels == s, 0] ** 2))
           for s in np.unique(labels)}
    sector = max(occ, key=occ.get)
    rows = np.where(labels == sector)[0]
    values, vectors = eigh(np.asarray(ham[np.ix_(rows, rows)].todense()))
    omega = values - values[0]
    e0 = float(values[0])
    gs = vectors[:, 0].astype(complex)
    spins = spins_all[rows]
    log(f"  sector {sector} dim {len(rows)}, E0 = {e0:.6f} K", t0)
    out.update({"sector_dim": int(len(rows)), "E0": e0,
                "E0_per_site": e0 / n})

    # gate: plaquette sum rule
    sec_states = states[rows]
    idx = {int(s): i for i, s in enumerate(sec_states)}
    wsum = 0.0
    for mask, p1, p2 in hmp:
        sel = sec_states & mask
        hit = np.where((sel == p1) | (sel == p2))[0]
        tgt = sec_states[hit] ^ mask
        for a, tv in zip(hit, tgt):
            b = idx.get(int(tv))
            if b is not None:
                wsum += 2.0 * (gs[a].real * gs[b].real)
    gate = wsum / (-e0)
    out["sumrule_ratio"] = float(gate)
    log(f"  gate sum_p <W+W^dag> / (-E0/K) = {gate:.8f}", t0)

    nf = nflip_diag(sec_states, hmp)
    nf_gs = float(np.sum(np.abs(gs) ** 2 * nf))
    out["nflip_ground"] = nf_gs

    groups, start = [], 0
    for i in range(1, len(omega) + 1):
        if i == len(omega) or omega[i] - omega[start] > EDEGEN:
            groups.append((float(omega[start]), slice(start, i)))
            start = i

    block = analytic_blocks(cl, unwrap_hexagons(cl, hexes))
    momenta, _ = fcc_momenta(cl)
    sub = np.arange(n) % 4
    out["L"] = []
    for qi, q in enumerate(momenta):
        if star(q) != "L":
            continue
        phase = np.exp(-2j * np.pi * (positions @ q))
        A = np.zeros((len(rows), 4), dtype=complex)
        for mu in range(4):
            A[:, mu] = vectors.conj().T @ ((spins @ (phase * (sub == mu)))
                                           * gs)
        A /= np.sqrt(n)
        w_tot = np.sum(np.abs(A) ** 2, axis=1)
        S_q = float(w_tot.sum())
        if S_q < 1e-12:
            continue
        u, s, vh = np.linalg.svd(block(-q), full_matrices=True)
        T = (A @ vh.conj().T)[:, :2]
        longres = float(np.max(np.abs(A @ vh.conj().T)[:, 2:] ** 2)) / S_q
        bright = sorted([(float(w_tot[sl].sum()), e, sl)
                         for e, sl in groups], key=lambda r: -r[0])[:2]
        Ms, lv = [], []
        for wt, e, sl in bright:
            An = T[sl]
            M = An.conj().T @ An
            tr = float(np.trace(M).real)
            pur = float(np.trace(M @ M).real / tr ** 2) if tr > 0 else 0.0
            Ms.append(M)
            # Hellmann-Feynman census for this level
            cen = float(np.mean([np.sum(np.abs(vectors[:, k]) ** 2 * nf)
                                 for k in range(sl.start, sl.stop)])) - nf_gs
            lv.append({"omega": e, "share": wt / S_q, "purity": pur,
                       "census": cen, "mult": int(sl.stop - sl.start)})
        ov = float(np.trace(Ms[0] @ Ms[1]).real
                   / (np.trace(Ms[0]).real * np.trace(Ms[1]).real))
        lv.sort(key=lambda r: r["omega"])
        rec = {"q": [float(x) for x in q], "S": S_q, "levels": lv,
               "overlap": ov, "longitudinal_residue": longres,
               "splitting": abs(lv[0]["omega"] - lv[1]["omega"])
               / (0.5 * (lv[0]["omega"] + lv[1]["omega"]))}
        out["L"].append(rec)
        log(f"  L at {np.round(q,3)}:  S = {S_q:.5f}   "
            f"omega = {lv[0]['omega']:.4f} (share {lv[0]['share']:.3f}, "
            f"census {lv[0]['census']:+.3f}) / {lv[1]['omega']:.4f} "
            f"(share {lv[1]['share']:.3f}, census {lv[1]['census']:+.3f})",
            t0)
        log(f"      splitting {rec['splitting']:.4f}   overlap {ov:.2e}   "
            f"purity {lv[0]['purity']:.6f}/{lv[1]['purity']:.6f}   "
            f"long {longres:.1e}", t0)
    return out


def main() -> int:
    t0 = time.perf_counter()
    allout = []
    for shape in SHAPES:
        try:
            allout.append(analyse(shape, t0))
        except Exception as exc:
            log(f"shape {shape} FAILED: {type(exc).__name__}: {exc}", t0)
            allout.append({"shape": list(shape), "error": str(exc)})
    print("\n=== shape dependence at 48 sites ===")
    print("  shape        E0/site    S(L)     omega_low  share   census   "
          "split")
    for r in allout:
        if "L" not in r or not r["L"]:
            print(f"  {str(tuple(r['shape'])):<12s} "
                  f"{r.get('error','no L momentum')}")
            continue
        b = r["L"][0]
        lo = b["levels"][0]
        print(f"  {str(tuple(r['shape'])):<12s} {r['E0_per_site']:8.5f}  "
              f"{b['S']:.5f}  {lo['omega']:8.4f}  {lo['share']:.3f}  "
              f"{lo['census']:+.3f}  {b['splitting']:.4f}")
    with open(OUT / "shape_dependence_L.json", "w") as fh:
        json.dump(allout, fh, indent=1, default=float)
    print(f"\nwrote shape_dependence_L.json "
          f"({time.perf_counter()-t0:.0f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
