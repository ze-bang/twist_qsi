"""Landing the L intruder: is it the roton of the emergent gauge field?

Two tests:

1. FEYNMAN-BIJL / SMA.  From the exact full spectrum (ring48_channels.npz):
   S(q) = sum_n w_n (equal-time structure factor) and f1(q) = sum_n w_n omega_n.
   The single-mode bound omega_SMA = f1/S dips wherever S(q) peaks.  If S(q)
   peaks at L and omega_SMA(L) sits far below the photon line while tracking
   it elsewhere, the intruder is the density (roton) mode of the ice liquid.
   Also: how much of the SMA state S^z(q_L)|0> is exhausted by the intruder.

2. SOFT-MODE TRAJECTORY.  Quantum ice is known to exit the liquid near
   V/K ~ -0.5 (Shannon et al.; ground state moves to a different flux
   sector).  Extend the scan to V = -0.6, -0.75, -0.9, -1.1 and record:
   (a) the intruder energy at L (does it condense?);
   (b) the full-manifold ground sector + low-lying sector occupancies
       (does the GS cross to another transport sector?);
   (c) S^zz(q) totals (does the L peak grow toward the transition?).
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
from run_momentum_resolved_spectrum import reduce_momentum, star  # noqa: E402
from run_ring48_dssf import contractible                         # noqa: E402
from ring48_momenta_fix import fcc_momenta                       # noqa: E402
from ring64_pipeline import (fast_labels, fast_ring,             # noqa: E402
                             hex_masks_patterns, probe_columns)

OUT = ROOT / "campaign" / "outputs" / "ring_model_dssf"
EDEGEN = 1e-8
VEXT = [-0.6, -0.75, -0.9, -1.1]


def main() -> int:
    t0 = time.perf_counter()

    # ---------------- 1. SMA from the exact V=0 spectrum -----------------
    z = np.load(OUT / "ring48_channels.npz", allow_pickle=True)
    ge, gw = z["group_energies"], z["odd"]
    qred = np.round(z["qreduced"], 6)
    print("1. Feynman-Bijl analysis (exact moments, V=0):")
    print("   q                    S(q)      f1        omega_SMA   "
          "omega_low   w_low/S")
    seen = {}
    sma = []
    for qi in range(gw.shape[1]):
        w = gw[:, qi]
        S = float(w.sum())
        if S < 1e-10:
            continue
        key = tuple(np.round(np.sort(np.abs(qred[qi])), 4))
        if key in seen:
            continue
        seen[key] = True
        f1 = float(np.dot(w, ge))
        wl = int(np.argmax(w))
        lab = star(z["momenta"][qi])
        print(f"   {lab:<20s} {S:.5f}   {f1:.5f}   {f1/S:.4f}      "
              f"{ge[wl]:.4f}      {w[wl]/S:.3f}")
        sma.append({"star": lab, "S": S, "f1": f1, "omega_sma": f1 / S,
                    "omega_dominant": float(ge[wl]),
                    "dominant_share": float(w[wl] / S)})
    rec = {"sma": sma}

    # ---------------- 2. extended negative-V trajectory ------------------
    cl = geometry.build_cluster("fcc", (2, 2, 3))
    n = int(cl.n_sites)
    states = np.sort(np.asarray(cl.ice_states, dtype=np.uint64))
    positions = np.asarray(cl.positions, dtype=float)
    labels, spins_all = fast_labels(states, positions)
    hmp = hex_masks_patterns(contractible(cl))
    momenta, _ = fcc_momenta(cl)
    Lq = next(qi for qi, m in enumerate(momenta) if star(m) == "L")

    print("\n2. soft-mode trajectory (known quantum-ice exit near V ~ -0.5):")
    print("   V       E0(full)     GS sector(occ)      next sectors     "
          "L lines (omega:share)   S_L")
    track = []
    for v in VEXT:
        ham, _ = fast_ring(states, hmp, v)
        vals, vecs = eigsh(ham, k=3, which="SA", tol=1e-10, ncv=60)
        order = np.argsort(vals)
        occ = []
        for k in order:
            per = {int(s): float(np.sum(vecs[labels == s, k] ** 2))
                   for s in np.unique(labels)}
            best = max(per, key=per.get)
            occ.append((float(vals[k]), best, per[best]))
        sector = occ[0][1]
        rows = np.where(labels == sector)[0]
        values, vectors = eigh(np.asarray(
            ham[np.ix_(rows, rows)].todense()))
        omega = values - values[0]
        gs = vectors[:, 0].astype(complex)
        w = np.zeros(len(rows))
        for col in probe_columns(spins_all[rows], positions, momenta[Lq],
                                 gs, n):
            w += np.abs(vectors.conj().T @ col) ** 2 / n
        S_L = float(w.sum())
        groups, start = [], 0
        for i in range(1, len(omega) + 1):
            if i == len(omega) or omega[i] - omega[start] > EDEGEN:
                groups.append((float(omega[start]), slice(start, i)))
                start = i
        gwv = np.array([float(w[sl].sum()) for _, sl in groups])
        gev = np.array([e for e, _ in groups])
        top = np.argsort(gwv)[::-1][:3]
        lines = "  ".join(f"{gev[i]:.4f}:{gwv[i]/S_L:.3f}" for i in top)
        nxt = ", ".join(f"{s}({o:.2f})@{e - occ[0][0]:.4f}"
                        for e, s, o in occ[1:])
        print(f"   {v:+.2f}   {occ[0][0]:+.5f}   {sector}({occ[0][2]:.4f})"
              f"   {nxt}   {lines}   {S_L:.4f}", flush=True)
        track.append({"V": v, "E0": occ[0][0], "gs_sector": int(sector),
                      "gs_occ": occ[0][2],
                      "low_states": [[e, int(s), o] for e, s, o in occ],
                      "L_lines": [[float(gev[i]), float(gwv[i] / S_L)]
                                  for i in top],
                      "S_L": S_L})
    rec["vext"] = track

    with open(OUT / "ring48_roton.json", "w") as fh:
        json.dump(rec, fh, indent=1, default=float)
    print(f"\nwrote ring48_roton.json ({time.perf_counter()-t0:.0f} s)")
    print("READ: roton reading requires (a) S(q) max and omega_SMA min at L;")
    print("(b) the intruder line falling toward zero (or a sector crossing)")
    print("as V decreases; (c) S_L growing on approach.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
