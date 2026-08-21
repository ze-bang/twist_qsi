"""Why L?  Feynman-Bijl at EVERY momentum, and along the RK trajectory.

Part 1.  All 12 momenta of the 48-site cluster (the earlier run collapsed
   inequivalent stars by an over-aggressive key).  For each: S(q), the
   analytic f_1(q), omega_SMA = f_1/S, the exact lowest bright line, and
   the ratio.  Establishes whether L is the maximum of S and the minimum
   of omega_SMA among ALL accessible momenta, not just a chosen five.

Part 2.  The Rokhsar-Kivelson trajectory.  The V*n_flip term is DIAGONAL
   in the ice basis, so it commutes with S^z(q) and does not enter the
   first moment at all: f_1(q) keeps its form, with w_p = <W_p+W_p^dag>
   evaluated in the V-dependent ground state.  So Feynman-Bijl makes a
   parameter-free prediction for how the roton softens as V is lowered:
       omega_SMA(V) = f_1(V) / S_L(V).
   Comparing that with the exact roton energy along the trajectory is an
   independent test of the mechanism -- the softening should track the
   growth of S_L, not merely be correlated with it.
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
from ring64_pipeline import (fast_labels, fast_ring,            # noqa: E402
                             hex_masks_patterns, probe_columns)
from ring48_roton_landing import (f1_analytic, hex_signed,      # noqa: E402
                                  plaquette_w)

OUT = ROOT / "campaign" / "outputs" / "ring_model_dssf"
VSCAN = [0.0, -0.2, -0.4, -0.6, -0.8, -1.0, -1.1]


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

    # ---------------- Part 1: every momentum, V = 0 ----------------
    ham, _ = fast_ring(states, hmp, 0.0)
    _, v0 = eigsh(ham, k=1, which="SA", tol=1e-11, ncv=60)
    per = {int(s): float(np.sum(v0[labels == s, 0] ** 2))
           for s in np.unique(labels)}
    sector = max(per, key=per.get)
    rows = np.where(labels == sector)[0]
    hs = np.asarray(ham[np.ix_(rows, rows)].todense())
    values, vectors = eigh(hs)
    e0 = float(values[0])
    omega = values - e0
    gs = vectors[:, 0].astype(complex)
    sec_states = states[rows]
    wp = plaquette_w(sec_states, hmp, gs)

    print(f"sector {sector} dim {len(rows)}  E0={e0:.6f}")
    print("\nPart 1: Feynman-Bijl at EVERY cluster momentum (V=0)")
    print("  q                       |q|     S(q)      f1        "
          "om_SMA    om_bright  ratio")
    rowsout = []
    for qi, q in enumerate(momenta):
        cols = probe_columns(spins_all[rows], positions, q, gs, n)
        wq = np.zeros(len(values))
        for col in cols:
            wq += np.abs(vectors.conj().T @ col) ** 2 / n
        S = float(wq.sum())
        if S < 1e-10:
            print(f"  {star(q):<22s} {np.linalg.norm(q):.3f}   "
                  f"DARK (S={S:.2e})")
            continue
        f1 = float(f1_analytic([np.asarray(q)], hexsign, positions,
                               wp, n)[0])
        jb = int(np.argmax(wq))
        rowsout.append({"q": [float(x) for x in q], "star": star(q),
                        "S": S, "f1": f1, "omega_sma": f1 / S,
                        "omega_bright": float(omega[jb]),
                        "ratio": float(omega[jb] / (f1 / S)),
                        "share": float(wq[jb] / S)})
        print(f"  {star(q):<22s} {np.linalg.norm(q):.3f}   {S:.5f}   "
              f"{f1:.5f}   {f1/S:.4f}    {omega[jb]:.4f}    "
              f"{omega[jb]/(f1/S):.4f}", flush=True)
    rec["all_momenta"] = rowsout
    if rowsout:
        iS = max(range(len(rowsout)), key=lambda i: rowsout[i]["S"])
        iW = min(range(len(rowsout)), key=lambda i: rowsout[i]["omega_sma"])
        iB = min(range(len(rowsout)),
                 key=lambda i: rowsout[i]["omega_bright"])
        print(f"\n  max S(q)          : {rowsout[iS]['star']} "
              f"(S={rowsout[iS]['S']:.4f})")
        print(f"  min omega_SMA     : {rowsout[iW]['star']} "
              f"(={rowsout[iW]['omega_sma']:.4f})")
        print(f"  min bright line   : {rowsout[iB]['star']} "
              f"(={rowsout[iB]['omega_bright']:.4f})")
        rec["argmax_S"] = rowsout[iS]["star"]
        rec["argmin_sma"] = rowsout[iW]["star"]
        rec["argmin_bright"] = rowsout[iB]["star"]

    # ---------------- Part 2: RK trajectory ----------------
    Lq = next(qi for qi, m in enumerate(momenta) if star(m) == "L")
    print("\nPart 2: Feynman-Bijl along the RK trajectory at L")
    print("  (V*n_flip is diagonal -> does NOT enter f_1)")
    print("   V       E0        S_L      f1_L     om_SMA   om_roton  ratio")
    traj = []
    for v in VSCAN:
        hv, _ = fast_ring(states, hmp, v)
        _, vv = eigsh(hv, k=1, which="SA", tol=1e-11, ncv=60)
        perv = {int(s): float(np.sum(vv[labels == s, 0] ** 2))
                for s in np.unique(labels)}
        sec = max(perv, key=perv.get)
        rr = np.where(labels == sec)[0]
        hsv = np.asarray(hv[np.ix_(rr, rr)].todense())
        val, vec = eigh(hsv)
        ev0 = float(val[0])
        om = val - ev0
        g = vec[:, 0].astype(complex)
        wpv = plaquette_w(states[rr], hmp, g)
        cols = probe_columns(spins_all[rr], positions, momenta[Lq], g, n)
        wq = np.zeros(len(val))
        for col in cols:
            wq += np.abs(vec.conj().T @ col) ** 2 / n
        S = float(wq.sum())
        f1 = float(f1_analytic([np.asarray(momenta[Lq])], hexsign,
                               positions, wpv, n)[0])
        jb = int(np.argmax(wq))
        traj.append({"V": v, "E0": ev0, "S_L": S, "f1_L": f1,
                     "omega_sma": f1 / S, "omega_roton": float(om[jb]),
                     "ratio": float(om[jb] / (f1 / S)), "sector": int(sec)})
        print(f"  {v:+.2f}  {ev0:+9.4f}  {S:.5f}  {f1:.5f}  "
              f"{f1/S:.4f}   {om[jb]:.4f}    {om[jb]/(f1/S):.4f}",
              flush=True)
    rec["rk_trajectory"] = traj
    if len(traj) > 1:
        r = np.array([t["ratio"] for t in traj])
        print(f"\n  omega_roton/omega_SMA along the trajectory: "
              f"mean {r.mean():.4f}  spread {r.max()-r.min():.4f}")
        print("  (a small spread means the softening is TRACKED by the "
              "Feynman-Bijl\n   ratio f_1/S, i.e. the mechanism, not a "
              "coincidence)")
        rec["traj_ratio_mean"] = float(r.mean())
        rec["traj_ratio_spread"] = float(r.max() - r.min())

    with open(OUT / "roton_mechanism.json", "w") as fh:
        json.dump(rec, fh, indent=1, default=float)
    print(f"\nwrote roton_mechanism.json ({time.perf_counter()-t0:.0f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
