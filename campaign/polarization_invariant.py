"""Is the polarization assignment of the bright levels basis-independent?

The two Gaussian transverse branches are EXACTLY degenerate at every
momentum (sigma_1 = sigma_2), so the singular-value decomposition fixes
the two polarization directions only up to an arbitrary SO(2)/SU(2)
rotation inside that subspace.  Any statement of the form "this level
is 68% polarization 1" is therefore meaningless on its own.

The invariant content is the geometry of the levels' polarization
vectors relative to EACH OTHER.  For every bright level n we build the
2x2 polarization density matrix

    M_n = A_n^dag A_n ,   A_n = <n,alpha| S^z(q) |0>  projected on the
                                 transverse subspace,

with alpha running over the multiplet.  Then
  * purity  tr(M^2)/tr(M)^2  = 1 means the level couples to a single
    polarization direction (basis-independent statement),
  * the normalized overlap tr(M_A M_B)/[tr M_A tr M_B] between the two
    brightest levels is zero iff a basis exists in which each is pure,
    i.e. iff they really are the two distinct transverse branches.

Both are invariant under rotations of the degenerate subspace.
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


def main() -> int:
    t0 = time.perf_counter()
    cl = geometry.build_cluster("fcc", (2, 2, 3))
    n = int(cl.n_sites)
    states = np.sort(np.asarray(cl.ice_states, dtype=np.uint64))
    positions = np.asarray(cl.positions, dtype=float)
    labels, spins_all = fast_labels(states, positions)
    hmp = hex_masks_patterns(contractible(cl))
    ham, _ = fast_ring(states, hmp, 0.0)
    _, v0 = eigsh(ham, k=1, which="SA", tol=1e-11, ncv=80)
    occ = {int(s): float(np.sum(v0[labels == s, 0] ** 2))
           for s in np.unique(labels)}
    sector = max(occ, key=occ.get)
    rows = np.where(labels == sector)[0]
    values, vectors = eigh(np.asarray(ham[np.ix_(rows, rows)].todense()))
    omega = values - values[0]
    gs = vectors[:, 0].astype(complex)
    spins = spins_all[rows]
    print(f"sector {sector} dim {len(rows)} ({time.perf_counter()-t0:.0f} s)",
          flush=True)

    block = analytic_blocks(cl, unwrap_hexagons(cl, contractible(cl)))
    momenta, _ = fcc_momenta(cl)
    sub = np.arange(n) % 4
    groups, start = [], 0
    for i in range(1, len(omega) + 1):
        if i == len(omega) or omega[i] - omega[start] > EDEGEN:
            groups.append((float(omega[start]), slice(start, i)))
            start = i

    rec, seen = {}, set()
    for qi, momentum in enumerate(momenta):
        lab = star(momentum)
        phase = np.exp(-2j * np.pi * (positions @ momentum))
        A = np.zeros((len(rows), 4), dtype=complex)
        for mu in range(4):
            A[:, mu] = vectors.conj().T @ ((spins @ (phase * (sub == mu)))
                                           * gs)
        A /= np.sqrt(n)
        w_tot = np.sum(np.abs(A) ** 2, axis=1)
        S_q = float(w_tot.sum())
        if S_q < 1e-12:
            continue
        u, s, vh = np.linalg.svd(block(-momentum), full_matrices=True)
        deg = abs(s[0] - s[1]) < 1e-10
        T = (A @ vh.conj().T)[:, :2]            # transverse amplitudes

        bright = sorted(
            [(float(w_tot[sl].sum()), e, sl) for e, sl in groups],
            key=lambda r: -r[0])[:2]
        Ms, info = [], []
        for wt, e, sl in bright:
            An = T[sl]                          # (multiplicity, 2)
            M = An.conj().T @ An
            tr = float(np.trace(M).real)
            pur = float((np.trace(M @ M).real) / tr ** 2) if tr > 0 else 0.0
            Ms.append(M)
            info.append({"omega": e, "share": wt / S_q, "purity": pur,
                         "mult": int(sl.stop - sl.start)})
        ov = float(np.trace(Ms[0] @ Ms[1]).real
                   / (np.trace(Ms[0]).real * np.trace(Ms[1]).real))
        split = abs(info[0]["omega"] - info[1]["omega"]) \
            / (0.5 * (info[0]["omega"] + info[1]["omega"]))
        key = f"{lab}"
        if key in seen:
            continue
        seen.add(key)
        print(f"\n{lab}   S={S_q:.5f}   sigma degenerate: {deg}")
        for d in info:
            print(f"   omega {d['omega']:7.4f}  share {d['share']:.4f}  "
                  f"multiplicity {d['mult']}  polarization purity "
                  f"{d['purity']:.6f}")
        print(f"   normalized overlap of the two bright levels: {ov:.3e}")
        print(f"   fractional splitting of the two branches:    {split:.4f}")
        rec[lab] = {"S": S_q, "levels": info, "overlap": ov,
                    "splitting": split, "sigma_degenerate": bool(deg)}

    print("\n=== summary: branch splitting and polarization overlap ===")
    print("  momentum              w1       w2      split    overlap")
    for k, v in rec.items():
        print(f"  {k:<20s} {v['levels'][0]['omega']:6.3f}  "
              f"{v['levels'][1]['omega']:6.3f}  {v['splitting']:7.4f}  "
              f"{v['overlap']:.2e}")
    with open(OUT / "ring48_pol_invariant.json", "w") as fh:
        json.dump(rec, fh, indent=1, default=float)
    print(f"\nwrote ring48_pol_invariant.json "
          f"({time.perf_counter()-t0:.0f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
