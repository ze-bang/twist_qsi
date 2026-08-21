"""Is the L intruder the second transverse polarization, rotonized?

Decisive test: project the neutron probe onto the two GAUSSIAN polarization
vectors (from the analytic curl SVD) and resolve every bright state's weight
into (pol-1, pol-2, longitudinal).  Outcomes:

  * intruder and photon couple to ORTHOGONAL polarizations
        -> no new particle: polarization-selective renormalization;
           one polarization rotonized at L, stiffened at X.
  * both couple to the same combination / both polarizations mixed
        -> the intruder is a genuine extra state beyond the two-mode space.

Gates: the 4 SVD directions (2 transverse + kernel) resolve the identity on
sublattice space, so per state  sum_lambda w_lambda + w_kernel = w_total
exactly; and div E = 0 makes the longitudinal residue a further check.
Probe at +q pairs with modes at -q (established convention).
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
from run_momentum_resolved_spectrum import star                  # noqa: E402
from run_ring48_dssf import contractible                         # noqa: E402
from ring48_momenta_fix import fcc_momenta                       # noqa: E402
from ring48_channels import unwrap_hexagons                      # noqa: E402
from ring48_channels import analytic_blocks                      # noqa: E402
from ring64_pipeline import fast_labels, fast_ring, hex_masks_patterns  # noqa: E402

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
    e0, v0 = eigsh(ham, k=1, which="SA", tol=1e-11, ncv=80)
    occ = {int(s): float(np.sum(v0[labels == s, 0] ** 2))
           for s in np.unique(labels)}
    sector = max(occ, key=occ.get)
    rows = np.where(labels == sector)[0]
    values, vectors = eigh(np.asarray(ham[np.ix_(rows, rows)].todense()))
    omega = values - values[0]
    gs = vectors[:, 0].astype(complex)
    spins = spins_all[rows]
    print(f"sector {sector} dim {len(rows)} diagonalized "
          f"({time.perf_counter()-t0:.0f} s)", flush=True)

    block = analytic_blocks(cl, unwrap_hexagons(cl, contractible(cl)))
    momenta, _ = fcc_momenta(cl)
    sub = np.arange(n) % 4
    rec = {}
    for qi, momentum in enumerate(momenta):
        lab = star(momentum)
        if lab not in ("L", "X"):
            continue
        # per-sublattice amplitudes A_{n,mu} = <n| S^z_mu(q) |0>
        phase = np.exp(-2j * np.pi * (positions @ momentum))
        A = np.zeros((len(rows), 4), dtype=complex)
        for mu in range(4):
            A[:, mu] = vectors.conj().T @ ((spins @ (phase * (sub == mu)))
                                           * gs)
        A /= np.sqrt(n)
        w_tot = np.sum(np.abs(A) ** 2, axis=1)

        # Gaussian mode directions at -q (probe convention), full 4-dim set
        u, s, vh = np.linalg.svd(block(-momentum), full_matrices=True)
        # rows of vh: sublattice-space directions; s>0 = transverse
        w_dir = np.abs(A @ vh.conj().T) ** 2        # (n_states, 4)
        resolve = float(np.max(np.abs(w_dir.sum(axis=1) - w_tot)))
        print(f"\n{lab}  (momentum index {qi}); completeness residue "
              f"{resolve:.2e}  sigma = "
              + ", ".join(f"{x:.4f}" for x in s), flush=True)
        if resolve > 1e-10:
            print("  *** completeness FAILED: polarization split invalid ***")
            continue

        groups, start = [], 0
        for i in range(1, len(omega) + 1):
            if i == len(omega) or omega[i] - omega[start] > EDEGEN:
                groups.append((float(omega[start]), slice(start, i)))
                start = i
        print("   omega      total     pol-1     pol-2     kernel")
        out = []
        for e, sl in groups:
            wt = float(w_tot[sl].sum())
            if wt < 3e-3:
                continue
            d = w_dir[sl].sum(axis=0)
            print(f"   {e:.4f}   {wt:.5f}   {d[0]:.5f}   {d[1]:.5f}   "
                  f"{d[2]+d[3]:.2e}")
            out.append({"omega": e, "total": wt,
                        "pol": [float(d[0]), float(d[1])],
                        "kernel": float(d[2] + d[3])})
            if e > 4.5:
                break
        rec[lab] = {"sigma": [float(x) for x in s], "lines": out}
        if lab == "L" and "L_amps" not in rec:
            keep = [i for i, (e, sl) in enumerate(groups) if e < 2.0]
            amps = {}
            for e, sl in groups[:8]:
                if float(w_tot[sl].sum()) > 3e-3:
                    a = (A[sl] @ vh.conj().T)[:, :2]
                    amps[f"{e:.4f}"] = [[float(x.real), float(x.imag)]
                                        for x in a.ravel()]
            rec["L_amps"] = amps

    with open(OUT / "ring48_polarization.json", "w") as fh:
        json.dump(rec, fh, indent=1, default=float)
    print(f"\nwrote ring48_polarization.json ({time.perf_counter()-t0:.0f} s)")
    print("READ: orthogonal pol-1/pol-2 assignment of intruder vs photon")
    print("      -> rotonized second polarization; shared/mixed -> new state.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
