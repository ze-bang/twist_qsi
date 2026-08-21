"""Is the roton really the lowest excitation?  Direct check on the exact
sector spectrum: list the lowest levels with momentum, S^zz weight, and
spin-flip (C) parity.  The block-Krylov run found Ritz values at 0.149K
(X) and 1.026K ((1/3,1/3,1/3)) far below the bright lines, which would
contradict the claim if they are real C-odd states.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.linalg import eigh

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
NSHOW = 30


def main() -> int:
    cl = geometry.build_cluster("fcc", (2, 2, 3))
    n = int(cl.n_sites)
    states = np.sort(np.asarray(cl.ice_states, dtype=np.uint64))
    positions = np.asarray(cl.positions, dtype=float)
    labels, spins_all = fast_labels(states, positions)
    hexes = contractible(cl)
    hmp = hex_masks_patterns(hexes)
    momenta, _ = fcc_momenta(cl)

    ham, _ = fast_ring(states, hmp, 0.0)
    from scipy.sparse.linalg import eigsh
    _, v0 = eigsh(ham, k=1, which="SA", tol=1e-11, ncv=60)
    per = {int(s): float(np.sum(v0[labels == s, 0] ** 2))
           for s in np.unique(labels)}
    sector = max(per, key=per.get)
    rows = np.where(labels == sector)[0]
    hs = np.asarray(ham[np.ix_(rows, rows)].todense())
    values, vectors = eigh(hs)
    e0 = values[0]
    omega = values - e0
    gs = vectors[:, 0].astype(complex)
    print(f"sector {sector} dim {len(rows)}  E0={e0:.6f}")

    # ---- spin-flip parity operator on the sector ----
    sec = states[rows]
    full = np.uint64((1 << n) - 1)
    flipped = sec ^ full
    pos = np.searchsorted(sec, flipped)
    ok = (pos < len(sec)) & (sec[np.minimum(pos, len(sec) - 1)] == flipped)
    print(f"spin-flip closes on sector: {ok.all()} "
          f"({np.count_nonzero(~ok)} escapes)")
    parity = np.full(len(values), np.nan)
    if ok.all():
        P = np.zeros((len(sec), len(sec)))
        P[pos, np.arange(len(sec))] = 1.0
        for j in range(len(values)):
            v = vectors[:, j]
            parity[j] = float(v @ (P @ v))

    # ---- momentum assignment + S^zz weight for every level ----
    wtot = np.zeros(len(values))
    qassign = np.full(len(values), -1)
    for qi, q in enumerate(momenta):
        cols = probe_columns(spins_all[rows], positions, q, gs, n)
        w = np.zeros(len(values))
        for col in cols:
            w += np.abs(vectors.conj().T @ col) ** 2 / n
        wtot += w
    # momentum label by translation projector would be heavier; use the
    # probe only to report brightness, and label by dominant q of weight
    for qi, q in enumerate(momenta):
        cols = probe_columns(spins_all[rows], positions, q, gs, n)
        w = np.zeros(len(values))
        for col in cols:
            w += np.abs(vectors.conj().T @ col) ** 2 / n
        better = w > 1e-12
        qassign[better & (w >= wtot - 1e-15)] = qi

    print("\nlowest excitations of the ground sector "
          "(omega, S^zz weight, parity, momentum if bright):")
    print(" idx   omega      Szz weight     parity   q(if bright)")
    shown = 0
    for j in range(1, len(values)):
        if shown >= NSHOW:
            break
        lab = star(momenta[qassign[j]]) if qassign[j] >= 0 else "-"
        print(f" {j:4d}  {omega[j]:9.5f}  {wtot[j]:12.3e}   "
              f"{parity[j]:+.3f}   {lab}")
        shown += 1

    bright = np.where(wtot > 1e-6)[0]
    bright = bright[bright > 0]
    print(f"\nlowest excitation overall:      {omega[1]:.5f} K "
          f"(weight {wtot[1]:.3e}, parity {parity[1]:+.3f})")
    print(f"lowest BRIGHT (S^zz) excitation: {omega[bright[0]]:.5f} K "
          f"(weight {wtot[bright[0]]:.3e})")
    ndark = np.count_nonzero(omega[1:bright[0]] > 0)
    print(f"number of levels below the lowest bright line: {ndark}")
    print(f"parity of ground state: {parity[0]:+.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
