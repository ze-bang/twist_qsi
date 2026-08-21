"""Separate the COST of painting the wave from the RELIEF the state finds.

The neutron operator is diagonal in the ice basis: S^z(q) multiplies each
ice configuration c by the Fourier amplitude of its own spin pattern,
    phi(c) = sum_i e^{-i q.r_i} sigma_i(c) / 2 .
It creates nothing; it RE-WEIGHTS the resonating superposition.

The ring exchange resonates between c and c' = W_p c.  Flipping hexagon p
changes the painted amplitude by exactly g_p(q) -- the same object that
appears in the analytic first moment.  So:

  * hexagons with g_p != 0 : partners now carry different weights, and the
    resonance interferes less well.  THIS is the energy cost.
  * hexagons with g_p  = 0 : flipping p does not change the weight at all,
    so those resonances are not degraded by the phase mismatch.

At L one whole family has g_p = 0 exactly.  Prediction: in the SMA state
that family keeps its resonance (no destructive mismatch), while the true
roton sacrifices it -- and that sacrifice is the backflow.  We test this by
computing the per-family resonance change in BOTH states.
"""

from __future__ import annotations

import json
import sys
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
from roton_realspace import hex_normals, plaq_expect            # noqa: E402

OUT = ROOT / "campaign" / "outputs" / "ring_model_dssf"
EDEGEN = 1e-8


def main() -> int:
    cl = geometry.build_cluster("fcc", (2, 2, 3))
    n = int(cl.n_sites)
    states = np.sort(np.asarray(cl.ice_states, dtype=np.uint64))
    positions = np.asarray(cl.positions, dtype=float)
    labels, spins_all = fast_labels(states, positions)
    hexes = contractible(cl)
    hmp = hex_masks_patterns(hexes)
    momenta, _ = fcc_momenta(cl)
    _, family = hex_normals(hexes, positions)

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
    sec = states[rows]

    Lq = next(qi for qi, m in enumerate(momenta) if star(m) == "L")
    qL = np.asarray(momenta[Lq], dtype=float)

    # SMA state = the raw painted state, S^z(L)|0> (all four channels)
    cols = probe_columns(spins_all[rows], positions, qL, gs, n)
    wq = np.zeros(len(values))
    amp = np.zeros(len(values), dtype=complex)
    for c in cols:
        pr = vectors.conj().T @ c
        wq += np.abs(pr) ** 2 / n
        amp += pr
    # The painted wave is NOT the coherent sum of the four sublattice
    # channels (that combination interferes and is not the SMA state).
    # It is the best state in the 4-dimensional space they span.
    V = np.column_stack([c for c in cols])
    Q, R = np.linalg.qr(V)
    Q = Q[:, np.abs(np.diag(R)) > 1e-10]
    Hb = Q.conj().T @ (hs @ Q)
    ev, evec = np.linalg.eigh((Hb + Hb.conj().T) / 2.0)
    sma = Q @ evec[:, 0]
    sma = sma / np.linalg.norm(sma)
    e_sma = float(ev[0]) - e0

    jb = int(np.argmax(wq))
    grp = np.where(np.abs(values - values[jb]) < EDEGEN)[0]
    tru = vectors[:, grp] @ amp[grp]
    tru = tru / np.linalg.norm(tru)

    wg, ng = plaq_expect(sec, hmp, gs)
    ws, ns = plaq_expect(sec, hmp, sma.astype(complex))
    wt, nt = plaq_expect(sec, hmp, tru.astype(complex))

    print(f"energies:  ground 0   SMA(painted) {e_sma:.5f} K   "
          f"true roton {omega[jb]:.5f} K")
    print(f"the state relieves {e_sma - omega[jb]:.5f} K "
          f"({100*(e_sma-omega[jb])/e_sma:.0f}% of the painting cost)\n")
    print("resonance change <W_p+W_p^dag> - ground, per hexagon family:")
    print("  family   g_p(L)     painted(SMA)   true roton    relief")
    for f in range(4):
        m = family == f
        dS = float((ws[m] - wg[m]).sum())
        dT = float((wt[m] - wg[m]).sum())
        tag = "DECOUPLED" if f == 2 else ""
        print(f"   {f}       {'0' if f == 2 else '8'}          "
              f"{dS:+.5f}      {dT:+.5f}     {dT-dS:+.5f}  {tag}")
    dS = float((ws - wg).sum())
    dT = float((wt - wg).sum())
    print(f"  TOTAL               {dS:+.5f}      {dT:+.5f}     "
          f"{dT-dS:+.5f}")
    print(f"  (-total must equal the energy: {-dS:.5f} vs {e_sma:.5f} ; "
          f"{-dT:.5f} vs {omega[jb]:.5f})")

    rec = {"e_sma": e_sma, "e_true": float(omega[jb]),
           "dW_sma_by_family": [float((ws[family == f] - wg[family == f]
                                       ).sum()) for f in range(4)],
           "dW_true_by_family": [float((wt[family == f] - wg[family == f]
                                        ).sum()) for f in range(4)]}
    with open(OUT / "roton_sma_vs_true.json", "w") as fh:
        json.dump(rec, fh, indent=1)
    print("\nwrote roton_sma_vs_true.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
