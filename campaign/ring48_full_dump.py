"""Dump the FULL group-resolved S^zz of the 48-site ring model (all groups,
not top-4) so lineshapes can be drawn.  Same validated pipeline as
ring48_momenta_fix; output ring48_full_spectrum.npz."""

from __future__ import annotations

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
from qsi_campaign.protocol import polarization_sector_labels    # noqa: E402
from run_momentum_resolved_spectrum import (                    # noqa: E402
    channel_weights, reduce_momentum, star)
from run_ring48_dssf import contractible, ring_matrix           # noqa: E402
from ring48_momenta_fix import fcc_momenta                      # noqa: E402

OUT = ROOT / "campaign" / "outputs" / "ring_model_dssf"
EDEGEN = 1e-8


def main() -> int:
    t0 = time.perf_counter()
    cluster = geometry.build_cluster("fcc", (2, 2, 3))
    n = int(cluster.n_sites)
    states = np.sort(np.asarray(cluster.ice_states, dtype=np.uint64))
    labels = polarization_sector_labels(cluster)
    ham, _ = ring_matrix(states, contractible(cluster), 0.0)
    e0, v0 = eigsh(ham, k=1, which="SA", tol=1e-11, ncv=80)
    occ = {int(s): float(np.sum(v0[labels == s, 0] ** 2))
           for s in np.unique(labels)}
    sector = max(occ, key=occ.get)
    rows = np.where(labels == sector)[0]
    values, vectors = eigh(np.asarray(ham[np.ix_(rows, rows)].todense()))
    omega = values - values[0]

    momenta, shape = fcc_momenta(cluster)
    positions = np.asarray(cluster.positions, dtype=float)
    total = channel_weights(vectors, states[rows], positions, momenta,
                            n).sum(axis=2)

    groups, start = [], 0
    for i in range(1, len(omega) + 1):
        if i == len(omega) or omega[i] - omega[start] > EDEGEN:
            groups.append((float(omega[start]), slice(start, i)))
            start = i
    ge = np.array([e for e, _ in groups])
    gw = np.array([[float(total[sl, qi].sum()) for qi in range(len(momenta))]
                   for _, sl in groups])            # (n_groups, 12)

    np.savez(OUT / "ring48_full_spectrum.npz",
             group_energies=ge, group_weights=gw,
             momenta=np.asarray(momenta),
             m_indices=np.array(list(np.ndindex(*shape))),
             qmags=np.array([float(np.linalg.norm(reduce_momentum(m)))
                             for m in momenta]),
             stars=np.array([star(m) for m in momenta]))
    print(f"saved {len(ge)} groups x {len(momenta)} momenta "
          f"({time.perf_counter()-t0:.0f} s); totals per momentum:",
          np.round(gw.sum(axis=0), 5))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
