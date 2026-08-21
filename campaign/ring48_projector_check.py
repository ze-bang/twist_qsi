"""Momentum labeling on fcc(2,2,3), verified by translation projectors.

The S^z-probe route to momentum labels produced a puzzle: the same energies
carry weight at several momenta.  Two possible causes: (a) genuine accidental
degeneracies between states of different momenta (plausible: 9104 states share
4481 distinct energies), or (b) a residual error in the momentum vectors.

This decides it with the projector P_q = (1/N) sum_t chi_q(t) T_t, which never
touches the constructed momenta except through the characters:

  1. COMPLETENESS: sum_q ||P_q v||^2 = ||v||^2 for every eigenvector.  Fails
     unless the 12 momenta are exactly the cluster's character set.
  2. GROUND STATE at Gamma: ||P_Gamma |0>||^2 = 1.
  3. PROBE CONSISTENCY: S^z_mu(q)|0> must lie entirely in momentum sector q.
     r_q = ||P_q S^z(q)|0>||^2 / ||S^z(q)|0>||^2 = 1 is the decisive check.
  4. The recurring energies: Tr(P_G P_q) per degenerate group G -- are the
     groups that appear at several q genuinely multi-momentum multiplets?

Translations preserve the transport sector (every ice state has total S^z = 0,
so the polarization is translation invariant), so P_q acts within the block.
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
from qsi_campaign.protocol import polarization_sector_labels    # noqa: E402
from run_momentum_resolved_spectrum import (                    # noqa: E402
    permute_states, translation_permutations)
from run_ring48_dssf import contractible, ring_matrix           # noqa: E402
from ring48_momenta_fix import fcc_momenta                      # noqa: E402

OUT = ROOT / "campaign" / "outputs" / "ring_model_dssf"
EDEGEN = 1e-8


def main() -> int:
    t0 = time.perf_counter()
    cluster = geometry.build_cluster("fcc", (2, 2, 3))
    n_sites = int(cluster.n_sites)
    states = np.sort(np.asarray(cluster.ice_states, dtype=np.uint64))
    labels = polarization_sector_labels(cluster)
    ham, _ = ring_matrix(states, contractible(cluster), 0.0)
    e0, v0 = eigsh(ham, k=1, which="SA", tol=1e-11, ncv=80)
    occ = {int(s): float(np.sum(v0[labels == s, 0] ** 2))
           for s in np.unique(labels)}
    sector = max(occ, key=occ.get)
    rows = np.where(labels == sector)[0]
    sec_states = states[rows]                    # ascending (states sorted)
    values, vectors = eigh(np.asarray(ham[np.ix_(rows, rows)].todense()))
    omega = values - values[0]
    print(f"sector {sector} dim {len(rows)}  ({time.perf_counter()-t0:.0f} s)",
          flush=True)

    momenta, _ = fcc_momenta(cluster)
    translations, permutations = translation_permutations(cluster)
    if len(translations) != len(momenta):
        raise SystemExit("translation/momentum count mismatch")

    # row map for each translation: (T_t v)[j] = v[i] with s_j = pi_t(s_i)
    maps = []
    for perm in permutations:
        permuted = permute_states(sec_states, perm)
        j = np.searchsorted(sec_states, permuted)
        if not np.array_equal(sec_states[j], permuted):
            raise SystemExit("translation left the sector: map is broken")
        maps.append(j)

    phases = np.exp(-2j * np.pi *
                    (np.asarray(translations) @ np.asarray(momenta).T))

    def project(columns, qi):
        acc = np.zeros(columns.shape, dtype=complex)
        for t, j in enumerate(maps):
            tv = np.zeros(columns.shape, dtype=complex)
            tv[j, :] = columns
            acc += phases[t, qi] * tv
        return acc / len(maps)

    # ---- 1. completeness on every eigenvector --------------------------
    norms = np.zeros((len(momenta), vectors.shape[1]))
    for qi in range(len(momenta)):
        norms[qi] = np.sum(np.abs(project(vectors, qi)) ** 2, axis=0)
    complete = norms.sum(axis=0)
    print(f"completeness: min {complete.min():.10f}, max {complete.max():.10f}"
          f"  -> {'PASS' if abs(complete-1).max() < 1e-8 else 'FAIL'}",
          flush=True)

    # ---- 2. ground state momentum --------------------------------------
    gs = norms[:, 0]
    print(f"ground state: max ||P_q|0>||^2 = {gs.max():.8f} at q index "
          f"{int(np.argmax(gs))} (q={[round(float(v),3) for v in momenta[np.argmax(gs)]]})",
          flush=True)

    # ---- 3. probe consistency ------------------------------------------
    bits = ((sec_states[:, None] >> np.arange(n_sites, dtype=np.uint64))
            & np.uint64(1)).astype(float) - 0.5
    positions = np.asarray(cluster.positions, dtype=float)
    sublattice = np.arange(n_sites) % 4
    worst = 1.0
    for qi, momentum in enumerate(momenta):
        phase = np.exp(-2j * np.pi * (positions @ momentum))
        cols = []
        for mu in range(4):
            col = (bits @ (phase * (sublattice == mu))) * vectors[:, 0]
            if np.linalg.norm(col) > 1e-12:
                cols.append(col / np.linalg.norm(col))
        if not cols:
            continue
        cols = np.array(cols).T
        r = np.sum(np.abs(project(cols, qi)) ** 2) / cols.shape[1]
        worst = min(worst, r)
    print(f"probe consistency: worst r_q = {worst:.8f}  -> "
          f"{'PASS: momenta correct, weights stand' if worst > 0.999 else 'FAIL: momenta wrong'}",
          flush=True)

    # ---- 4. the recurring energies -------------------------------------
    groups, start = [], 0
    for i in range(1, len(omega) + 1):
        if i == len(omega) or omega[i] - omega[start] > EDEGEN:
            groups.append((float(omega[start]), slice(start, i)))
            start = i
    print("\nrecurring energies: momentum content Tr(P_G P_q) per group")
    for target in (1.8949, 1.9802, 2.5327, 2.6867):
        for e, sl in groups:
            if abs(e - target) < 2e-4:
                content = norms[:, sl].sum(axis=1)
                nz = [(qi, round(float(c), 3)) for qi, c in enumerate(content)
                      if c > 1e-6]
                print(f"  omega={e:.4f} dim={sl.stop-sl.start}: {nz}")
                break

    with open(OUT / "ring48_projector_check.json", "w") as fh:
        json.dump({"completeness_err": float(abs(complete - 1).max()),
                   "gs_momentum_weight": float(gs.max()),
                   "worst_probe_r": float(worst)}, fh, indent=1)
    print(f"\ndone ({time.perf_counter()-t0:.0f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
