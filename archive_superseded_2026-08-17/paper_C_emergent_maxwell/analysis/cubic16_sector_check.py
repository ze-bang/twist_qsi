"""Exact (truncation-free) test of emergent electrostatics on cubic-16.

Why cubic-16, when the paper is about FCC-32:

  * On cubic-16 the virtual space at levels=4 IS the complete space, so the
    per-sector ground energies can be computed with NO depth truncation.  The
    FCC-32 numbers in the draft are all depth d=1, and the only place the drift
    is visible (d=1 -> d=2 at Jpm=-0.05) it is +34%.  Running the depth ladder
    1,2,3,4 here measures that systematic directly on the same observable.
  * cubic-16 reportedly has SIX degenerate ground transport sectors against
    FCC-32's one.  If the electrostatic minimum is not at T=0, Eq. (law) of the
    draft is in trouble and we want to know now.
  * It is a second host for the 1/L claim, which currently rests on one ratio.

What it does NOT do: give a better eps_eff.  cubic-16 is one conventional cell
across, so the weak-field window is essentially empty.  This is a systematics
and structure check, not a measurement.

Method is identical to campaign/run_fcc32_sector_grounds.py (bordered Hermitian
matrix per sector, sparse Lanczos) so the two clusters are directly comparable.

Run through the scheduler:  sbatch analysis/run_cubic16_check.sbatch
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy.sparse.linalg import eigsh

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "notes"))
sys.path.insert(0, str(ROOT / "campaign"))

import recompute_finite_size_artifact as geometry            # noqa: E402
from qsi_campaign.protocol import polarization_sector_labels  # noqa: E402
from run_truncated_feshbach import bfs_space, build_hamiltonian  # noqa: E402

OUT = Path(__file__).resolve().parent / "cubic16_sector_check.json"
COUPLINGS = [-0.05, -0.20, -0.30]
LEVELS = [1, 2, 3, 4]          # 4 == exact on cubic-16


def transport_vectors(cluster):
    """T = 2p per sector, and the sector label of every ice state.

    Reproduces protocol.polarization_sector_labels and attaches the actual
    polarization to each label, asserting the sector is really constant.
    """
    labels = polarization_sector_labels(cluster)
    states = np.asarray(cluster.ice_states)
    # The bordered solver pairs `labels` (in cluster.ice_states order) with a
    # np.sort'ed copy of the states.  That is only safe if the states are
    # already sorted; fail loudly rather than silently mislabel sectors.
    if not np.all(np.asarray(states, dtype=np.uint64)[:-1]
                  <= np.asarray(states, dtype=np.uint64)[1:]):
        raise SystemExit("cluster.ice_states is NOT sorted; the label/row "
                         "pairing used by the sector solver would be wrong")
    n_sites = int(cluster.n_sites)
    bits = np.array([[(int(s) >> site) & 1 for site in range(n_sites)]
                     for s in states])
    pol = (bits - 0.5) @ np.asarray(cluster.positions, dtype=float)
    n_sec = int(labels.max()) + 1
    T = np.zeros((n_sec, pol.shape[1]))
    for lab in range(n_sec):
        rows = pol[labels == lab]
        if len(rows) > 1 and np.ptp(rows, axis=0).max() > 1e-9:
            raise SystemExit("sector %d has non-constant polarization" % lab)
        T[lab] = 2.0 * rows[0]
    return labels, T


def sector_grounds(cluster, labels, levels, jpm):
    space = bfs_space(cluster, levels)
    ice = np.sort(np.asarray(cluster.ice_states, dtype=np.uint64))
    ice_rows = np.searchsorted(space, ice)
    non_ice = np.setdiff1d(np.arange(len(space)), ice_rows)
    ham = build_hamiltonian(cluster, space, jpm).tocsr()
    out = {}
    for sec in np.unique(labels):
        rows = np.concatenate([ice_rows[labels == sec], non_ice])
        block = ham[np.ix_(rows, rows)]
        vals = eigsh(block, k=2, which="SA", return_eigenvectors=False,
                     tol=1e-10, ncv=min(60, block.shape[0] - 1))
        out[int(sec)] = float(np.min(vals))
    return out, len(space)


def report(tag, T2, dims, grounds):
    """Shell table + isotropy + quadraticity for one (cluster, level, jpm)."""
    n = len(T2)
    E = np.array([grounds[i] for i in range(n)])
    dE = E - E.min()
    bottom = int(np.argmin(E))
    print("\n  bottom sector %d: |T|^2 = %.4f, dim %d, degeneracy %d"
          % (bottom, T2[bottom], dims[bottom], int(np.sum(dE < 1e-10))))
    if T2[bottom] > 1e-9:
        print("  *** GROUND SECTOR IS NOT T=0 -- Eq.(law) premise fails here ***")

    print("  energy shells:")
    print("    mult  dim        dE            |T|^2 in group")
    for v in sorted(set(np.round(dE, 11))):
        sel = np.abs(dE - v) < 1e-11
        t2s = sorted(set(np.round(T2[sel], 4)))
        print("    %4d  %4d  %13.9f    %s"
              % (sel.sum(), int(dims[sel][0]), v,
                 ", ".join("%.4f" % t for t in t2s)))

    print("  isotropy (equal |T|^2 must give equal dE):")
    worst = 0.0
    groups = []
    for v in sorted(set(np.round(T2, 6))):
        sel = np.abs(T2 - v) < 1e-6
        spread = float(dE[sel].max() - dE[sel].min())
        worst = max(worst, spread)
        groups.append((v, int(sel.sum()), float(dE[sel].mean()), spread))
        print("    |T|^2 %8.4f  mult %3d  dE %13.9f  spread %.2e%s"
              % (v, sel.sum(), dE[sel].mean(), spread,
                 "   <-- VIOLATION" if spread > 1e-9 else ""))
    print("    worst spread: %.3e" % worst)

    nz = [g for g in groups if g[0] > 1e-9]
    nz.sort(key=lambda g: g[0])
    fit = {}
    if nz:
        x = np.array([g[0] for g in nz])
        y = np.array([g[2] for g in nz])
        for name, k in (("weak-field(3)", min(3, len(nz))),
                        ("all", len(nz))):
            s = float(np.dot(x[:k], y[:k]) / np.dot(x[:k], x[:k]))
            rel = float(np.abs(y[:k] - s * x[:k]).max() / max(y[:k].max(), 1e-300))
            fit[name] = {"slope": s, "max_rel_resid": rel}
            print("  quadratic fit, %s (%d shells): dE = %.6e |T|^2, "
                  "max rel resid %.3f%s"
                  % (name, k, s, rel,
                     "   <-- NOT QUADRATIC" if rel > 0.1 else ""))
    return {"tag": tag, "bottom_sector": bottom,
            "bottom_T2": float(T2[bottom]),
            "ground_degeneracy": int(np.sum(dE < 1e-10)),
            "worst_isotropy_spread": worst,
            "shells": [{"T2": g[0], "mult": g[1], "dE": g[2], "spread": g[3]}
                       for g in groups],
            "fits": fit,
            "sector_grounds": {str(i): float(E[i]) for i in range(n)}}


def verify_fcc32(record):
    """Re-derive the FCC-32 shell table from the cached JSONs and join |T|^2.

    This is the pending item 1 of the paper README, and it also checks the
    hand-read table that is currently printed in the draft.
    """
    cl = geometry.build_cluster("fcc", (2, 2, 2))
    labels, T = transport_vectors(cl)
    T2 = np.sum(T ** 2, axis=1)
    dims = np.array([int((labels == i).sum()) for i in range(len(T))])
    print("\n" + "=" * 70)
    print("FCC-32 join: %d sites, %d ice states, %d sectors, %d distinct |T|^2"
          % (cl.n_sites, len(cl.ice_states), len(T), len(set(np.round(T2, 6)))))
    print("=" * 70)
    src = ROOT / "campaign" / "outputs" / "fcc32_sector_grounds"
    for path in sorted(src.glob("grounds_*.json")):
        data = json.load(open(path))
        n = len(data["sector_grounds"])
        if n != len(T):
            print("  %s: %d sectors != %d, skipped" % (path.name, n, len(T)))
            continue
        grounds = {i: data["sector_grounds"][str(i)] for i in range(n)}
        print("\n--- %s  (Jpm=%+0.3f, depth %d, space %d)"
              % (path.name, data["jpm"], data["levels"], data["space"]))
        record.append(report("fcc32:" + path.stem, T2, dims, grounds))


def main():
    results = []
    cl = geometry.build_cluster("cubic", (1, 1, 1))
    labels, T = transport_vectors(cl)
    T2 = np.sum(T ** 2, axis=1)
    dims = np.array([int((labels == i).sum()) for i in range(len(T))])
    print("=" * 70)
    print("cubic-16: %d sites, %d ice states, %d sectors, %d distinct |T|^2"
          % (cl.n_sites, len(cl.ice_states), len(T),
             len(set(np.round(T2, 6)))))
    print("sector dims:", np.bincount(dims)[1:].tolist(), "(count by dim)")
    print("=" * 70)

    first_gap = {}
    for jpm in COUPLINGS:
        for lev in LEVELS:
            t0 = time.perf_counter()
            grounds, space = sector_grounds(cl, labels, lev, jpm)
            print("\n" + "-" * 70)
            print("cubic-16  Jpm=%+0.3f  levels=%d  space=%d  (%.1f s)"
                  % (jpm, lev, space, time.perf_counter() - t0))
            print("-" * 70)
            rec = report("cubic16:jpm%+0.3f:L%d" % (jpm, lev), T2, dims, grounds)
            rec.update({"cluster": "cubic16", "jpm": jpm, "levels": lev,
                        "space": space})
            results.append(rec)
            # Gap of the smallest NONZERO flux measured FROM the T=0 sector.
            # (Measuring from the ground is wrong here: on cubic-16 the ground
            # sector is not T=0 at weak coupling.)
            byT2 = {s["T2"]: s["dE"] for s in rec["shells"]}
            zero = byT2.get(0.0, 0.0)
            nz = sorted(t for t in byT2 if t > 1e-9)
            first_gap[(jpm, lev)] = (byT2[nz[0]] - zero) if nz else float("nan")

    print("\n" + "=" * 70)
    print("DEPTH TRUNCATION OF THE FIRST SHELL GAP (the systematic that")
    print("controls eps_eff; levels=4 is EXACT on cubic-16)")
    print("=" * 70)
    print("  Jpm      L=1          L=2          L=3          L=4(exact)"
          "    L=1 error")
    for jpm in COUPLINGS:
        row = [first_gap[(jpm, l)] for l in LEVELS]
        err = (row[0] - row[-1]) / row[-1] * 100 if row[-1] else float("nan")
        print("  %+0.3f  %11.8f  %11.8f  %11.8f  %11.8f   %+7.1f%%"
              % (jpm, row[0], row[1], row[2], row[3], err))

    verify_fcc32(results)

    with open(OUT, "w") as fh:
        json.dump(results, fh, indent=1, default=float)
    print("\nwrote %s" % OUT)


if __name__ == "__main__":
    main()
