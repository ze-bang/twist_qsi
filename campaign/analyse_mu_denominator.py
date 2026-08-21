"""Is mu an intensive renormalized denominator?  Two-cluster moment test.

The spectral run (analyse_mu_anatomy.py) found that mu looks like a renormalized
virtual denominator rather than destructive interference, but it measured that
denominator as (E_bar_abs - z0), which contains the EXTENSIVE ground energy z0.
mu itself is intensive (cubic-16 and FCC-32 agree to 5-8%), so those two facts
cannot both be the whole story.  This settles it on both clusters.

Method, and why it works where diagonalization does not.  With
v_a = QHP|a> (a sparse column of H) the amplitude and its moments are

    M_n = <b| PHQ (QHQ)^n QHP |a> = v_b . (H_QQ)^n v_a
    K   = <b| PHQ (z0 - QHQ)^-1 QHP |a> = - v_b . (H_QQ - z0)^-1 v_a

M_0 vanishes identically for a hexagon pair -- PHQ.QHP applies H twice and moves
at most four spins, while a hexagon pair differs by six.  That is the selection
rule that made the first run's D_eff/cancel/E_bar columns meaningless.  Given
M_0 = 0, expanding 1/(E - z0) about the weight centroid E* gives

    K = M_1 / (E* - z0)^2      ->      D_eff = sqrt(M_1 / K),

a denominator built from two computable numbers with no eigen-decomposition:
one sparse matvec and one CG solve.  (H_QQ - z0) is positive definite whenever
z0 lies below the Q spectrum, which is the regime of interest.

Reported per cluster and coupling:
    M_0        must be ~0 (selection-rule check on the whole construction)
    D_eff      the effective virtual denominator
    D_eff^2    compare with 1/mu: the hexagon carries two denominators
    D_eff / D_eff(weakest coupling), against 1/mu normalised the same way

VERDICT: if D_eff at matched coupling is the same on 16 and 32 sites, the
renormalization is intensive and mu is a genuine dressed virtual gap.  If it
roughly doubles, the cubic-16 agreement was a coincidence of that cluster.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse.linalg as sla
from scipy.sparse.linalg import eigsh

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "notes"))
sys.path.insert(0, str(ROOT / "campaign"))

import recompute_finite_size_artifact as geometry              # noqa: E402
from qsi_campaign.protocol import polarization_sector_labels   # noqa: E402
from run_truncated_feshbach import bfs_space, build_hamiltonian  # noqa: E402

OUT = ROOT / "campaign" / "outputs" / "map_normalization"
OUT.mkdir(parents=True, exist_ok=True)

JOBS = [("cubic16", ("cubic", (1, 1, 1)), [4]),
        ("fcc32", ("fcc", (2, 2, 2)), [1, 2])]
COUPLINGS = [-0.03, -0.06, -0.10, -0.15, -0.20, -0.25, -0.30]
HEAVY_COUPLINGS = [-0.06, -0.15, -0.30]      # levels=2 subset on FCC-32


def contractible_hexagon_pair(cluster, labels):
    """One same-sector ice pair related by flipping one contractible hexagon."""
    ice = np.asarray(cluster.ice_states, dtype=np.uint64)
    lookup = {int(s): i for i, s in enumerate(ice)}
    for entry in cluster.hexes:
        path, winding = entry[0], entry[1]
        if tuple(int(w) for w in winding) != (0, 0, 0):
            continue                       # wrapping: the torus artifact
        mask = 0
        for site in path:
            mask |= 1 << int(site)
        for i, state in enumerate(ice):
            partner = int(state) ^ mask
            j = lookup.get(partner)
            if j is not None and labels[i] == labels[j]:
                return i, j, mask
    raise SystemExit("no contractible flippable hexagon found")


def cg_solve(matrix, rhs, shift, tol=1e-10):
    """Solve (H_QQ - shift) x = rhs by CG; positive definite for shift < E_min."""
    n = matrix.shape[0]
    operator = sla.LinearOperator(
        (n, n), matvec=lambda v: matrix @ v - shift * v, dtype=float)
    try:
        x, info = sla.cg(operator, rhs, rtol=tol, atol=0.0, maxiter=20000)
    except TypeError:                       # older scipy
        x, info = sla.cg(operator, rhs, tol=tol, atol=0.0, maxiter=20000)
    if info != 0:
        print("    WARNING: CG returned info=%d" % info, flush=True)
    return x


def run(name, basis_shape, levels_list, records):
    cluster = geometry.build_cluster(*basis_shape)
    labels = polarization_sector_labels(cluster)
    i, j, mask = contractible_hexagon_pair(cluster, labels)
    print("\n%s: %d sites, %d ice states; hexagon pair (%d, %d), mask 0x%x"
          % (name, cluster.n_sites, len(cluster.ice_states), i, j, mask),
          flush=True)

    for levels in levels_list:
        space = bfs_space(cluster, levels)
        ice = np.sort(np.asarray(cluster.ice_states, dtype=np.uint64))
        ice_rows = np.searchsorted(space, ice)
        non_ice = np.setdiff1d(np.arange(len(space)), ice_rows)
        # ice_rows follows the sorted states; labels follow cluster.ice_states,
        # which is already sorted (checked in cubic16_sector_check.py)
        row_a, row_b = ice_rows[i], ice_rows[j]
        couplings = (COUPLINGS if levels == 1 or name == "cubic16"
                     else HEAVY_COUPLINGS)
        print("  levels=%d  space=%d  Q=%d" % (levels, len(space), len(non_ice)),
              flush=True)

        for jpm in couplings:
            t0 = time.perf_counter()
            ham = build_hamiltonian(cluster, space, jpm).tocsr()
            z0 = float(eigsh(ham, k=1, which="SA", tol=1e-11, ncv=80,
                             return_eigenvectors=False)[0])
            hqq = ham[np.ix_(non_ice, non_ice)].tocsr()
            v_a = np.asarray(ham[non_ice, row_a].todense()).ravel()
            v_b = np.asarray(ham[non_ice, row_b].todense()).ravel()

            m0 = float(v_b @ v_a)
            hv = hqq @ v_a
            m1 = float(v_b @ hv)
            m2 = float(v_b @ (hqq @ hv))
            x = cg_solve(hqq, v_a, z0)
            amp = -float(v_b @ x)
            d_eff = float(np.sqrt(m1 / amp)) if amp and m1 / amp > 0 else np.nan

            rec = {"cluster": name, "levels": levels, "jpm": jpm, "z0": z0,
                   "M0": m0, "M1": m1, "M2": m2, "K": amp, "D_eff": d_eff,
                   "space": len(space),
                   "runtime": time.perf_counter() - t0}
            records.append(rec)
            print("    jpm=%+0.3f z0=%+.5f  M0=%+.2e  M1=%+.4e  K=%+.4e"
                  "  D_eff=%.4f  D^2=%.4f  (%.0f s)"
                  % (jpm, z0, m0, m1, amp, d_eff, d_eff ** 2,
                     rec["runtime"]), flush=True)
            with open(OUT / "mu_denominator.json", "w") as fh:
                json.dump(records, fh, indent=1, default=float)
    return records


def main():
    allrecs = []
    for name, shape, levels in JOBS:
        run(name, shape, levels, allrecs)

    print("\n" + "=" * 78)
    print("INTENSIVITY VERDICT: D_eff at matched coupling, 16 vs 32 sites")
    print("=" * 78)
    print("  jpm      cubic16(exact)   fcc32(L1)   fcc32(L2)   ratio 32/16 (L1)")
    for jpm in COUPLINGS:
        def get(cl, lv):
            for r in allrecs:
                if (r["cluster"] == cl and r["levels"] == lv
                        and abs(r["jpm"] - jpm) < 1e-9):
                    return r["D_eff"]
            return None
        c, f1, f2 = get("cubic16", 4), get("fcc32", 1), get("fcc32", 2)
        print("  %+0.3f   %s   %s   %s   %s"
              % (jpm,
                 "%12.4f" % c if c else "      --    ",
                 "%9.4f" % f1 if f1 else "    --   ",
                 "%9.4f" % f2 if f2 else "    --   ",
                 "%.3f" % (f1 / c) if (c and f1) else " -- "))
    print("\n  ~1.0  -> intensive: mu IS a dressed virtual gap.")
    print("  ~2.0  -> extensive: the cubic-16 agreement was a coincidence.")
    print("\n  Compare D_eff^2 (the hexagon carries two denominators) with 1/mu:")
    print("    1/mu = 1.17 (0.03) 1.37 (0.06) 1.69 (0.10) 2.15 (0.15)"
          " 2.64 (0.20) 3.14 (0.25) 3.65 (0.30)")


if __name__ == "__main__":
    main()
