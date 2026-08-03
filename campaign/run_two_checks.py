#!/usr/bin/env python3
"""Two claims that were inferred rather than measured, now measured.

(1) Is the residue Z extensive, so that a larger cluster makes it worse?
    The ice weight of the ground state is computed on cubic-16 and on FCC-32 at
    the same coupling and the same spinon truncation.  If the pair content is
    extensive, <n_pairs> ~ 2N(J_pm/J_zz)^2, then the probability of finding no
    pair at all falls exponentially with N and Z_32 should be near Z_16^2.

(2) Does the spinon sector carry a winding artifact of the same kind as the ice
    manifold?  The diagnostic is sensitivity to a twist: a sector whose
    dynamics is dominated by paths that wrap the box changes a lot when the
    boundary condition is twisted, one whose finite-size error is ordinary
    momentum discretisation changes little.  We twist the boundary condition
    and watch (a) the bottom of the spinon spectrum, min spec(QHQ), and
    (b) the ice-block bandwidth, whose twist sensitivity is the known g_4
    artifact and therefore sets the scale of what "dominated by winding" looks
    like.

Usage: run_two_checks.py <jpm> [<jpm> ...]
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.sparse.linalg import eigsh

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "notes"))
sys.path.insert(0, str(ROOT / "campaign"))

import recompute_finite_size_artifact as geometry  # noqa: E402
from qsi_campaign.exact_band import (  # noqa: E402
    fixed_magnetization_basis,
    microscopic_character_hamiltonian,
)
from run_truncated_feshbach import bfs_space, build_hamiltonian  # noqa: E402

OUTPUT = ROOT / "campaign" / "outputs" / "two_checks"
OUTPUT.mkdir(parents=True, exist_ok=True)


def ice_weight(cluster, jpm, levels):
    """Ground-state weight on the ice manifold, in a truncated space."""
    space = bfs_space(cluster, levels)
    hamiltonian = build_hamiltonian(cluster, space, jpm).tocsr()
    values, vectors = eigsh(hamiltonian, k=1, which="SA", tol=1e-9)
    ground = vectors[:, 0]
    ice = np.sort(np.asarray(cluster.ice_states, dtype=np.uint64))
    rows = np.searchsorted(space, ice)
    weight = float(np.sum(np.abs(ground[rows]) ** 2))
    ups = np.bitwise_count(space[:, None] & cluster.tet_masks[None, :])
    defects = (ups != 2).sum(axis=1)
    pairs = float(np.sum(defects * np.abs(ground) ** 2)) / 2.0
    return {"space": int(len(space)), "ice_weight": weight,
            "pairs": pairs, "energy": float(values[0])}


def twist_sensitivity(cluster, jpm, twists):
    """How much the two sectors move when the boundary condition is twisted."""
    states = fixed_magnetization_basis(cluster.n_sites, cluster.n_sites // 2)
    index = {int(s): i for i, s in enumerate(states)}
    ice_rows = np.asarray([index[int(s)] for s in cluster.ice_states])
    keep = np.ones(len(states), dtype=bool)
    keep[ice_rows] = False
    complement = np.where(keep)[0]

    spinon_floor, ice_width, ground = [], [], []
    for theta in twists:
        # the sourced Hamiltonian is complex for a general twist; keeping only
        # its real part would diagonalize a different operator
        hamiltonian = microscopic_character_hamiltonian(
            cluster, states, jpm, np.asarray(theta)).tocsc()
        block = hamiltonian[complement][:, complement]
        spinon_floor.append(float(eigsh(block, k=1, which="SA", tol=1e-9,
                                        return_eigenvectors=False)[0]))
        low = np.sort(eigsh(hamiltonian, k=90, which="SA", tol=1e-9,
                            return_eigenvectors=False).real)
        ground.append(float(low[0]))
        ice_width.append(float(low[-1] - low[0]))
    return (np.asarray(spinon_floor), np.asarray(ice_width),
            np.asarray(ground))


def main() -> None:
    couplings = [float(v) for v in sys.argv[1:]
                 if not v.startswith("--")]
    cubic = geometry.build_cluster("cubic", (1, 1, 1))
    fcc = geometry.build_cluster("fcc", (2, 2, 2))
    twists = [(0.0, 0.0, 0.0), (np.pi / 2, 0.0, 0.0), (np.pi, 0.0, 0.0),
              (np.pi / 2, np.pi / 2, 0.0), (np.pi / 2, np.pi / 2, np.pi / 2)]

    for jpm in couplings:
        started = time.perf_counter()
        record = {"jpm": jpm}

        # ---- check 1: extensivity of Z
        for name, cluster in ((("cubic16", cubic), ("fcc32", fcc))
                              if "--twist-only" not in sys.argv else ()):
            for levels in (1, 2):
                try:
                    result = ice_weight(cluster, jpm, levels)
                    record[f"{name}_L{levels}"] = result
                    print(f"  {name} L{levels}: space={result['space']} "
                          f"ice weight={result['ice_weight']:.4f} "
                          f"pairs={result['pairs']:.3f}", flush=True)
                except Exception as error:
                    record[f"{name}_L{levels}"] = {"error": str(error)[:100]}
                    print(f"  {name} L{levels}: {str(error)[:80]}", flush=True)

        # ---- check 2: twist sensitivity of the two sectors
        floor, width, ground = twist_sensitivity(cubic, jpm, twists)
        # normalise each sector by its own energy scale: the spinon floor by
        # the spinon gap it sits at, the ice band by its own bandwidth
        gap = float(np.mean(floor - ground))
        record["spinon_floor"] = floor.tolist()
        record["ice_width"] = width.tolist()
        record["ground"] = ground.tolist()
        record["spinon_gap"] = gap
        record["spinon_twist_spread"] = float(np.ptp(floor - ground))
        record["ice_twist_spread"] = float(np.ptp(width))
        record["spinon_relative"] = float(np.ptp(floor - ground) / abs(gap))
        record["ice_relative"] = float(np.ptp(width) / float(np.mean(width)))
        record["runtime"] = time.perf_counter() - started

        tag = f"{jpm:+.3f}".replace(".", "p")
        with open(OUTPUT / f"checks_{tag}.json", "w") as handle:
            json.dump(record, handle, indent=1, default=float)
        print(f"jpm={jpm:+.3f} spinon floor spread={record['spinon_twist_spread']:.5f} "
              f"({record['spinon_relative']*100:.2f}% of its value) | "
              f"ice bandwidth spread={record['ice_twist_spread']:.5f} "
              f"({record['ice_relative']*100:.1f}% of its value) "
              f"({record['runtime']:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
