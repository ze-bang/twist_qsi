#!/usr/bin/env python3
"""Per-sector winding-free ground energies on FCC-32 via bordered matrices.

Each transport sector's masked spectrum is exactly the spectrum of the
Hermitian bordered matrix M_s = [[PHP_ss, PHQ_s], [QHP_s, QHQ]], with Q
truncated to the BFS space of ``levels`` exchange hops.  Sparse Lanczos on
M_s gives the sector ground energy with no z anchor and no self-consistency
loop.  This is the instrument for the 6->2 ground-multiplicity crossing
question (is the cubic-16 crossing at Jpm ~ -0.25 physical?).

Calibration on cubic-16 (scratchpad calibrate_bordered.py, Aug-01): L=1
never sees the crossing; L=2 sees it displaced into (-0.25, -0.30); L=3
within ~0.02 of the exact coupling; L=4 (= the full space there) exact to
all digits.  Read FCC-32 verdicts through that ladder.

Usage: run_fcc32_sector_grounds.py <levels> <jpm> [<jpm> ...]
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
from qsi_campaign.protocol import polarization_sector_labels  # noqa: E402
from run_truncated_feshbach import bfs_space, build_hamiltonian  # noqa: E402

OUTPUT = ROOT / "campaign" / "outputs" / "fcc32_sector_grounds"
OUTPUT.mkdir(parents=True, exist_ok=True)


def main() -> None:
    levels = int(sys.argv[1])
    couplings = [float(v) for v in sys.argv[2:]]
    cluster = geometry.build_cluster("fcc", (2, 2, 2))
    space = bfs_space(cluster, levels)
    labels = polarization_sector_labels(cluster)
    ice_states = np.sort(np.asarray(cluster.ice_states, dtype=np.uint64))
    ice_rows = np.searchsorted(space, ice_states)
    non_ice = np.setdiff1d(np.arange(len(space)), ice_rows)
    sector_ids = np.unique(labels)
    print(f"L={levels}: space {len(space):,}, {len(sector_ids)} sectors",
          flush=True)

    for jpm in couplings:
        started = time.perf_counter()
        hamiltonian = build_hamiltonian(cluster, space, jpm).tocsr()
        grounds, dims = {}, {}
        for sector in sector_ids:
            rows = np.concatenate([ice_rows[labels == sector], non_ice])
            block = hamiltonian[np.ix_(rows, rows)]
            values = eigsh(block, k=2, which="SA",
                           return_eigenvectors=False, tol=1e-10)
            grounds[int(sector)] = float(np.min(values))
            dims[int(sector)] = int((labels == sector).sum())

        energies = np.sort(np.array(list(grounds.values())))
        rel = energies - energies[0]
        bottom_sector = min(grounds, key=grounds.get)
        record = {
            "jpm": jpm, "levels": levels, "space": len(space),
            "sector_grounds": grounds, "sector_dims": dims,
            "bottom_sector": bottom_sector,
            "bottom_sector_dim": dims[bottom_sector],
            "lowest_rel": rel[:10].tolist(),
            "runtime": time.perf_counter() - started,
        }
        tag = f"{jpm:+.3f}".replace(".", "p")
        with open(OUTPUT / f"grounds_L{levels}_{tag}.json", "w") as handle:
            json.dump(record, handle, indent=1, default=float)
        print(f"jpm={jpm:+.3f} L={levels} bottom sector {bottom_sector} "
              f"(dim {dims[bottom_sector]}) | lowest rel: "
              + np.array2string(np.round(rel[:6], 6), separator=",")
              + f" ({record['runtime']:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
