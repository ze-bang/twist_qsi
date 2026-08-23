"""Certify the ground winding sector, as the draft demands.

At 48 and 72 sites every winding sector can be enumerated and its
lowest energy computed EXHAUSTIVELY, which converts 'ground state in
the largest sector' from a pattern into a checked fact and gives the
margin to the best competing sector.  At 96 sites exhaustion is out of
reach (401 sectors seen in sampling, sector matrices up to 2.7e9
nonzeros), so the twelve largest candidates are verified directly and
the exhaustive small-size result bounds the plausibility of the rest:
at the exhaustive sizes, sector ground energies fall monotonically
with sector dimension, with every sub-leading sector at least O(0.5K)
above the ground sector.

Usage: sector_certification.py n1 n2 n3 [n_candidates]
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

import recompute_finite_size_artifact as geometry               # noqa: E402
from run_ring48_dssf import contractible                        # noqa: E402
from ice_enumerate_fast import enumerate_ice_branched           # noqa: E402
from ring96_pipeline import (Table128, winding_keys, hex_words,  # noqa: E402
                             ring_matrix)

OUT = ROOT / "campaign" / "outputs" / "ring_model_dssf"
U64 = np.uint64


def log(m, t0):
    print(f"[{time.perf_counter()-t0:8.1f}s] {m}", flush=True)


def main() -> int:
    t0 = time.perf_counter()
    shape = tuple(int(x) for x in sys.argv[1:4])
    ncand = int(sys.argv[4]) if len(sys.argv) >= 5 else 0   # 0 = all

    orig = geometry.enumerate_ice_states
    geometry.enumerate_ice_states = lambda a, b: np.array([0], dtype=U64)
    cl = geometry.build_cluster("fcc", shape)
    geometry.enumerate_ice_states = orig
    n = int(cl.n_sites)
    positions = np.asarray(cl.positions, dtype=float)
    tets = [tuple(int(s) for s in t) for t in cl.tets]
    hexes = contractible(cl)
    states = enumerate_ice_branched(n, tets, log=lambda m: None)
    ipos = np.rint(positions * 4).astype(np.int64)
    keys = winding_keys(states, ipos)
    uk, cnt = np.unique(keys, return_counts=True)
    order = np.argsort(cnt)[::-1]
    if ncand:
        order = order[:ncand]
    log(f"fcc{shape}: {len(states):,} states in {len(uk)} sectors; "
        f"checking {len(order)}", t0)

    hw = hex_words(hexes)
    rows_out = []
    for rank, oi in enumerate(order):
        c, dim = int(uk[oi]), int(cnt[oi])
        rows = np.where(keys == c)[0]
        if dim == 1:
            e = 0.0            # a frozen single state has E = 0
        else:
            tab = Table128(states[rows])
            H = ring_matrix(tab, hw, t0) if dim > 1 else None
            if H.nnz == 0:
                e = 0.0
            else:
                k = 1
                e = float(eigsh(H, k=k, which="SA", tol=1e-9,
                                ncv=min(dim - 1, 40),
                                return_eigenvectors=False)[0]) \
                    if dim > 2 else float(
                        np.linalg.eigvalsh(H.toarray())[0])
            del tab, H
        rows_out.append({"key": c, "dim": dim, "E0": e})
        if rank < 15 or rank % 25 == 0:
            log(f"  #{rank:3d} sector {c:>12d}  dim {dim:>9,}  "
                f"E0 = {e:12.6f}", t0)
    rows_out.sort(key=lambda r: r["E0"])
    log("\nlowest five sectors by energy:", t0)
    for r in rows_out[:5]:
        log(f"  key {r['key']:>12d}  dim {r['dim']:>9,}  "
            f"E0 = {r['E0']:12.6f}", t0)
    gap = rows_out[1]["E0"] - rows_out[0]["E0"] if len(rows_out) > 1 else 0
    log(f"margin to the best competing sector: {gap:.6f} K", t0)
    tag = "".join(str(s) for s in shape)
    with open(OUT / f"sector_cert_fcc{tag}.json", "w") as fh:
        json.dump({"shape": list(shape), "checked": len(order),
                   "total_sectors": int(len(uk)),
                   "margin": gap, "sectors": rows_out}, fh, indent=1)
    log(f"wrote sector_cert_fcc{tag}.json", t0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
