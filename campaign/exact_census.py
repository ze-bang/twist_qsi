"""Exact flippability census of the lowest excitations, no Lanczos.

The Ritz-vector census reconstructs an eigenvector from the Krylov
basis.  Eigenvalues and spectral weights converge quickly in a Lanczos
run; EIGENVECTORS converge more slowly, and the census is an
expectation value in the eigenvector, so it is the quantity most at
risk of being under-converged.

This computes it the other way, with no continued fraction at all:
take the lowest few eigenpairs of the sector directly, and evaluate
<n| sum_p n_p^flip |n> - <0| ... |0> exactly.  The probe weight of each
level at the L point is computed too, so the anomalous level can be
identified without reference to the Lanczos run.

Run on 48 and 64 sites, where the census is already known, this is a
gate; run on 72 it decides whether the smaller census seen there is
physics or a convergence artifact.
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
from run_momentum_resolved_spectrum import star                 # noqa: E402
from run_ring48_dssf import contractible                        # noqa: E402
from ring48_momenta_fix import fcc_momenta                      # noqa: E402
from ice_enumerate_fast import enumerate_ice_fast, bit_of       # noqa: E402
from ring96_pipeline import (Table128, winding_keys, hex_words,  # noqa: E402
                             ring_matrix, nflip_counts)

OUT = ROOT / "campaign" / "outputs" / "ring_model_dssf"
U64 = np.uint64
CHUNK = 1 << 21


def log(m, t0):
    print(f"[{time.perf_counter()-t0:7.1f}s] {m}", flush=True)


def main() -> int:
    t0 = time.perf_counter()
    shape = tuple(int(x) for x in sys.argv[1:4])
    kmax = int(sys.argv[4]) if len(sys.argv) >= 5 else 8

    orig = geometry.enumerate_ice_states
    geometry.enumerate_ice_states = lambda a, b: np.array([0], dtype=U64)
    cl = geometry.build_cluster("fcc", shape)
    geometry.enumerate_ice_states = orig
    n = int(cl.n_sites)
    positions = np.asarray(cl.positions, dtype=float)
    tets = [tuple(int(s) for s in t) for t in cl.tets]
    hexes = contractible(cl)
    log(f"fcc{shape}: {n} sites, {len(hexes)} hexagons", t0)

    states = enumerate_ice_fast(n, tets, log=lambda m: None)
    log(f"enumerated {len(states):,} ice configurations", t0)
    ipos = np.rint(positions * 4).astype(np.int64)
    keys = winding_keys(states, ipos)
    uk, cnt = np.unique(keys, return_counts=True)
    hw = hex_words(hexes)
    best = None
    for c in uk[np.argsort(cnt)[::-1][:3]]:
        rows = np.where(keys == c)[0]
        tab = Table128(states[rows])
        H = ring_matrix(tab, hw, t0)
        e = float(eigsh(H, k=1, which="SA", tol=1e-10, ncv=30,
                        return_eigenvectors=False)[0])
        if best is None or e < best[0] - 1e-9:
            best = (e, tab, H)
    _, tab, H = best
    del states, keys
    log(f"ground sector dim {len(tab):,}, nnz {H.nnz:,}", t0)

    vals, vecs = eigsh(H, k=kmax, which="SA", tol=1e-12, ncv=max(60, 4*kmax))
    order = np.argsort(vals)
    vals, vecs = vals[order], vecs[:, order]
    e0 = float(vals[0])
    gs = vecs[:, 0]
    nf = nflip_counts(tab.w, hw)
    nf_gs = float(np.sum(gs ** 2 * nf))
    log(f"E0 = {e0:.8f} K,  <n_flip> ground = {nf_gs:.4f} "
        f"of {len(hexes)} hexagons", t0)

    # probe weight at each L momentum, to identify the anomalous level
    subl = np.arange(n) % 4
    Lq = [q for q in fcc_momenta(cl)[0] if star(q) == "L"]
    probes = []
    for q in Lq:
        phase = np.exp(-2j * np.pi * (positions @ np.asarray(q)))
        cols = []
        for mu in range(4):
            sites = np.where(subl == mu)[0]
            col = np.zeros(len(tab.w), dtype=complex)
            for a in range(0, len(tab.w), CHUNK):
                b = min(a + CHUNK, len(tab.w))
                acc = np.zeros(b - a, dtype=complex)
                for s in sites:
                    acc += (bit_of(tab.w[a:b], int(s)) - 0.5) * phase[s]
                col[a:b] = acc * gs[a:b]
            cols.append(col)
        probes.append((q, cols))

    log("\n  level   omega/K    census d(omega)/dV    "
        + "   ".join(f"w(L{i})" for i in range(len(Lq))), t0)
    rows_out = []
    for j in range(kmax):
        v = vecs[:, j]
        cen = float(np.sum(v ** 2 * nf)) - nf_gs
        ws = []
        for q, cols in probes:
            w = sum(abs(np.vdot(v, c)) ** 2 for c in cols) / n
            ws.append(w)
        log(f"   {j:3d}   {vals[j]-e0:8.4f}      {cen:+9.4f}        "
            + "   ".join(f"{w:.4f}" for w in ws), t0)
        rows_out.append({"level": j, "omega": float(vals[j] - e0),
                         "census": cen, "L_weights": [float(x) for x in ws]})

    tag = "".join(str(s) for s in shape)
    with open(OUT / f"exact_census_fcc{tag}.json", "w") as fh:
        json.dump({"shape": list(shape), "n_sites": n, "E0": e0,
                   "nflip_ground": nf_gs, "sector_dim": int(len(tab)),
                   "L_momenta": [[float(x) for x in q] for q in Lq],
                   "levels": rows_out}, fh, indent=1, default=float)
    log(f"wrote exact_census_fcc{tag}.json", t0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
