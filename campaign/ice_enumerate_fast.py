"""Vectorized ice enumeration, and a geometry probe for large clusters.

The depth-first enumerator in recompute_finite_size_artifact is pure
Python: it took 11 minutes for the 12.8 million configurations of the
72-site cluster, so the ~2 billion expected at 96 sites would take
close to a day.  It also returns Python ints, which the 128-bit
pipeline then has to split one at a time.

This builds the manifold tetrahedron by tetrahedron instead, entirely
in numpy.  Carrying a frontier of partial configurations, each
tetrahedron is imposed by grouping the frontier on how many of its
ALREADY-assigned sites point up, and expanding each group by exactly
the assignments of its new sites that bring the total to two.  Nothing
is ever rejected after the fact, so no work is wasted, and every step
is a handful of vectorized ORs and concatenations.

Tetrahedra are visited breadth-first over shared sites, which keeps
the number of new sites per step near two and the frontier as small as
the manifold allows.

States are (lo, hi) uint64 pairs throughout, so there is no 64-site
cap.  Validated against the known counts at 48 and 72 sites.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "notes"))
sys.path.insert(0, str(ROOT / "campaign"))

U64 = np.uint64
KNOWN = {48: 87_546, 64: 2_756_394, 72: 12_846_186}


def bit_of(states, site):
    """0/1 occupation of `site` for every state, as int8."""
    col = 0 if site < 64 else 1
    sh = U64(site if site < 64 else site - 64)
    return ((states[:, col] >> sh) & U64(1)).astype(np.int8)


def site_mask(sites):
    """(2,) uint64 mask with the given sites set."""
    lo = hi = 0
    for s in sites:
        if s < 64:
            lo |= (1 << s)
        else:
            hi |= (1 << (s - 64))
    return np.array([lo, hi], dtype=U64)


def bfs_order(tets):
    """Visit tetrahedra breadth-first over shared sites."""
    site_tets = {}
    for ti, t in enumerate(tets):
        for s in t:
            site_tets.setdefault(s, []).append(ti)
    seen = {0}
    order = [0]
    queue = [0]
    while queue:
        ti = queue.pop(0)
        for s in tets[ti]:
            for tj in site_tets[s]:
                if tj not in seen:
                    seen.add(tj)
                    order.append(tj)
                    queue.append(tj)
    for ti in range(len(tets)):          # any disconnected remainder
        if ti not in seen:
            order.append(ti)
    return order


def subsets_with_popcount(sites, k):
    """All masks over `sites` with exactly k bits set."""
    from itertools import combinations
    return [site_mask(c) for c in combinations(sites, k)]


def enumerate_ice_fast(n_sites, tets, budget_gb=400.0, log=print):
    order = bfs_order(tets)
    states = np.zeros((1, 2), dtype=U64)
    assigned = set()
    t0 = time.perf_counter()
    for step, ti in enumerate(order):
        S = [int(x) for x in tets[ti]]
        K = [s for s in S if s in assigned]
        N = [s for s in S if s not in assigned]
        if K:
            upsK = np.zeros(len(states), dtype=np.int8)
            for s in K:
                upsK += bit_of(states, s)
        else:
            upsK = np.zeros(len(states), dtype=np.int8)

        pieces = []
        for k in range(0, len(K) + 1):
            need = 2 - k
            if need < 0 or need > len(N):
                continue
            sel = np.where(upsK == k)[0]
            if not len(sel):
                continue
            sub = states[sel]
            for m in subsets_with_popcount(N, need):
                new = sub.copy()
                new[:, 0] |= m[0]
                new[:, 1] |= m[1]
                pieces.append(new)
        del upsK
        if not pieces:
            log("  frontier empty -- no ice states")
            return np.zeros((0, 2), dtype=U64)
        states = np.concatenate(pieces)
        del pieces
        assigned.update(N)
        gb = states.nbytes / 2**30
        if gb > budget_gb:
            raise MemoryError(f"frontier {gb:.1f} GB exceeds "
                              f"budget {budget_gb} GB at step {step}")
        if step % 6 == 0 or step == len(order) - 1:
            log(f"  tet {step+1:3d}/{len(order)}  "
                f"{len(assigned):3d} sites assigned  "
                f"frontier {len(states):>14,}  {gb:7.2f} GB  "
                f"({time.perf_counter()-t0:6.1f} s)")
    return states


def main() -> int:
    import recompute_finite_size_artifact as geometry
    from run_ring48_dssf import contractible
    from run_momentum_resolved_spectrum import star
    from ring48_momenta_fix import fcc_momenta

    t0 = time.perf_counter()
    shapes = [(2, 2, 3), (2, 3, 3), (2, 3, 4)]
    orig = geometry.enumerate_ice_states
    for shape in shapes:
        geometry.enumerate_ice_states = lambda n, t: np.array([0], dtype=U64)
        cl = geometry.build_cluster("fcc", shape)
        geometry.enumerate_ice_states = orig
        n = int(cl.n_sites)
        tets = [tuple(int(s) for s in t) for t in cl.tets]
        hexes = contractible(cl)
        momenta, _ = fcc_momenta(cl)
        nL = sum(1 for q in momenta if star(q) == "L")
        print(f"\n=== fcc{shape}: {n} sites ===", flush=True)
        print(f"  tetrahedra {len(tets)} (need {n//2}: "
              f"{'OK' if len(tets)==n//2 else 'BAD'})")
        print(f"  contractible hexagons {len(hexes)} (need {n}: "
              f"{'OK' if len(hexes)==n else 'BAD'})")
        print(f"  momenta {len(momenta)}, of which L-type: {nL}")
        if n > 80:
            print("  (enumeration deferred to the production run)")
            continue
        st = enumerate_ice_fast(n, tets, log=print)
        got = len(st)
        ref = KNOWN.get(n)
        ok = "OK" if ref is None or got == ref else f"MISMATCH (want {ref})"
        print(f"  enumerated {got:,} ice configurations -- {ok}")
        # spot-check the ice rule directly on a random sample
        rng = np.random.default_rng(0)
        idx = rng.choice(got, size=min(2000, got), replace=False)
        bad = 0
        for t in tets:
            ups = np.zeros(len(idx), dtype=np.int8)
            for s in t:
                ups += bit_of(st[idx], int(s))
            bad += int((ups != 2).sum())
        print(f"  ice-rule check on {len(idx)} sampled states: "
              f"{bad} violations")
    print(f"\ntotal {time.perf_counter()-t0:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
