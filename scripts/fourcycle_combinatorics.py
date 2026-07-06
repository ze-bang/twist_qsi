"""Detailed combinatoric characterisation of the ice-rule-preserving 4-site
loops on the (1,1,1) cubic and (2,2,2) FCC pyrochlore clusters.

For each cluster it reports:
  * total number of ice-preserving simple 4-cycles
  * breakdown by winding vector w (and by the unsigned class |w| up to the
    orientation reversal w -> -w that pairs a loop with its h.c.)
  * how many distinct tetrahedron-pairs the 4-cycles connect, and the
    multiplicity (number of 4-cycles per tetrahedron pair)
  * sublattice content of each 4-cycle
  * survival under the Nphi=2 corner projector  K[w]=prod_a (1+(-1)^{w_a})/2
"""
from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np

import cluster_geometry_audit as cg


def analyse(basis: str, shape, sublat_of):
    if basis == "cubic":
        vertices, edges, _tets, bond_wrap, adj = cg.build_graph(*shape)
    else:
        vertices, edges, _tets, bond_wrap, adj = cg.build_graph_fcc(*shape)
    # Use the basis-independent 4-clique enumeration so that BOTH up- and
    # down-tetrahedra enter the ice-rule test (the cubic generator returns
    # only the up-tetrahedra).
    tets = cg._enumerate_4_cliques(adj, len(vertices))
    site_tets = cg.site_to_tetrahedra(tets)

    cycles4 = cg.enumerate_simple_cycles(adj, 4)
    phys = [
        (sites, w)
        for sites, w in cycles4
        if cg.is_ice_preserving(sites, site_tets, tets)
    ]

    print(f"\n==== {basis} {shape} : N={len(vertices)} sites, "
          f"{len(edges)} bonds, {len(tets)} tetrahedra ====")
    print(f"total ice-preserving simple 4-cycles : {len(phys)}")

    # winding classes (unsigned: a cycle and its reverse have w and -w)
    def canon(w):
        w = tuple(int(x) for x in w)
        return min(w, tuple(-x for x in w))

    by_w = Counter(canon(w) for _, w in phys)
    contract = sum(1 for _, w in phys if tuple(w) == (0, 0, 0))
    print(f"contractible (w=0)                   : {contract}")
    print("winding classes (|w| up to w->-w)    :")
    for w, n in sorted(by_w.items(), key=lambda kv: (sum(abs(c) for c in kv[0]), kv[0])):
        absw = tuple(sorted(abs(c) for c in w))
        print(f"   w~{w}  |w|-pattern {absw}  multiplicity {n}")

    # how many tetrahedra each 4-cycle threads (consecutive sites share a tet)
    ntet_per_cycle = Counter()
    for sites, w in phys:
        s = list(sites)
        threaded = set()
        for i in range(4):
            a, b = s[i], s[(i + 1) % 4]
            common = set(site_tets[a]) & set(site_tets[b])
            threaded.update(common)
        ntet_per_cycle[len(threaded)] += 1
    for nt, c in sorted(ntet_per_cycle.items()):
        print(f"4-cycles threading exactly {nt} tetrahedra : {c}")

    # sublattice content
    sub_patterns = Counter()
    for sites, w in phys:
        subs = tuple(sorted(sublat_of[s] for s in sites))
        sub_patterns[subs] += 1
    print("sublattice content of the 4 sites    :")
    for subs, c in sorted(sub_patterns.items()):
        print(f"   sublattices {subs} : {c}")

    # Nphi=2 corner projector survival
    def kernel2(w):
        return int(np.prod([1 + (-1) ** wa for wa in w])) // 8  # 1 if all even else 0
    surv = sum(1 for _, w in phys if kernel2(w))
    print(f"survive Nphi=2 corner projector      : {surv} (killed: {len(phys)-surv})")

    # --- 6-cycles (ice-preserving, [111]-planar hexagons) ---
    pbc_vecs = cg.cluster_pbc_vectors(*shape, basis)
    cycles6 = cg.enumerate_simple_cycles(adj, 6)
    hexs = [
        (sites, w)
        for sites, w in cycles6
        if cg.is_ice_preserving(sites, site_tets, tets)
        and cg.is_111_planar_with_wrap(sites, vertices, bond_wrap, pbc_vecs)
    ]
    print(f"\n  -- ice-preserving [111]-planar 6-cycles (hexagons): {len(hexs)} --")
    by_w6 = Counter(canon(w) for _, w in hexs)
    contract6 = sum(1 for _, w in hexs if tuple(w) == (0, 0, 0))
    print(f"  contractible (w=0)        : {contract6}")
    print(f"  wrapping (w!=0)           : {len(hexs)-contract6}")
    # group by |w|-pattern
    patt6 = Counter()
    for w, n in by_w6.items():
        patt6[tuple(sorted(abs(c) for c in w))] += n
    print("  by |w|-pattern (classes merged w->-w):")
    for absw, n in sorted(patt6.items(), key=lambda kv: (sum(kv[0]), kv[0])):
        nclasses = sum(1 for w in by_w6 if tuple(sorted(abs(c) for c in w)) == absw)
        print(f"     |w|-pattern {absw} : {n} hexagons in {nclasses} classes")
    surv6 = sum(1 for _, w in hexs if kernel2(w))
    print(f"  survive Nphi=2 corner proj: {surv6} (killed: {len(hexs)-surv6})")


def sublat_map(basis, shape):
    """Return list sublat_of[site] in 0..3 from the geometry generator."""
    if basis == "cubic":
        vertices, edges, tets, bond_wrap, adj = cg.build_graph(*shape)
        # pyrochlore sublattice = which of the 4 quarter-offset positions the
        # site occupies within its cubic cell (mod 1, times 4)
        out = {}
        for s, pos in vertices.items():
            frac = np.asarray(pos, dtype=float)
            frac = frac - np.floor(frac + 1e-9)
            key = tuple(int(round(4 * x)) % 4 for x in frac)
            out[s] = key
        keys = sorted(set(out.values()))
        idx = {k: i for i, k in enumerate(keys)}
        return {s: idx[k] for s, k in out.items()}
    else:
        # fcc generator orders sites as 4*cell + sublattice
        return {s: s % 4 for s in range(4 * shape[0] * shape[1] * shape[2])}


if __name__ == "__main__":
    for basis, shape in [("cubic", (1, 1, 1)), ("fcc", (2, 2, 2))]:
        analyse(basis, shape, sublat_map(basis, shape))
