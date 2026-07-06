"""Search for a cluster where the N_phi=2 corner average kills all
ice-preserving 4-cycles (odd winding) but KEEPS some wrapping hexagons
(all-even winding, |w_a|=2), i.e. where 4-cycles and wrapping hexagons
fall into different winding-parity classes.

For each shape we report:
  4cyc: total ice 4-cycles / how many survive N_phi=2 (all-even winding)
  hexW: total WRAPPING ice [111] hexagons / how many survive N_phi=2
A cluster answers the user's question iff  4cyc_survive == 0  and
hexW_survive > 0.
"""
from __future__ import annotations

import numpy as np

import cluster_geometry_audit as cg


def kernel2_survives(w):
    return all(wa % 2 == 0 for wa in w)


def analyse(basis, shape):
    if basis == "cubic":
        vertices, edges, _t, bond_wrap, adj = cg.build_graph(*shape)
    else:
        vertices, edges, _t, bond_wrap, adj = cg.build_graph_fcc(*shape)
    n = len(vertices)
    tets = cg._enumerate_4_cliques(adj, n)
    site_tets = cg.site_to_tetrahedra(tets)
    pbc = cg.cluster_pbc_vectors(*shape, basis)

    deg = {u: 0 for u in range(n)}
    for u, v in edges:
        deg[u] += 1
        deg[v] += 1
    faithful = (min(deg.values()) == 6 and max(deg.values()) == 6)

    c4 = [
        (s, w) for s, w in cg.enumerate_simple_cycles(adj, 4)
        if cg.is_ice_preserving(s, site_tets, tets)
    ]
    c4_surv = sum(1 for _, w in c4 if kernel2_survives(w))

    hexes = [
        (s, w) for s, w in cg.enumerate_simple_cycles(adj, 6)
        if cg.is_ice_preserving(s, site_tets, tets)
        and cg.is_111_planar_with_wrap(s, vertices, bond_wrap, pbc)
    ]
    hexW = [(s, w) for s, w in hexes if tuple(w) != (0, 0, 0)]
    hexW_surv = sum(1 for _, w in hexW if kernel2_survives(w))

    hit = (c4_surv == 0 and hexW_surv > 0 and len(c4) > 0)
    print(f"{basis:>5s} {str(shape):>9s} N={n:<3d} faithful={str(faithful):<5s} "
          f"4cyc {len(c4):>3d}/surv {c4_surv:<3d}   "
          f"hexWrap {len(hexW):>3d}/surv {hexW_surv:<3d}"
          f"{'   <-- SEPARATES' if hit else ''}")


if __name__ == "__main__":
    shapes = [
        ("cubic", (1, 1, 1)), ("cubic", (2, 1, 1)), ("cubic", (3, 1, 1)),
        ("cubic", (4, 1, 1)), ("cubic", (2, 2, 1)), ("cubic", (3, 2, 1)),
        ("cubic", (3, 3, 1)), ("cubic", (4, 2, 1)),
        ("fcc", (2, 2, 2)), ("fcc", (3, 2, 2)), ("fcc", (4, 2, 2)),
        ("fcc", (3, 3, 2)),
    ]
    for basis, shape in shapes:
        try:
            analyse(basis, shape)
        except Exception as e:
            print(f"{basis} {shape}: failed: {e}")
