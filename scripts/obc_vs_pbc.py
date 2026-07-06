"""Compare open (OBC) vs periodic (PBC) boundary conditions for small
pyrochlore clusters: NN coordination, ice-preserving 4-cycles, and
[111]-planar ice hexagons (contractible). Shows what OBC costs in lost
bulk hexagons and surface under-coordination, vs what PBC+twist costs in
spurious wrapping loops.
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1].parent / "QED" / "python" / "edlib"))
from helper_pyrochlore_super import generate_pyrochlore_super_cluster  # noqa: E402

import cluster_geometry_audit as cg  # noqa: E402


def build(dim, use_pbc):
    vertices, edges, _t, _, _ = generate_pyrochlore_super_cluster(*dim, use_pbc=use_pbc)
    L = np.array(dim, dtype=float)
    bond_wrap = {}
    for u, v in edges:
        dr = np.asarray(vertices[v]) - np.asarray(vertices[u])
        nn = np.round(dr / L).astype(int) if use_pbc else np.zeros(3, int)
        bond_wrap[(u, v)] = tuple(int(x) for x in nn)
        bond_wrap[(v, u)] = tuple(-int(x) for x in nn)
    adj = {v: [] for v in vertices}
    for u, v in edges:
        adj[u].append((v, bond_wrap[(u, v)]))
        adj[v].append((u, bond_wrap[(v, u)]))
    return vertices, list(edges), bond_wrap, adj


def report(dim, use_pbc):
    vertices, edges, bond_wrap, adj = build(dim, use_pbc)
    n = len(vertices)
    tets = cg._enumerate_4_cliques(adj, n)
    site_tets = cg.site_to_tetrahedra(tets)
    deg = Counter()
    for u, v in edges:
        deg[u] += 1
        deg[v] += 1
    deg_hist = Counter(deg[u] for u in range(n))
    n6 = deg_hist.get(6, 0)

    c4 = [s for s, w in cg.enumerate_simple_cycles(adj, 4)
          if cg.is_ice_preserving(s, site_tets, tets)]

    pbc_vecs = cg.cluster_pbc_vectors(*dim, "cubic")
    hexes = [(s, w) for s, w in cg.enumerate_simple_cycles(adj, 6)
             if cg.is_ice_preserving(s, site_tets, tets)
             and cg.is_111_planar_with_wrap(s, vertices, bond_wrap, pbc_vecs)]
    hex_contract = sum(1 for _, w in hexes if tuple(w) == (0, 0, 0))

    tag = "PBC" if use_pbc else "OBC"
    nu = hex_contract / n
    print(f"  {tag}: N={n:<3d} bonds={len(edges):<3d} "
          f"deg-hist={dict(sorted(deg_hist.items()))} (6-coord: {n6}/{n})  "
          f"ice-4cyc={len(c4):<3d}  hex(contractible)={hex_contract:<3d}  nu_c={nu:.3f}")


if __name__ == "__main__":
    for dim in [(1, 1, 1), (2, 2, 2)]:
        print(f"cubic {dim}:")
        report(dim, use_pbc=True)
        report(dim, use_pbc=False)
