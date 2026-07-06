"""Show explicitly, for the cubic (1,1,1) cluster, the geometry of a
contractible vs a wrapping hexagon: the 6 real NN bond vectors (minimum
image, each length d_NN) and their vector sum around the loop.

Key identity: summing the real bond vectors around a closed cluster loop
gives  -w . L  (L = cluster PBC vectors). For a contractible loop this is
0 (the lift closes in R^3); for a wrapping loop it is a nonzero lattice
vector -- the loop does NOT come back in R^3, it ends one period away and
'closes' only because the torus identifies that image with the start.
"""
from __future__ import annotations

import numpy as np

import cluster_geometry_audit as cg


def order_cycle(sites, adj):
    """Return the sites of a (frozenset-found) cycle in traversal order."""
    sites = list(sites)
    nbr = {u: {v for v, _ in adj[u]} for u in sites}
    path = [sites[0]]
    used = {sites[0]}
    while len(path) < len(sites):
        for v in nbr[path[-1]]:
            if v in sites and v not in used:
                path.append(v)
                used.add(v)
                break
        else:
            break
    return path


def show(sites, label, vertices, bond_wrap, L):
    path = sites
    print(f"\n{label}: sites {path}")
    total = np.zeros(3)
    for i in range(len(path)):
        u, v = path[i], path[(i + 1) % len(path)]
        n = np.array(bond_wrap[(u, v)])
        dr_home = np.asarray(vertices[v]) - np.asarray(vertices[u])
        bond = dr_home - n * L           # real minimum-image bond vector
        total += bond
        tag = "  (boundary-crossing!)" if tuple(n) != (0, 0, 0) else ""
        print(f"   {u:2d}->{v:2d}  wrap n={tuple(int(x) for x in n)}  "
              f"bond={np.round(bond,3)} |bond|={np.linalg.norm(bond):.3f}{tag}")
    print(f"   SUM of real bond vectors = {np.round(total,3)}  (|.|={np.linalg.norm(total):.3f})")
    print(f"   -> {'CLOSES in R^3 (contractible)' if np.linalg.norm(total)<1e-6 else 'does NOT close in R^3: ends '+str(np.round(total,3))+' away'}")


def main():
    L = np.array([1.0, 1.0, 1.0])
    vertices, edges, _t, bond_wrap, adj = cg.build_graph(1, 1, 1)
    tets = cg._enumerate_4_cliques(adj, len(vertices))
    site_tets = cg.site_to_tetrahedra(tets)
    pbc = [np.array([1.0, 0, 0]), np.array([0, 1.0, 0]), np.array([0, 0, 1.0])]

    hexes = [(s, w) for s, w in cg.enumerate_simple_cycles(adj, 6)
             if cg.is_ice_preserving(s, site_tets, tets)
             and cg.is_111_planar_with_wrap(s, vertices, bond_wrap, pbc)]
    contr = next((s, w) for s, w in hexes if tuple(w) == (0, 0, 0))
    wrap = next((s, w) for s, w in hexes if tuple(w) != (0, 0, 0))

    c4 = [(s, w) for s, w in cg.enumerate_simple_cycles(adj, 4)
          if cg.is_ice_preserving(s, site_tets, tets)]
    fc = c4[0]

    print("cubic (1,1,1) cluster, L=(1,1,1)")
    show(order_cycle(contr[0], adj), f"CONTRACTIBLE hexagon w={contr[1]}", vertices, bond_wrap, L)
    show(order_cycle(wrap[0], adj), f"WRAPPING hexagon w={wrap[1]}", vertices, bond_wrap, L)
    show(order_cycle(fc[0], adj), f"4-cycle w={fc[1]}", vertices, bond_wrap, L)


if __name__ == "__main__":
    main()
