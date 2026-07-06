"""Full short-loop / wrapping census of the FCC-primitive (3,3,1) cluster.

The L3=1 direction is below the FCC faithfulness threshold (|a_k|=a/sqrt2
spans only 2 NN spacings, so L>=2 is required). Below it, a pair of sites
can be nearest neighbours via TWO different lattice images -> a 2-site
loop (double bond) with winding w = n_a - n_b. We enumerate:
  * all NN images of every pair (true coordination = 6 if faithful)
  * 2-site loops (degenerate pairs) and their winding classes
  * ice-preserving 4-cycles and [111] hexagons and their windings
on the *representative simple graph* (one image per pair).
"""
from __future__ import annotations

from collections import Counter
from itertools import combinations

import numpy as np

import cluster_geometry_audit as cg

A1, A2, A3 = cg.FCC_A1, cg.FCC_A2, cg.FCC_A3
BASIS = cg.FCC_BASIS
SUB = cg.SUBLAT
DNN = cg.NN_DIST_PYRO


def canon(w):
    w = tuple(int(x) for x in w)
    return min(w, tuple(-x for x in w))


def census(L):
    Ls = np.array(L, float)
    verts = {}
    vid = 0
    for i in range(L[0]):
        for j in range(L[1]):
            for k in range(L[2]):
                origin = i * A1 + j * A2 + k * A3
                for s in range(4):
                    verts[vid] = origin + SUB[s]
                    vid += 1
    n = vid

    # all NN images per ordered pair
    images = {}  # (u,v) -> list of wrap vectors n (u<v)
    rng_i = range(-1, 2)
    rng_j = range(-1, 2)
    rng_k = range(-2, 3)
    for u, v in combinations(range(n), 2):
        dr = verts[v] - verts[u]
        imgs = []
        for i in rng_i:
            for j in rng_j:
                for k in rng_k:
                    nn = np.array([i, j, k], float) * Ls
                    d = np.linalg.norm(dr - BASIS @ nn)
                    if 0.01 < d < 0.4:
                        imgs.append((i, j, k))
        if imgs:
            images[(u, v)] = imgs

    # true coordination (counting images)
    coord = Counter()
    for (u, v), imgs in images.items():
        coord[u] += len(imgs)
        coord[v] += len(imgs)
    coord_hist = Counter(coord[x] for x in range(n))

    # 2-site loops from degenerate pairs
    two_loops = []  # (u, v, winding)
    for (u, v), imgs in images.items():
        if len(imgs) >= 2:
            for a, b in combinations(imgs, 2):
                w = tuple(np.array(a) - np.array(b))
                two_loops.append((u, v, w))
    two_by_w = Counter(canon(w) for _, _, w in two_loops)

    print(f"\n==== FCC {L} : {n} sites ====")
    print(f"distinct NN pairs        : {len(images)}")
    print(f"true coordination hist   : {dict(sorted(coord_hist.items()))}")
    print(f"degenerate (multi-image) pairs : "
          f"{sum(1 for v in images.values() if len(v) >= 2)}")
    print(f"2-site loops (double bonds)    : {len(two_loops)}")
    print("  2-loop winding classes (w~-w):")
    for w, c in sorted(two_by_w.items(), key=lambda kv: (sum(abs(x) for x in kv[0]), kv[0])):
        print(f"     w~{w}  |w|={tuple(sorted(abs(x) for x in w))}  multiplicity {c}")

    # representative simple graph (first image) for 4/6-cycle census
    bond_wrap = {}
    adj = {x: [] for x in range(n)}
    for (u, v), imgs in images.items():
        nn = imgs[0]
        bond_wrap[(u, v)] = nn
        bond_wrap[(v, u)] = tuple(-x for x in nn)
        adj[u].append((v, nn))
        adj[v].append((u, tuple(-x for x in nn)))
    tets = cg._enumerate_4_cliques(adj, n)
    site_tets = cg.site_to_tetrahedra(tets)
    pbc = [L[0] * A1, L[1] * A2, L[2] * A3]

    c4 = [(s, w) for s, w in cg.enumerate_simple_cycles(adj, 4)
          if cg.is_ice_preserving(s, site_tets, tets)]
    c4w = Counter(canon(w) for _, w in c4)
    print(f"ice 4-cycles (repr. graph): {len(c4)}  "
          f"contractible {sum(1 for _, w in c4 if tuple(w)==(0,0,0))}")
    for w, c in sorted(c4w.items(), key=lambda kv: (sum(abs(x) for x in kv[0]), kv[0])):
        print(f"     w~{w}  |w|={tuple(sorted(abs(x) for x in w))}  mult {c}")

    hexes = [(s, w) for s, w in cg.enumerate_simple_cycles(adj, 6)
             if cg.is_ice_preserving(s, site_tets, tets)
             and cg.is_111_planar_with_wrap(s, verts, bond_wrap, pbc)]
    hc = sum(1 for _, w in hexes if tuple(w) == (0, 0, 0))
    print(f"ice [111] hexagons        : {len(hexes)}  contractible {hc}  wrapping {len(hexes)-hc}")
    hexw = Counter(canon(w) for _, w in hexes if tuple(w) != (0, 0, 0))
    pat = Counter()
    for w, c in hexw.items():
        pat[tuple(sorted(abs(x) for x in w))] += c
    print("  wrapping-hexagon |w|-patterns:")
    for absw, c in sorted(pat.items(), key=lambda kv: (sum(kv[0]), kv[0])):
        print(f"     |w|={absw}  count {c}")


if __name__ == "__main__":
    census((3, 3, 1))
    census((2, 2, 2))  # faithful reference
