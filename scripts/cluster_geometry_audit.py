"""
Cluster geometry audit for pyrochlore ED targeting the QSI g_hex = 12|Jpm|^3/Jzz^2 scale.

For each candidate cluster shape (dim1, dim2, dim3) it reports:
    N_sites, N_bonds, N_wrap_bonds
    girth_phys = length of the shortest ICE-RULE-PRESERVING cycle on the
                 PBC graph (a tetrahedron is touched 0 or 2 times by the cycle)
    N_4cycles_phys, N_4cycles_phys_contractible
    N_hex,         N_hex_contractible,         N_hex_wrap
    rho_hex = N_hex_contractible / N_hex   (fraction of physical hexagons
              that survive the contractible-loop projector)
    Hilbert space dimension in Sz=0 sector

The intent: explain to the user how cluster geometry controls whether the
twist-averaged ED can resolve the genuine g_hex scale, or is bottlenecked by
boundary-truncated hexagons.
"""
from __future__ import annotations

import sys
from itertools import combinations, product
from math import comb
from pathlib import Path

import numpy as np

ED_HELPER_PATH = (
    Path(__file__).resolve().parents[1].parent
    / "QED"
    / "python"
    / "edlib"
)
sys.path.insert(0, str(ED_HELPER_PATH))
from helper_pyrochlore_super import generate_pyrochlore_super_cluster  # noqa: E402


def build_graph(dim1: int, dim2: int, dim3: int):
    """Return (vertices, edges, tetrahedra, bond_wrap, adj) for a cluster
    spanned by L1, L2, L3 cubic conventional cells.

    bond_wrap[(u,v)] = integer wrap vector n (in cubic-cell units) with
                       r_v - r_u - n*L being the minimum-image displacement
                       (n is zero in the bulk).
    """
    vertices, edges, tetrahedra, _, _ = generate_pyrochlore_super_cluster(
        dim1, dim2, dim3, use_pbc=True
    )
    L = np.array([dim1, dim2, dim3], dtype=float)

    bond_wrap = {}
    for u, v in edges:
        dr = np.asarray(vertices[v]) - np.asarray(vertices[u])
        n = np.round(dr / L).astype(int)
        bond_wrap[(u, v)] = tuple(int(x) for x in n)
        bond_wrap[(v, u)] = tuple(-int(x) for x in n)

    adj: dict[int, list[tuple[int, tuple[int, int, int]]]] = {v: [] for v in vertices}
    for u, v in edges:
        adj[u].append((v, bond_wrap[(u, v)]))
        adj[v].append((u, bond_wrap[(v, u)]))

    return vertices, edges, tetrahedra, bond_wrap, adj


# ----------------------------------------------------------------------------
# FCC-primitive cluster geometry
# ----------------------------------------------------------------------------
# Primitive translations of the pyrochlore lattice are the FCC vectors
#       a1 = (1,1,0)/2,    a2 = (1,0,1)/2,    a3 = (0,1,1)/2
# (in units of the cubic conventional lattice constant). Each primitive cell
# contains one up-tetrahedron with 4 sublattice sites at fractional offsets
#       s0 = (0, 0, 0),                s1 = (0, 1/4, 1/4),
#       s2 = (1/4, 0, 1/4),            s3 = (1/4, 1/4, 0).
# A cluster of multiplicities (L1, L2, L3) holds 4 L1 L2 L3 sites.
# ----------------------------------------------------------------------------

FCC_A1 = np.array([1.0, 1.0, 0.0]) / 2.0
FCC_A2 = np.array([1.0, 0.0, 1.0]) / 2.0
FCC_A3 = np.array([0.0, 1.0, 1.0]) / 2.0
FCC_BASIS = np.column_stack([FCC_A1, FCC_A2, FCC_A3])
FCC_BASIS_INV = np.linalg.inv(FCC_BASIS)
SUBLAT = np.array(
    [
        [0.0, 0.0, 0.0],
        [0.0, 0.25, 0.25],
        [0.25, 0.0, 0.25],
        [0.25, 0.25, 0.0],
    ]
)
NN_DIST_PYRO = np.sqrt(2.0) / 4.0  # 1/sqrt(8)


def build_graph_fcc(L1: int, L2: int, L3: int):
    """Build the pyrochlore NN graph on the FCC-primitive parallelepiped of
    multiplicities (L1, L2, L3). Bond wrap vectors are in (a1, a2, a3) units.

    Also returns a ``degenerate_pairs`` list: pairs of sites that admit more
    than one minimum-image NN displacement (i.e., the cluster fundamental
    domain is too small in some direction). On such clusters the NN graph is
    ill-defined: only one of the equivalent images is recorded as an edge,
    causing some sites to have fewer than 6 NNs.
    """
    Ls = np.array([L1, L2, L3], dtype=float)
    vertices: dict[int, tuple[float, float, float]] = {}
    vid = 0
    for i in range(L1):
        for j in range(L2):
            for k in range(L3):
                cell_origin = i * FCC_A1 + j * FCC_A2 + k * FCC_A3
                for s in range(4):
                    pos = cell_origin + SUBLAT[s]
                    vertices[vid] = tuple(pos)
                    vid += 1

    n_sites = vid
    edges: set[tuple[int, int]] = set()
    bond_wrap: dict[tuple[int, int], tuple[int, int, int]] = {}
    degenerate_pairs: list[tuple[int, int, int]] = []

    for u in range(n_sites):
        ru = np.asarray(vertices[u])
        for v in range(u + 1, n_sites):
            rv = np.asarray(vertices[v])
            dr_home = rv - ru
            nn_images = []
            for i in (-1, 0, 1):
                for j in (-1, 0, 1):
                    for k in (-1, 0, 1):
                        n = np.array([i, j, k], dtype=float) * Ls
                        actual = dr_home - FCC_BASIS @ n
                        d = float(np.linalg.norm(actual))
                        if 0.01 < d < 0.4:
                            nn_images.append(
                                ((int(round(i)), int(round(j)), int(round(k))), d)
                            )
            if not nn_images:
                continue
            if len(nn_images) > 1:
                degenerate_pairs.append((u, v, len(nn_images)))
            n_chosen, _ = nn_images[0]
            edges.add((u, v))
            bond_wrap[(u, v)] = n_chosen
            bond_wrap[(v, u)] = tuple(-int(x) for x in n_chosen)

    adj: dict[int, list[tuple[int, tuple[int, int, int]]]] = {v: [] for v in vertices}
    for u, v in edges:
        adj[u].append((v, bond_wrap[(u, v)]))
        adj[v].append((u, bond_wrap[(v, u)]))

    tetrahedra = _enumerate_4_cliques(adj, n_sites)
    # Stash degenerate-pair info on the bond_wrap dict so audit() can read it
    bond_wrap.setdefault("__degenerate_pairs__", degenerate_pairs)
    return vertices, list(edges), tetrahedra, bond_wrap, adj


def _enumerate_4_cliques(adj, n_sites):
    """The pyrochlore lattice's 4-cliques are exactly its tetrahedra (each
    site participates in two 4-cliques: its up- and down-tetrahedron)."""
    tets: set[tuple[int, ...]] = set()
    nbrs_set = {u: {v for v, _ in adj[u]} for u in adj}
    for u in range(n_sites):
        nbrs_u = sorted(v for v in nbrs_set[u] if v > u)
        m = len(nbrs_u)
        for i in range(m):
            for j in range(i + 1, m):
                v, w = nbrs_u[i], nbrs_u[j]
                if w not in nbrs_set[v]:
                    continue
                for k in range(j + 1, m):
                    x = nbrs_u[k]
                    if x in nbrs_set[v] and x in nbrs_set[w]:
                        tets.add(tuple(sorted([u, v, w, x])))
    return list(tets)


def site_to_tetrahedra(tetrahedra):
    """Return dict site -> list of tet ids it belongs to (each site lies in 2)."""
    out: dict[int, list[int]] = {}
    for t_id, sites in enumerate(tetrahedra):
        for s in sites:
            out.setdefault(s, []).append(t_id)
    return out


def is_ice_preserving(cycle_sites, site_tets, tetrahedra):
    """A simple cycle preserves the 2in-2out constraint iff each tetrahedron is
    visited by 0 or 2 of the cycle's sites (tetrahedra of length 4)."""
    counts: dict[int, int] = {}
    for s in cycle_sites:
        for t in site_tets[s]:
            counts[t] = counts.get(t, 0) + 1
    return all(c in (0, 2) for c in counts.values())


def enumerate_simple_cycles(adj, length: int):
    """Return list of (sites_tuple, winding_tuple) for simple cycles of given
    length, each cycle counted once (no rotation / reflection duplicates)."""
    cycles: dict[frozenset, tuple[tuple[int, ...], tuple[int, int, int]]] = {}
    n_sites = len(adj)

    # Cycle-search rooted at each vertex; require visited in increasing order
    # to break rotation symmetry, and require neighbour increasing to break
    # reflection.
    for start in range(n_sites):
        path = [start]
        wind = np.zeros(3, dtype=int)
        _dfs(start, start, path, wind, length, adj, cycles)
    return list(cycles.values())


def _dfs(start, u, path, wind, target_len, adj, cycles):
    if len(path) == target_len:
        for v, n in adj[u]:
            if v == start:
                total = wind + np.array(n)
                key = frozenset(path)
                if key in cycles:
                    continue
                cycles[key] = (tuple(path), tuple(int(x) for x in total))
        return
    for v, n in adj[u]:
        if v <= start:  # break rotation symmetry
            continue
        if v in path:
            continue
        path.append(v)
        wind += np.array(n)
        _dfs(start, v, path, wind, target_len, adj, cycles)
        path.pop()
        wind -= np.array(n)


def count_wrap_bonds(edges, bond_wrap):
    n = 0
    for u, v in edges:
        if bond_wrap[(u, v)] != (0, 0, 0):
            n += 1
    return n


def cluster_pbc_vectors(dim1, dim2, dim3, basis: str):
    """Cluster's three PBC vectors in Cartesian, indexed [k] for k=0,1,2."""
    if basis == "cubic":
        return [
            np.array([dim1, 0.0, 0.0]),
            np.array([0.0, dim2, 0.0]),
            np.array([0.0, 0.0, dim3]),
        ]
    elif basis == "fcc":
        return [
            dim1 * FCC_A1,
            dim2 * FCC_A2,
            dim3 * FCC_A3,
        ]
    else:
        raise ValueError(basis)


def is_111_planar_with_wrap(path, vertices, bond_wrap, pbc_vecs, tol: float = 1e-6) -> bool:
    """Test whether the simple cycle (in path order) is planar with normal in
    the four [111]-family directions, using *unwrapped* edge displacements."""
    n = len(path)
    edge_disps = []
    for i in range(n):
        u = path[i]
        v = path[(i + 1) % n]
        n_uv = bond_wrap.get((u, v), (0, 0, 0))
        dr_home = np.asarray(vertices[v]) - np.asarray(vertices[u])
        wrap_disp = (
            n_uv[0] * pbc_vecs[0]
            + n_uv[1] * pbc_vecs[1]
            + n_uv[2] * pbc_vecs[2]
        )
        edge_disps.append(dr_home - wrap_disp)
    edge_disps = np.asarray(edge_disps)
    normals = [
        np.array([1.0, 1.0, 1.0]),
        np.array([1.0, 1.0, -1.0]),
        np.array([1.0, -1.0, 1.0]),
        np.array([-1.0, 1.0, 1.0]),
    ]
    for nn in normals:
        nn = nn / np.linalg.norm(nn)
        if np.max(np.abs(edge_disps @ nn)) < tol:
            return True
    return False


def audit(dim1, dim2, dim3, max_len: int = 6, basis: str = "cubic"):
    if basis == "cubic":
        vertices, edges, tetrahedra, bond_wrap, adj = build_graph(dim1, dim2, dim3)
    elif basis == "fcc":
        vertices, edges, tetrahedra, bond_wrap, adj = build_graph_fcc(dim1, dim2, dim3)
    else:
        raise ValueError(f"unknown basis {basis!r}")
    site_tets = site_to_tetrahedra(tetrahedra)
    n_sites = len(vertices)
    pbc_vecs = cluster_pbc_vectors(dim1, dim2, dim3, basis)

    degree = {u: 0 for u in range(n_sites)}
    for u, v in edges:
        degree[u] += 1
        degree[v] += 1
    min_deg = min(degree.values()) if degree else 0
    max_deg = max(degree.values()) if degree else 0
    n_degenerate_pairs = len(bond_wrap.get("__degenerate_pairs__", []))
    is_degenerate = (min_deg < 6) or (n_degenerate_pairs > 0)

    # --- 4-cycles ---
    cycles4 = enumerate_simple_cycles(adj, 4)
    cycles4_phys = [
        (sites, w)
        for sites, w in cycles4
        if is_ice_preserving(sites, site_tets, tetrahedra)
    ]
    n4_phys_contract = sum(1 for _, w in cycles4_phys if w == (0, 0, 0))
    n4_phys_wrap = len(cycles4_phys) - n4_phys_contract

    # --- 6-cycles (hexagons) ---
    if max_len >= 6 and n_sites <= 64:
        cycles6 = enumerate_simple_cycles(adj, 6)
        # Planarity must be checked using *unwrapped* edge displacements,
        # otherwise wrapping hexagons (whose 6 home-position representatives
        # do not span a [111] plane) get falsely rejected.
        hexagons = [
            (sites, w)
            for sites, w in cycles6
            if is_ice_preserving(sites, site_tets, tetrahedra)
            and is_111_planar_with_wrap(sites, vertices, bond_wrap, pbc_vecs)
        ]
        n_hex = len(hexagons)
        n_hex_contract = sum(1 for _, w in hexagons if w == (0, 0, 0))
        n_hex_wrap = n_hex - n_hex_contract
    else:
        n_hex = n_hex_contract = n_hex_wrap = -1

    # --- Sz=0 Hilbert space dimension ---
    sz0_dim = comb(n_sites, n_sites // 2) if n_sites % 2 == 0 else None

    return {
        "basis": basis,
        "shape": (dim1, dim2, dim3),
        "N_sites": n_sites,
        "N_bonds": len(edges),
        "N_wrap_bonds": count_wrap_bonds(edges, bond_wrap),
        "N_tets": len(tetrahedra),
        "N_4cycles_phys": len(cycles4_phys),
        "N_4cycles_phys_contract": n4_phys_contract,
        "N_4cycles_phys_wrap": n4_phys_wrap,
        "N_hex": n_hex,
        "N_hex_contract": n_hex_contract,
        "N_hex_wrap": n_hex_wrap,
        # ratio against unique 6-cycles found in the graph
        "rho_hex_local": (n_hex_contract / n_hex) if n_hex > 0 else float("nan"),
        # ratio against the bulk density of 1 hexagonal plaquette per site
        # (4 hexagons per primitive cell, 4 sites per primitive cell)
        "nu_hex_c": (n_hex_contract / n_sites) if n_hex_contract >= 0 else float("nan"),
        "sz0_dim": sz0_dim,
        "min_degree": min_deg,
        "max_degree": max_deg,
        "n_degenerate_pairs": n_degenerate_pairs,
        "is_degenerate": is_degenerate,
    }


def is_111_planar(sites, vertices, tol: float = 1e-6) -> bool:
    """Check whether the 6 sites lie in a single plane whose normal is one of
    the four [111]-family directions (1,1,1), (1,1,-1), (1,-1,1), (-1,1,1)."""
    pts = np.array([vertices[s] for s in sites])
    centroid = pts.mean(axis=0)
    normals = [
        np.array([1.0, 1.0, 1.0]),
        np.array([1.0, 1.0, -1.0]),
        np.array([1.0, -1.0, 1.0]),
        np.array([-1.0, 1.0, 1.0]),
    ]
    for n in normals:
        n = n / np.linalg.norm(n)
        d = (pts - centroid) @ n
        if np.max(np.abs(d)) < tol:
            return True
    return False


def _format_row(r):
    flag = "  DEGEN" if r["is_degenerate"] else ""
    return (
        f"{r['basis']:>5s} {str(r['shape']):>10s}  {r['N_sites']:>3d}  "
        f"{r['N_bonds']:>3d}/{r['N_wrap_bonds']:<3d}  "
        f"deg=[{r['min_degree']},{r['max_degree']}]/{r['n_degenerate_pairs']:<2d}  "
        f"{r['N_4cycles_phys']:>4d}/{r['N_4cycles_phys_contract']:<4d}/"
        f"{r['N_4cycles_phys_wrap']:<4d}    "
        f"{r['N_hex']:>4d}/{r['N_hex_contract']:<4d}/{r['N_hex_wrap']:<4d}    "
        f"{r['nu_hex_c']:.3f}  {r['rho_hex_local']:.3f}  "
        f"{str(r['sz0_dim']):>14s}{flag}"
    )


def main():
    cubic_candidates = [
        (1, 1, 1),
        (2, 1, 1),
        (2, 2, 1),
        (2, 2, 2),
    ]
    fcc_candidates = [
        (1, 1, 1),
        (2, 1, 1),
        (2, 2, 1),
        (2, 2, 2),
        (3, 2, 2),
        (4, 2, 2),
        (3, 3, 2),
    ]

    print(
        f"{'basis':>5s} {'shape':>10s}  {'N':>3s}  {'b/wb':>7s}  "
        f"{'deg/multi':>14s}  "
        f"{'4cyc(p/c/w)':>14s}  "
        f"{'hex(N/c/w)':>16s}  {'nu_c':>5s}  {'rho_l':>5s}  {'dim Sz=0':>14s}"
    )
    print("-" * 130)

    print("# --- cubic conventional clusters (N = 16 L1 L2 L3) ---")
    for shape in cubic_candidates:
        try:
            r = audit(*shape, basis="cubic")
            print(_format_row(r))
        except Exception as e:
            print(f"  cubic {shape}: failed: {e}")

    print("# --- FCC primitive clusters (N = 4 L1 L2 L3) ---")
    for shape in fcc_candidates:
        try:
            r = audit(*shape, basis="fcc")
            print(_format_row(r))
        except Exception as e:
            print(f"  fcc {shape}: failed: {e}")


if __name__ == "__main__":
    main()
