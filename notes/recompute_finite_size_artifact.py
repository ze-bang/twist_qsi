#!/usr/bin/env python3
"""Independent recomputation of the QSI finite-size loop artifact.

This script intentionally does not import the old twist_qsi_demo scripts or
read git history. It rebuilds the geometry, ice manifold, order-2/order-3
effective Hamiltonian, zero-transport projector, and specific-heat peaks from
first principles.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from itertools import combinations, product
from pathlib import Path

import numpy as np


NN_DIST = np.sqrt(2.0) / 4.0
SUBLAT = np.array(
    [
        [0.0, 0.0, 0.0],
        [0.0, 0.25, 0.25],
        [0.25, 0.0, 0.25],
        [0.25, 0.25, 0.0],
    ],
)
CUBIC_FCC_CENTERS = np.array(
    [
        [0.0, 0.0, 0.0],
        [0.0, 0.5, 0.5],
        [0.5, 0.0, 0.5],
        [0.5, 0.5, 0.0],
    ],
)
FCC_A = np.array(
    [
        [0.5, 0.5, 0.0],
        [0.5, 0.0, 0.5],
        [0.0, 0.5, 0.5],
    ],
)


@dataclass
class Cluster:
    label: str
    basis: str
    shape: tuple[int, int, int]
    positions: np.ndarray
    Lvecs: np.ndarray
    bonds: np.ndarray
    bond_wrap: np.ndarray
    adj: dict[int, list[tuple[int, tuple[int, int, int]]]]
    tets: list[tuple[int, int, int, int]]
    tet_masks: np.ndarray
    loops4: list[tuple[tuple[int, ...], tuple[int, int, int]]] = field(default_factory=list)
    hexes: list[tuple[tuple[int, ...], tuple[int, int, int]]] = field(default_factory=list)
    ice_states: np.ndarray | None = None
    ice_index: dict[int, int] | None = None

    @property
    def n_sites(self) -> int:
        return len(self.positions)

    @property
    def n_ice(self) -> int:
        return 0 if self.ice_states is None else len(self.ice_states)


def make_positions(basis: str, shape: tuple[int, int, int]) -> tuple[np.ndarray, np.ndarray]:
    L1, L2, L3 = shape
    pos = []
    if basis == "cubic":
        for i in range(L1):
            for j in range(L2):
                for k in range(L3):
                    cell = np.array([i, j, k], dtype=float)
                    for center in CUBIC_FCC_CENTERS:
                        for sub in SUBLAT:
                            pos.append(cell + center + sub)
        Lvecs = np.diag([L1, L2, L3]).astype(float)
    elif basis == "fcc":
        for i in range(L1):
            for j in range(L2):
                for k in range(L3):
                    cell = i * FCC_A[0] + j * FCC_A[1] + k * FCC_A[2]
                    for sub in SUBLAT:
                        pos.append(cell + sub)
        Lvecs = np.array([L1 * FCC_A[0], L2 * FCC_A[1], L3 * FCC_A[2]], dtype=float)
    else:
        raise ValueError(basis)
    return np.array(pos, dtype=float), Lvecs


def build_bonds(positions: np.ndarray, Lvecs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    bonds = []
    wraps = []
    tol = 1e-6
    n_sites = len(positions)
    shifts = list(product([-1, 0, 1], repeat=3))
    for u in range(n_sites):
        for v in range(u + 1, n_sites):
            dr = positions[v] - positions[u]
            candidates = []
            for n in shifts:
                nvec = np.array(n, dtype=float)
                actual = dr - nvec @ Lvecs
                dist = np.linalg.norm(actual)
                if abs(dist - NN_DIST) < tol:
                    candidates.append(tuple(int(x) for x in n))
            if candidates:
                candidates.sort(key=lambda x: (sum(abs(a) for a in x), x))
                bonds.append((u, v))
                wraps.append(candidates[0])
    return np.array(bonds, dtype=np.int64), np.array(wraps, dtype=np.int64)


def adjacency(n_sites: int, bonds: np.ndarray, wraps: np.ndarray):
    adj: dict[int, list[tuple[int, tuple[int, int, int]]]] = {i: [] for i in range(n_sites)}
    for (u, v), n in zip(bonds, wraps):
        nt = tuple(int(x) for x in n)
        adj[int(u)].append((int(v), nt))
        adj[int(v)].append((int(u), tuple(-x for x in nt)))
    return adj


def enumerate_tetrahedra(adj) -> list[tuple[int, int, int, int]]:
    n_sites = len(adj)
    nbrs = {u: {v for v, _ in adj[u]} for u in adj}
    tets = set()
    for u in range(n_sites):
        for tri in combinations([v for v in nbrs[u] if v > u], 3):
            a, b, c = tri
            if b in nbrs[a] and c in nbrs[a] and c in nbrs[b]:
                tets.add(tuple(sorted((u, a, b, c))))
    return sorted(tets)


def site_to_tets(tets):
    out = defaultdict(list)
    for tid, tet in enumerate(tets):
        for s in tet:
            out[s].append(tid)
    return out


def ice_preserving(path, site_tets) -> bool:
    counts = Counter()
    for s in path:
        counts.update(site_tets[s])
    return all(c in (0, 2) for c in counts.values())


def simple_cycles(adj, length: int):
    cycles = {}
    for start in range(len(adj)):
        dfs_cycles(adj, start, start, [start], np.zeros(3, dtype=int), length, cycles)
    return list(cycles.values())


def dfs_cycles(adj, start, u, path, wind, length, cycles):
    if len(path) == length:
        for v, n in adj[u]:
            if v == start:
                total = wind + np.array(n, dtype=int)
                key = frozenset(path)
                if key not in cycles:
                    cycles[key] = (tuple(path), tuple(int(x) for x in total))
        return
    for v, n in adj[u]:
        if v <= start or v in path:
            continue
        path.append(v)
        wind += np.array(n, dtype=int)
        dfs_cycles(adj, start, v, path, wind, length, cycles)
        wind -= np.array(n, dtype=int)
        path.pop()


def unwrapped_path(path, positions, adj):
    coords = [positions[path[0]]]
    current = coords[0].copy()
    for a, b in zip(path, path[1:] + path[:1]):
        for v, n in adj[a]:
            if v == b:
                current = current + (positions[b] - positions[a] - np.array(n) @ CLUSTER_FOR_PLANES.Lvecs)
                coords.append(current.copy())
                break
    return np.array(coords[:-1])


CLUSTER_FOR_PLANES: Cluster


def is_hexagon(path, positions, adj, Lvecs) -> bool:
    """Keep ordinary pyrochlore hexagons: six distinct sites, planar polygon."""
    coords = []
    current = positions[path[0]].copy()
    coords.append(current.copy())
    for a, b in zip(path, path[1:]):
        for v, n in adj[a]:
            if v == b:
                current = current + (positions[b] - positions[a] - np.array(n) @ Lvecs)
                coords.append(current.copy())
                break
    pts = np.array(coords)
    center = pts.mean(axis=0)
    _, svals, _ = np.linalg.svd(pts - center)
    return bool(svals[-1] < 1e-7)


def enumerate_ice_states(n_sites: int, tets: list[tuple[int, ...]]) -> np.ndarray:
    tets_of_site = [[] for _ in range(n_sites)]
    for tid, tet in enumerate(tets):
        for site in tet:
            tets_of_site[site].append(tid)
    assigned = np.zeros(len(tets), dtype=np.int8)
    ups = np.zeros(len(tets), dtype=np.int8)
    states: list[int] = []

    def rec(site: int, state: int):
        if site == n_sites:
            states.append(state)
            return
        for spin in (0, 1):
            ok = True
            for tid in tets_of_site[site]:
                u = ups[tid] + spin
                d = assigned[tid] + 1 - u
                if u > 2 or d > 2:
                    ok = False
                    break
            if not ok:
                continue
            for tid in tets_of_site[site]:
                assigned[tid] += 1
                ups[tid] += spin
            rec(site + 1, state | (spin << site))
            for tid in tets_of_site[site]:
                assigned[tid] -= 1
                ups[tid] -= spin

    rec(0, 0)
    return np.array(sorted(states), dtype=np.uint64)


def build_cluster(basis: str, shape: tuple[int, int, int]) -> Cluster:
    positions, Lvecs = make_positions(basis, shape)
    bonds, wraps = build_bonds(positions, Lvecs)
    adj = adjacency(len(positions), bonds, wraps)
    tets = enumerate_tetrahedra(adj)
    tet_masks = np.array(
        [sum(np.uint64(1) << np.uint64(s) for s in tet) for tet in tets],
        dtype=np.uint64,
    )
    cl = Cluster(
        label=f"{basis}{shape}",
        basis=basis,
        shape=shape,
        positions=positions,
        Lvecs=Lvecs,
        bonds=bonds,
        bond_wrap=wraps,
        adj=adj,
        tets=tets,
        tet_masks=tet_masks,
    )
    st = site_to_tets(tets)
    cl.loops4 = [(p, w) for p, w in simple_cycles(adj, 4) if ice_preserving(p, st)]
    cl.hexes = [
        (p, w)
        for p, w in simple_cycles(adj, 6)
        if ice_preserving(p, st) and is_hexagon(p, positions, adj, Lvecs)
    ]
    cl.ice_states = enumerate_ice_states(cl.n_sites, tets)
    cl.ice_index = {int(s): i for i, s in enumerate(cl.ice_states)}
    return cl


def ising_energy(cl: Cluster, states: np.ndarray) -> np.ndarray:
    bits = states[:, None] & cl.tet_masks[None, :]
    ups = np.bitwise_count(bits).astype(np.int16)
    return 0.5 * ((ups - 2) ** 2).sum(axis=1).astype(float)


def apply_v(cl: Cluster, states, cols, Ns, amps):
    out_s, out_c, out_N, out_a = [], [], [], []
    one = np.uint64(1)
    for (i, j), nij in zip(cl.bonds, cl.bond_wrap):
        bi = one << np.uint64(i)
        bj = one << np.uint64(j)
        flip = bi | bj
        up_i = (states & bi) != 0
        up_j = (states & bj) != 0
        m = (~up_i) & up_j
        if m.any():
            out_s.append(states[m] ^ flip)
            out_c.append(cols[m])
            out_N.append(Ns[m] + nij)
            out_a.append(-amps[m])
        m = up_i & (~up_j)
        if m.any():
            out_s.append(states[m] ^ flip)
            out_c.append(cols[m])
            out_N.append(Ns[m] - nij)
            out_a.append(-amps[m])
    if not out_s:
        return (
            np.empty(0, dtype=np.uint64),
            np.empty(0, dtype=np.int64),
            np.empty((0, 3), dtype=np.int64),
            np.empty(0, dtype=float),
        )
    return np.concatenate(out_s), np.concatenate(out_c), np.concatenate(out_N), np.concatenate(out_a)


def merge_rows(parts):
    if not parts:
        return empty_rows()
    s = np.concatenate([p["s"] for p in parts])
    t = np.concatenate([p["t"] for p in parts])
    N = np.concatenate([p["N"] for p in parts])
    c = np.concatenate([p["c"] for p in parts])
    key = np.stack([s, t, N[:, 0], N[:, 1], N[:, 2]], axis=1)
    uniq, inv = np.unique(key, axis=0, return_inverse=True)
    summed = np.zeros(len(uniq), dtype=float)
    np.add.at(summed, inv, c)
    keep = np.abs(summed) > 1e-13
    return {"s": uniq[keep, 0], "t": uniq[keep, 1], "N": uniq[keep, 2:5], "c": summed[keep]}


def empty_rows():
    return {
        "s": np.empty(0, dtype=np.int64),
        "t": np.empty(0, dtype=np.int64),
        "N": np.empty((0, 3), dtype=np.int64),
        "c": np.empty(0, dtype=float),
    }


def dedupe_projected(states, cols, Ns, amps, targets):
    return merge_rows([{"s": cols, "t": targets, "N": Ns, "c": amps}])


def sw_order23(cl: Cluster, verbose=True):
    ice = cl.ice_states
    h2_parts = []
    h3_parts = []
    for col, state in enumerate(ice):
        st0 = np.array([state], dtype=np.uint64)
        c0 = np.array([col], dtype=np.int64)
        n0 = np.zeros((1, 3), dtype=np.int64)
        a0 = np.ones(1, dtype=float)
        s1, c1, n1, a1 = apply_v(cl, st0, c0, n0, a0)
        e1 = ising_energy(cl, s1)
        a1g = a1 / (-e1)

        s2, c2, n2, a2 = apply_v(cl, s1, c1, n1, a1g)
        e2 = ising_energy(cl, s2)
        in_ice = e2 == 0.0
        if in_ice.any():
            targets = np.array([cl.ice_index[int(x)] for x in s2[in_ice]], dtype=np.int64)
            h2_parts.append(dedupe_projected(s2[in_ice], c2[in_ice], n2[in_ice], a2[in_ice], targets))

        mid = ~in_ice
        if mid.any():
            s2q, c2q, n2q = s2[mid], c2[mid], n2[mid]
            a2q = a2[mid] / (-e2[mid])
            s3, c3, n3, a3 = apply_v(cl, s2q, c2q, n2q, a2q)
            e3 = ising_energy(cl, s3)
            in3 = e3 == 0.0
            if in3.any():
                targets = np.array([cl.ice_index[int(x)] for x in s3[in3]], dtype=np.int64)
                h3_parts.append(dedupe_projected(s3[in3], c3[in3], n3[in3], a3[in3], targets))
        if verbose and col % 250 == 0:
            print(f"    SW column {col}/{cl.n_ice}", flush=True)
    return {"H2": merge_rows(h2_parts), "H3": merge_rows(h3_parts)}


def transport_delta(cl: Cluster, rows) -> np.ndarray:
    pos4 = np.rint(4 * cl.positions).astype(np.int64)
    L4 = np.rint(4 * cl.Lvecs).astype(np.int64)
    src = cl.ice_states[rows["s"]]
    dst = cl.ice_states[rows["t"]]
    raised = dst & ~src
    lowered = src & ~dst
    bits = np.uint64(1) << np.arange(cl.n_sites, dtype=np.uint64)
    rmask = (raised[:, None] & bits[None, :]) != 0
    lmask = (lowered[:, None] & bits[None, :]) != 0
    rho = rmask.astype(np.int64) @ pos4 - lmask.astype(np.int64) @ pos4
    return rho + rows["N"] @ L4


def transport_character(cl: Cluster, rows) -> np.ndarray:
    """Return the integer source character q=2*delta for completed paths."""
    twice_character = transport_delta(cl, rows)
    if np.any(twice_character % 2):
        raise RuntimeError("completed ice-to-ice path has noninteger character")
    return twice_character // 2


def assemble(cl: Cluster, pt, jpm: float, mode: str) -> np.ndarray:
    H = np.zeros((cl.n_ice, cl.n_ice), dtype=float)
    for key, power in (("H2", 2), ("H3", 3)):
        rows = pt[key]
        if mode == "all":
            keep = np.ones(len(rows["c"]), dtype=bool)
        elif mode == "delta0":
            keep = (transport_delta(cl, rows) == 0).all(axis=1)
        else:
            raise ValueError(mode)
        vals = (jpm ** power) * rows["c"][keep]
        np.add.at(H, (rows["t"][keep], rows["s"][keep]), vals)
    return 0.5 * (H + H.T)


def specific_heat(E, T):
    E = np.asarray(E, dtype=float)
    E = E - E.min()
    beta = 1.0 / T[:, None]
    w = np.exp(-beta * E[None, :])
    Z = w.sum(axis=1)
    e1 = (w * E[None, :]).sum(axis=1) / Z
    e2 = (w * E[None, :] ** 2).sum(axis=1) / Z
    return (e2 - e1**2) / T**2


def refined_peak(T, C):
    k = int(np.argmax(C))
    if 0 < k < len(T) - 1:
        x = np.log(T[k - 1 : k + 2])
        y = C[k - 1 : k + 2]
        den = y[0] - 2 * y[1] + y[2]
        if abs(den) > 1e-30:
            return float(np.exp(x[1] + 0.5 * (y[0] - y[2]) / den * (x[1] - x[0])))
    return float(T[k])


def cycle_summary(cycles):
    return dict(
        total=len(cycles),
        contractible=sum(1 for _, w in cycles if tuple(w) == (0, 0, 0)),
        by_abs_w={str(k): v for k, v in sorted(Counter(tuple(sorted(abs(x) for x in w)) for _, w in cycles).items())},
    )


def row_survival(cl: Cluster, pt):
    out = {}
    for key in ("H2", "H3"):
        rows = pt[key]
        d0 = (transport_delta(cl, rows) == 0).all(axis=1)
        even = (rows["N"] % 2 == 0).all(axis=1)
        out[key] = {
            "rows": int(len(rows["c"])),
            "abs_coeff_all": float(np.sum(np.abs(rows["c"]))),
            "abs_coeff_corner_even": float(np.sum(np.abs(rows["c"][even]))),
            "abs_coeff_delta0": float(np.sum(np.abs(rows["c"][d0]))),
        }
    return out


def path_mask(path):
    mask = 0
    for site in path:
        mask |= 1 << int(site)
    return mask


def channel_survival(cl: Cluster, pt):
    loop4_by_mask = {path_mask(path): tuple(w) for path, w in cl.loops4}
    hex_by_mask = {path_mask(path): tuple(w) for path, w in cl.hexes}
    out = {}
    ice = cl.ice_states
    for key, nbits, known in (("H2", 4, loop4_by_mask), ("H3", 6, hex_by_mask)):
        rows = pt[key]
        if len(rows["c"]) == 0:
            continue
        character = transport_character(cl, rows)
        d0 = (character == 0).all(axis=1)
        even = (rows["N"] % 2 == 0).all(axis=1)
        finite_grid = {
            size: (character % size == 0).all(axis=1) for size in (2, 3, 4)
        }
        diff = ice[rows["s"]] ^ ice[rows["t"]]
        offdiag = rows["s"] != rows["t"]
        stats = defaultdict(
            lambda: {
                "masks": 0,
                "corner": [],
                "delta0": [],
                "n0": [],
                "finite_grid": {size: [] for size in (2, 3, 4)},
            }
        )
        for mask in sorted(set(int(x) for x in diff[offdiag])):
            if mask.bit_count() != nbits or mask not in known:
                continue
            select = offdiag & (diff == np.uint64(mask))
            total = float(np.sum(np.abs(rows["c"][select])))
            if total == 0:
                continue
            winding = known[mask]
            kind = "contractible" if winding == (0, 0, 0) else "wrapping"
            label = f"{'4_loop' if key == 'H2' else 'hexagon'}_{kind}"
            stats[label]["masks"] += 1
            stats[label]["corner"].append(float(np.sum(np.abs(rows["c"][select & even])) / total))
            stats[label]["delta0"].append(float(np.sum(np.abs(rows["c"][select & d0])) / total))
            n0 = (rows["N"] == 0).all(axis=1)
            stats[label]["n0"].append(float(np.sum(np.abs(rows["c"][select & n0])) / total))
            for size, keep in finite_grid.items():
                retained = np.sum(np.abs(rows["c"][select & keep])) / total
                stats[label]["finite_grid"][size].append(float(retained))
        out[key] = {}
        for label, vals in sorted(stats.items()):
            out[key][label] = {
                "masks": vals["masks"],
                "finite_grid_terms": {
                    f"M{size}": int(np.count_nonzero(np.asarray(retained) > 1.0e-12))
                    for size, retained in vals["finite_grid"].items()
                },
                "corner_mean": float(np.mean(vals["corner"])),
                "corner_min": float(np.min(vals["corner"])),
                "corner_max": float(np.max(vals["corner"])),
                "n0_mean": float(np.mean(vals["n0"])),
                "delta0_mean": float(np.mean(vals["delta0"])),
                "delta0_min": float(np.min(vals["delta0"])),
                "delta0_max": float(np.max(vals["delta0"])),
                "finite_grid_retained": {
                    f"M{size}": float(np.mean(retained))
                    for size, retained in vals["finite_grid"].items()
                },
            }
    return out


def run_case(basis, shape, jpms, T):
    t0 = time.time()
    cl = build_cluster(basis, shape)
    print(
        f"\n=== {basis} {shape}: N={cl.n_sites}, bonds={len(cl.bonds)}, "
        f"tets={len(cl.tets)}, ice={cl.n_ice} ===",
        flush=True,
    )
    print(f"  4-cycles: {cycle_summary(cl.loops4)}", flush=True)
    print(f"  hexagons: {cycle_summary(cl.hexes)}", flush=True)
    pt = sw_order23(cl)
    print(f"  rows: H2={len(pt['H2']['c'])}, H3={len(pt['H3']['c'])}", flush=True)
    curves = []
    for jpm in jpms:
        for mode in ("all", "delta0"):
            H = assemble(cl, pt, jpm, mode)
            E = np.linalg.eigvalsh(H)
            C = specific_heat(E, T)
            Tpk = refined_peak(T, C)
            g4 = 4 * jpm * jpm
            ghex = 12 * abs(jpm) ** 3
            rec = {
                "basis": basis,
                "shape": shape,
                "Jpm": jpm,
                "mode": mode,
                "Tpk": Tpk,
                "Tpk_over_g4": Tpk / g4,
                "Tpk_over_ghex": Tpk / ghex,
                "g4": g4,
                "ghex": ghex,
                "gap": float(np.sort(E)[1] - np.sort(E)[0]),
            }
            curves.append(rec)
            print(
                f"  Jpm={jpm:+.3f} {mode:6s}: Tpk={Tpk:.6g}, "
                f"Tpk/g4={Tpk/g4:.3g}, Tpk/ghex={Tpk/ghex:.3g}",
                flush=True,
            )
    return {
        "cluster": {
            "basis": basis,
            "shape": shape,
            "n_sites": cl.n_sites,
            "n_bonds": int(len(cl.bonds)),
            "n_tets": int(len(cl.tets)),
            "n_ice": int(cl.n_ice),
            "loops4": cycle_summary(cl.loops4),
            "hexagons": cycle_summary(cl.hexes),
            "elapsed_s": time.time() - t0,
        },
        "row_survival": row_survival(cl, pt),
        "channel_survival": channel_survival(cl, pt),
        "curves": curves,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("notes/recomputed_finite_size_artifact.json"))
    ap.add_argument("--nt", type=int, default=1200)
    ap.add_argument(
        "--clusters",
        choices=["all", "cubic", "fcc"],
        default="all",
        help="which cluster set to run",
    )
    args = ap.parse_args()
    T = np.geomspace(1e-4, 0.3, args.nt)
    cases = []
    if args.clusters in ("all", "cubic"):
        cases.append(run_case("cubic", (1, 1, 1), [-0.10, -0.05, 0.05], T))
    if args.clusters in ("all", "fcc"):
        cases.append(run_case("fcc", (2, 2, 2), [-0.10, -0.05, 0.05], T))
    results = {
        "description": "Independent no-git-history recomputation from notes/recompute_finite_size_artifact.py",
        "T_grid": {"min": float(T[0]), "max": float(T[-1]), "n": int(len(T))},
        "cases": cases,
    }
    args.out.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
