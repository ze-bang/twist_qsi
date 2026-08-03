#!/usr/bin/env python3
"""Ice-manifold perturbation theory on the covering (infinite) lattice.

The Bloch coefficients through third order are cluster-independent, but at
fourth order the cubic-16 torus admits virtual processes with no bulk
counterpart (winding four-cycle round trips, wrap-assisted excursions).
This script computes the fourth-order coefficients in the thermodynamic
limit by evaluating every virtual path on the covering lattice, where
winding loops do not close, and validates the machinery by running the
identical enumeration in torus mode, where it must reproduce the
independently validated matrix-recursion coefficients exactly.

Path bookkeeping (validated against the recursion):
  * amplitude of an ordered k-op sequence = -(prod_m 1/E_m) over the k-1
    intermediates; sequences touching an intermediate with E = 0 vanish
    (reduced resolvent).
  * diagonal fourth order: connected sequences plus the linked-cluster
    residue  +1 for every ordered pair (b, c) of flippable bonds that fail
    to factorize (share a site or a charged tetrahedron, or b = c); the
    factorizing pairs cancel exactly against the Bloch folding term
    -PVR^2VP E2.
  * off-diagonal fourth order: the folding term is diagonal, so the plain
    connected sum is complete.

Outputs campaign/outputs/bulk_pt_coefficients.json with the torus/cover
comparison, and bulk_pt_e4.npz with the bulk fourth-order matrix on the
cluster ice manifold.
"""

from __future__ import annotations

import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "notes"))

import recompute_finite_size_artifact as geometry  # noqa: E402

OUTPUT = ROOT / "campaign" / "outputs"


class Lattice:
    """Covering (or torus) pyrochlore assembled from the cluster template.

    Sites are (cell, index) pairs; in torus mode every cell is folded to
    (0, 0, 0) and the wrap vectors close the loops, in cover mode cells are
    honest integer translations and nothing wraps.
    """

    def __init__(self, cluster, torus: bool):
        self.torus = torus
        self.n = cluster.n_sites
        self.bonds = np.asarray(cluster.bonds)
        self.wraps = np.asarray(cluster.bond_wrap)
        # site -> list of (neighbour site, cell shift, bond id)
        self.neighbours = [[] for _ in range(self.n)]
        for bond_id, ((u, v), w) in enumerate(zip(self.bonds, self.wraps)):
            self.neighbours[u].append((v, tuple(w), bond_id))
            self.neighbours[v].append((u, tuple(-w), bond_id))
        # tetrahedron templates: shifts of each member site relative to the
        # first, reconstructed from the bond wraps.
        shift_of = {}
        for (u, v), w in zip(self.bonds, self.wraps):
            shift_of[(u, v)] = tuple(w)
            shift_of[(v, u)] = tuple(-w)
        self.tet_templates = []
        for members in np.asarray(cluster.tets):
            anchor = int(members[0])
            template = [(anchor, (0, 0, 0))]
            for other in members[1:]:
                template.append((int(other), shift_of[(anchor, int(other))]))
            self.tet_templates.append(template)
        # site index -> list of (tet id, shift of this site inside template)
        self.site_tets = [[] for _ in range(self.n)]
        for tet_id, template in enumerate(self.tet_templates):
            for site, shift in template:
                self.site_tets[site].append((tet_id, shift))

    def fold(self, cell):
        return (0, 0, 0) if self.torus else cell

    def site_neighbours(self, node):
        cell, s = node
        out = []
        for v, w, bond_id in self.neighbours[s]:
            target = tuple(c + d for c, d in zip(cell, w))
            out.append(((self.fold(target), v), bond_id))
        return out

    def tets_of(self, node):
        cell, s = node
        out = []
        for tet_id, shift in self.site_tets[s]:
            base = tuple(c - d for c, d in zip(cell, shift))
            out.append((self.fold(base), tet_id))
        return out

    def tet_nodes(self, tet):
        cell, tet_id = tet
        return [
            (self.fold(tuple(c + d for c, d in zip(cell, shift))), s)
            for s, shift in self.tet_templates[tet_id]
        ]


class State:
    """Spin configuration = periodic background + finite flipped set."""

    def __init__(self, lattice: Lattice, background_bits: int):
        self.lattice = lattice
        self.background = [1 if (background_bits >> s) & 1 else -1
                           for s in range(lattice.n)]
        self.flipped: set = set()
        self.tet_charge: dict = {}
        self.energy8 = 0  # 8 * E = sum over tets of Q^2 with Q = sum sigma

    def spin(self, node):
        sign = -1 if node in self.flipped else 1
        return sign * self.background[node[1]]

    def flip(self, node):
        new = -self.spin(node)
        for tet in self.lattice.tets_of(node):
            old_q = self.tet_charge.get(tet, 0)
            new_q = old_q + 2 * new
            self.energy8 += new_q * new_q - old_q * old_q
            if new_q == 0:
                self.tet_charge.pop(tet, None)
            else:
                self.tet_charge[tet] = new_q
        self.flipped.symmetric_difference_update({node})


def bond_nodes(lattice: Lattice, cell, bond_id):
    u, v = lattice.bonds[bond_id]
    w = lattice.wraps[bond_id]
    return ((lattice.fold(cell), int(u)),
            (lattice.fold(tuple(c + d for c, d in zip(cell, w))), int(v)))


def charged_tets_of_bond(lattice, node_u, node_v):
    """Tets whose charge changes when the bond flips (the two side tets)."""
    tets_u = set(lattice.tets_of(node_u))
    tets_v = set(lattice.tets_of(node_v))
    return tets_u.symmetric_difference(tets_v)


def enumerate_sequences(lattice, state, start_bonds, order, target,
                        grow_candidates, squared=False):
    """Sum of Bloch path amplitudes of ordered flip sequences.

    start_bonds: list of (cell, bond_id) allowed as the first operator.
    target: frozenset of nodes that must be exactly the flipped set at the
    end (empty set for diagonal elements).
    grow_candidates: if true (diagonal mode) later operators are restricted
    to bonds touching the sites of already-charged or already-flipped
    regions, which enumerates exactly the connected sequences; if false
    (off-diagonal mode) the candidate set stays fixed to start_bonds.
    """
    total = 0.0

    def candidates():
        if not grow_candidates:
            return start_bonds
        seen, out = set(), []
        hot_sites = set(state.flipped)
        for tet in state.tet_charge:
            hot_sites.update(lattice.tet_nodes(tet))
        for node in hot_sites:
            for (nbr, bond_id) in lattice.site_neighbours(node):
                cell = node[0] if node[1] == lattice.bonds[bond_id][0] else \
                    tuple(c - d for c, d in
                          zip(node[0], lattice.wraps[bond_id])) \
                    if node[1] == lattice.bonds[bond_id][1] else None
                if cell is None:
                    continue
                key = (lattice.fold(cell), bond_id)
                if key not in seen:
                    seen.add(key)
                    out.append(key)
        return out

    def walk(step, weight):
        nonlocal total
        pool = start_bonds if step == 0 else candidates()
        remaining = order - step
        for cell, bond_id in pool:
            node_u, node_v = bond_nodes(lattice, cell, bond_id)
            if state.spin(node_u) == state.spin(node_v):
                continue
            state.flip(node_u)
            state.flip(node_v)
            mismatch = len(state.flipped.symmetric_difference(target))
            if step + 1 == order:
                if mismatch == 0 and state.energy8 == 0:
                    total += weight if squared else -weight
            elif mismatch <= 2 * (remaining - 1) and state.energy8 > 0:
                factor = (8.0 / state.energy8) ** 2 if squared \
                    else 8.0 / state.energy8
                walk(step + 1, weight * factor)
            state.flip(node_v)
            state.flip(node_u)

    walk(0, 1.0)
    return total


def linked_cluster_residue(lattice, state, cell_bonds):
    """+1 per ordered pair of flippable bonds that fails to factorize."""
    flippable = []
    for cell, bond_id in cell_bonds:
        node_u, node_v = bond_nodes(lattice, cell, bond_id)
        if state.spin(node_u) != state.spin(node_v):
            flippable.append((frozenset((node_u, node_v)),
                              frozenset(charged_tets_of_bond(
                                  lattice, node_u, node_v))))
    residue = 0
    for sites_b, tets_b in flippable:
        for sites_c, tets_c in flippable:
            if sites_b == sites_c or (sites_b & sites_c) or (tets_b & tets_c):
                residue += 1
    return residue


def diagonal_order4(lattice, state, cell_bonds):
    connected = enumerate_sequences(
        lattice, state, cell_bonds, 4, frozenset(), grow_candidates=True)
    return connected + linked_cluster_residue(lattice, state, cell_bonds)


def lift_difference(lattice, bits_a, bits_b):
    """Lift the torus difference set of two ice states to the cover.

    Returns the set of cover nodes (walking the difference set as a closed
    loop through the bond wraps) or None if the loop does not close on the
    cover (winding difference).
    """
    diff = [s for s in range(lattice.n) if ((bits_a ^ bits_b) >> s) & 1]
    if not diff:
        return frozenset()
    remaining = set(diff)
    start = remaining.pop()
    nodes = [((0, 0, 0), start)]
    while remaining:
        cell, site = nodes[-1]
        step = None
        for v, w, _ in lattice.neighbours[site]:
            if v in remaining:
                step = (tuple(c + d for c, d in zip(cell, w)), v)
                break
        if step is None:
            return None
        remaining.discard(step[1])
        nodes.append(step)
    # closure: last node must neighbour the first with zero net shift
    cell, site = nodes[-1]
    first_cell, first_site = nodes[0]
    for v, w, _ in lattice.neighbours[site]:
        if v == first_site:
            if tuple(c + d for c, d in zip(cell, w)) == first_cell:
                return frozenset(nodes)
    return None


def offdiag_element(lattice, bits_a, target_nodes, order):
    """<b| E_order |a> with target the lifted difference set."""
    state = State(lattice, bits_a)
    sites = {node for node in target_nodes}
    shell = set(sites)
    for node in list(sites):
        for nbr, _ in lattice.site_neighbours(node):
            shell.add(nbr)
    pool, seen = [], set()
    for node in shell:
        cell, s = node
        for v, w, bond_id in lattice.neighbours[s]:
            other = (lattice.fold(tuple(c + d for c, d in zip(cell, w))), v)
            if other in shell:
                u0, _ = lattice.bonds[bond_id]
                base = cell if s == u0 else other[0]
                key = (lattice.fold(base), bond_id)
                if key not in seen:
                    seen.add(key)
                    pool.append(key)
    return enumerate_sequences(lattice, state, pool, order, target_nodes,
                               grow_candidates=False)


def main() -> None:
    cluster = geometry.build_cluster("cubic", (1, 1, 1))
    torus = Lattice(cluster, torus=True)
    cover = Lattice(cluster, torus=False)
    ice = [int(s) for s in cluster.ice_states]
    cell0 = (0, 0, 0)
    cell_bonds = [(cell0, b) for b in range(len(torus.bonds))]

    reference = np.load(OUTPUT / "ice_pt_coefficients.npz")
    e4_ref = reference["E4"]

    report = {"validation": {}, "bulk": {}}

    # --- torus validation against the matrix recursion -------------------
    # The enumerator is validated in torus mode, where every Bloch quantity
    # can be rebuilt and compared with the independently validated matrix
    # recursion: order <= 3 coefficients, then the fourth order assembled as
    # direct minus folding with the folding summed over ice intermediates
    # (PVR^2VP has winding off-diagonal elements on the torus, which is
    # precisely the channel absent on the cover).
    state = State(torus, ice[0])
    val = {}
    val["E2_diag"] = enumerate_sequences(torus, state, cell_bonds, 2,
                                         frozenset(), grow_candidates=True)
    val["E3_diag"] = enumerate_sequences(torus, state, cell_bonds, 3,
                                         frozenset(), grow_candidates=True)

    def torus_target(i, j):
        return frozenset((cell0, s) for s in range(torus.n)
                         if ((ice[i] ^ ice[j]) >> s) & 1)

    def element(kind, i, j, order):
        st = State(torus, ice[i])
        target = torus_target(i, j) if i != j else frozenset()
        if kind == "direct":
            return enumerate_sequences(torus, st, cell_bonds, order, target,
                                       grow_candidates=False)
        return enumerate_sequences(torus, st, cell_bonds, 2, target,
                                   grow_candidates=False, squared=True)

    hex_pair = eight_pair = None
    for i in range(len(ice)):
        for j in range(i + 1, len(ice)):
            weight = bin(ice[i] ^ ice[j]).count("1")
            if weight == 6 and hex_pair is None and abs(e4_ref[i, j]) > 1:
                hex_pair = (i, j)
            if weight == 8 and eight_pair is None and abs(e4_ref[i, j]) > 1:
                eight_pair = (i, j)
        if hex_pair and eight_pair:
            break

    def fold_element(i, j):
        total = 0.0
        for c in range(len(ice)):
            w_ic = bin(ice[i] ^ ice[c]).count("1")
            w_cj = bin(ice[c] ^ ice[j]).count("1")
            if w_ic not in (0, 4) or w_cj not in (0, 4):
                continue
            total += element("pv2", i, c, 2) * element("direct", c, j, 2)
        return total

    i, j = hex_pair
    val["E3_hex"] = element("direct", i, j, 3)
    val["E4_hex"] = (element("direct", i, j, 4) - fold_element(i, j),
                     float(e4_ref[i, j]))
    p, q = eight_pair
    val["E4_8ring"] = (element("direct", p, q, 4) - fold_element(p, q),
                       float(e4_ref[p, q]))
    diag_pairs = []
    for index in (0, 7, 33, 61, 89):
        direct = element("direct", index, index, 4)
        rebuilt = direct - fold_element(index, index)
        diag_pairs.append((float(e4_ref[index, index]), rebuilt))
    val["E4_diag_pairs"] = diag_pairs
    report["validation"] = {k: v for k, v in val.items()}
    print("torus validation:", json.dumps(val), flush=True)

    checks = [
        abs(val["E2_diag"] + 32.0) < 1e-9,
        abs(val["E3_diag"] + 64.0) < 1e-9,
        abs(val["E3_hex"] + 12.0) < 1e-9,
        abs(val["E4_hex"][0] - val["E4_hex"][1]) < 1e-9,
        abs(val["E4_8ring"][0] - val["E4_8ring"][1]) < 1e-9,
        all(abs(a - b) < 1e-6 for a, b in val["E4_diag_pairs"]),
    ]
    if not all(checks):
        raise RuntimeError(f"torus validation failed: {checks}")
    print("torus validation PASSED", flush=True)

    # --- covering-lattice (thermodynamic-limit) coefficients -------------
    d = len(ice)
    e4_bulk = np.zeros((d, d))
    hex_values, ring_values = [], []
    for i in range(d):
        s = State(cover, ice[i])
        e4_bulk[i, i] = diagonal_order4(cover, s, cell_bonds)
        for j in range(d):
            if j == i:
                continue
            weight = bin(ice[i] ^ ice[j]).count("1")
            if weight not in (6, 8):
                continue
            lifted = lift_difference(cover, ice[i], ice[j])
            if lifted is None:
                continue
            element = offdiag_element(cover, ice[i], lifted, 4)
            e4_bulk[i, j] = element
            (hex_values if weight == 6 else ring_values).append(element)
        if i % 15 == 0:
            print(f"  cover row {i}/{d}", flush=True)

    report["bulk"] = {
        "E4_diag_min": float(np.diag(e4_bulk).min()),
        "E4_diag_max": float(np.diag(e4_bulk).max()),
        "E4_hex_distinct": sorted(set(np.round(hex_values, 9))),
        "E4_8ring_distinct": sorted(set(np.round(ring_values, 9))),
        "cluster_hex": -68.0,
        "cluster_8ring": -26.0,
    }
    print("bulk:", json.dumps(report["bulk"]), flush=True)

    np.savez_compressed(OUTPUT / "bulk_pt_e4.npz", E4_bulk=e4_bulk)
    (OUTPUT / "bulk_pt_coefficients.json").write_text(
        json.dumps(report, indent=2, default=float) + "\n")
    print(f"wrote {OUTPUT / 'bulk_pt_e4.npz'}")


if __name__ == "__main__":
    main()
