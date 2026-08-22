"""Pyrochlore lattices large enough for quantum Monte Carlo.

The exact-diagonalization code finds hexagons by enumerating every
six-cycle of the adjacency graph and filtering.  That is fine at 96
sites and hopeless at the thousands of sites needed to resolve a dense
momentum grid.

The lattice is periodic, so the hexagons are not independent: they fall
into a handful of translation classes.  We therefore extract those
classes ONCE from a cluster large enough that no hexagon wraps the
torus, expressing each as a list of (cell offset, sublattice) pairs,
and then generate the hexagons of any supercell by translation.  The
same is done for tetrahedra.  Both are validated by regenerating the
48- and 72-site clusters and comparing set-for-set against the
cycle-search result the ED work is built on.

Conventions match make_positions("fcc", shape): site index is
4*cell + sublattice, cells ordered (i,j,k) with k fastest, primitive
vectors a1=(1/2,1/2,0), a2=(1/2,0,1/2), a3=(0,1/2,1/2), and the
four-site basis (0,0,0), (0,1/4,1/4), (1/4,0,1/4), (1/4,1/4,0).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "notes"))
sys.path.insert(0, str(ROOT / "campaign"))

import recompute_finite_size_artifact as geometry               # noqa: E402
from run_ring48_dssf import contractible                        # noqa: E402

FCC_A = np.array([[0.5, 0.5, 0.0], [0.5, 0.0, 0.5], [0.0, 0.5, 0.5]])
SUBLAT = np.array([[0.0, 0.0, 0.0], [0.0, 0.25, 0.25],
                   [0.25, 0.0, 0.25], [0.25, 0.25, 0.0]])


def _build_no_states(shape):
    orig = geometry.enumerate_ice_states
    geometry.enumerate_ice_states = lambda a, b: np.array([0], dtype=np.uint64)
    cl = geometry.build_cluster("fcc", tuple(shape))
    geometry.enumerate_ice_states = orig
    return cl


def cell_of(site, shape):
    c, sub = divmod(int(site), 4)
    n1, n2, n3 = shape
    k = c % n3
    j = (c // n3) % n2
    i = c // (n3 * n2)
    return np.array([i, j, k]), sub


def index_of(cell, sub, shape):
    n1, n2, n3 = shape
    i, j, k = [int(x) % n for x, n in zip(cell, shape)]
    return 4 * ((i * n2 + j) * n3 + k) + int(sub)


def _minimal_image(d, shape):
    sh = np.asarray(shape)
    return ((np.asarray(d) + sh // 2) % sh) - sh // 2


def extract_templates(shape=(3, 3, 3)):
    """Translation classes of tetrahedra and hexagons, as offset lists.

    Taken from a cluster wide enough (>=3 cells per direction) that a
    contractible hexagon never wraps, so the cell offsets are
    unambiguous.
    """
    cl = _build_no_states(shape)
    hexes = contractible(cl)
    tets = [tuple(int(s) for s in t) for t in cl.tets]

    def classify(path, keep_order):
        cs = [cell_of(s, shape) for s in path]
        base = cs[0][0]
        ent = []
        for cell, sub in cs:
            d = _minimal_image(cell - base, shape)
            ent.append((int(d[0]), int(d[1]), int(d[2]), int(sub)))
        # shift so the first entry sits at the origin cell already true
        if keep_order:
            return tuple(ent)
        return tuple(sorted(ent))

    def canon(ent):
        """Canonical form of a cycle, invariant under rotation AND
        reversal.  Without the reversal the two traversal directions of
        one hexagon become distinct classes and every hexagon is
        generated twice -- which a set-based comparison silently hides
        but which would double every ring term in the Hamiltonian."""
        best = None
        for seq in (ent, tuple(reversed(ent))):
            m = len(seq)
            for r in range(m):
                rot = seq[r:] + seq[:r]
                base = np.array(rot[0][:3])
                shifted = tuple((int(a - base[0]), int(b - base[1]),
                                 int(c - base[2]), int(s))
                                for a, b, c, s in rot)
                if best is None or shifted < best:
                    best = shifted
        return best

    htemp, ttemp = {}, {}
    for path in hexes:
        ent = classify(path, True)
        htemp.setdefault(canon(ent), ent)
    for t in tets:
        ent = classify(t, False)
        base = np.array(ent[0][:3])
        shifted = tuple((int(a - base[0]), int(b - base[1]),
                         int(c - base[2]), int(s))
                        for a, b, c, s in ent)
        ttemp.setdefault(shifted, shifted)
    return list(htemp.values()), list(ttemp.values())


class PyroLattice:
    """A periodic pyrochlore supercell with hexagons built by translation."""

    def __init__(self, shape, htemplates, ttemplates):
        self.shape = tuple(int(x) for x in shape)
        n1, n2, n3 = self.shape
        self.n_cells = n1 * n2 * n3
        self.n_sites = 4 * self.n_cells
        cells = np.array([[i, j, k] for i in range(n1) for j in range(n2)
                          for k in range(n3)], dtype=float)
        self.positions = np.concatenate(
            [(cells @ FCC_A)[:, None, :] + SUBLAT[None, :, :]], axis=1
        ).reshape(-1, 3)
        self.sublattice = np.tile(np.arange(4), self.n_cells)
        self.hexes = self._generate(htemplates)
        self.tets = self._generate(ttemplates)

    def _generate(self, templates):
        out = np.empty((self.n_cells * len(templates), len(templates[0])),
                       dtype=np.int64)
        r = 0
        n1, n2, n3 = self.shape
        for i in range(n1):
            for j in range(n2):
                for k in range(n3):
                    base = np.array([i, j, k])
                    for t in templates:
                        out[r] = [index_of(base + np.array(e[:3]), e[3],
                                           self.shape) for e in t]
                        r += 1
        return out


def build_lattice(shape, templates=None):
    if templates is None:
        templates = extract_templates()
    return PyroLattice(shape, templates[0], templates[1])


def _validate(shape, templates):
    """Regenerate a cluster and compare against the cycle-search result."""
    lat = build_lattice(shape, templates)
    cl = _build_no_states(shape)
    ref_h = {frozenset(int(s) for s in p) for p in contractible(cl)}
    got_h = {frozenset(int(s) for s in row) for row in lat.hexes}
    ref_t = {frozenset(int(s) for s in t) for t in cl.tets}
    got_t = {frozenset(int(s) for s in row) for row in lat.tets}
    pos_ok = np.allclose(np.sort(lat.positions, axis=0),
                         np.sort(np.asarray(cl.positions), axis=0))
    cnt_ok = (len(lat.hexes) == lat.n_sites
                and len(lat.tets) == lat.n_sites // 2)
    print(f"  fcc{shape}: {lat.n_sites} sites; generated rows "
          f"{len(lat.hexes)} hexagons / {len(lat.tets)} tetrahedra "
          f"{'OK' if cnt_ok else 'WRONG COUNT'}")
    print(f"    hexagons  generated {len(got_h)}  reference {len(ref_h)}  "
          f"{'MATCH' if got_h == ref_h else 'MISMATCH'}")
    print(f"    tetrahedra generated {len(got_t)}  reference {len(ref_t)}  "
          f"{'MATCH' if got_t == ref_t else 'MISMATCH'}")
    print(f"    positions {'MATCH' if pos_ok else 'MISMATCH'}")
    return got_h == ref_h and got_t == ref_t and pos_ok and cnt_ok


def main() -> int:
    import time
    t0 = time.perf_counter()
    templates = extract_templates()
    print(f"templates: {len(templates[0])} hexagon classes, "
          f"{len(templates[1])} tetrahedron classes "
          f"({time.perf_counter()-t0:.1f} s)")
    ok = True
    for shape in [(2, 2, 3), (2, 2, 4), (2, 3, 3), (3, 3, 3)]:
        ok &= _validate(shape, templates)
    print(f"\nvalidation {'PASSED' if ok else 'FAILED'}")
    for shape in [(4, 4, 4), (6, 6, 6), (8, 8, 8)]:
        t1 = time.perf_counter()
        lat = build_lattice(shape, templates)
        print(f"  fcc{shape}: {lat.n_sites:,} sites, "
              f"{len(lat.hexes):,} hexagons, {len(lat.tets):,} tetrahedra "
              f"({time.perf_counter()-t1:.1f} s)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
