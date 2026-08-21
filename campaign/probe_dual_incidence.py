#!/usr/bin/env python3
"""Find the plaquette -> dual-site incidence empirically, not by assumption.

The distance-based guess (four plaquettes at a dual site are mutually nearest)
failed its own check: one clique instead of sixteen.  Rather than guess again,
enumerate the candidate relations and report which one produces the structure
the dual diamond lattice must have -- n/2 sites, four plaquettes each, every
plaquette in exactly two.
"""

from __future__ import annotations

import sys
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "notes"))
sys.path.insert(0, str(ROOT / "campaign"))

import recompute_finite_size_artifact as geometry  # noqa: E402


def contractible(cluster):
    out = []
    positions = np.asarray(cluster.positions, dtype=float)
    for path, winding in cluster.hexes:
        if any(int(v) != 0 for v in winding):
            continue
        sites = frozenset(int(v) for v in path)
        out.append((sites, positions[sorted(sites)].mean(axis=0)))
    return out


def report(cluster, name):
    hexes = contractible(cluster)
    count = len(hexes)
    print(f"\n=== {name}: {count} contractible plaquettes, "
          f"expect {count // 2} dual sites x 4")

    shared = Counter()
    for (a, _), (b, _) in combinations(hexes, 2):
        shared[len(a & b)] += 1
    print(f"  shared-site histogram over plaquette pairs: {dict(sorted(shared.items()))}")

    centres = np.array([c for _, c in hexes])
    lvecs = np.asarray(cluster.Lvecs, dtype=float)
    separation = np.full((count, count), np.inf)
    for shift in np.ndindex(3, 3, 3):
        offset = (np.asarray(shift, dtype=float) - 1.0) @ lvecs
        candidate = np.linalg.norm(
            centres[:, None, :] - centres[None, :, :] - offset, axis=2)
        separation = np.minimum(separation, candidate)
    off = separation[~np.eye(count, dtype=bool)]
    shells = np.unique(np.round(off, 6))
    print(f"  centre-separation shells: "
          f"{[float(v) for v in shells[:5]]}")
    for radius in shells[:4]:
        adjacency = np.isclose(separation, radius) & ~np.eye(count, dtype=bool)
        print(f"    r={radius:.4f}: degree "
              f"{adjacency.sum(axis=1).min()}-{adjacency.sum(axis=1).max()}")

    # candidate relation: plaquettes sharing exactly k sites
    for k in sorted(shared):
        if k == 0:
            continue
        adjacency = np.zeros((count, count), dtype=bool)
        for i, (a, _) in enumerate(hexes):
            for j, (b, _) in enumerate(hexes):
                if i != j and len(a & b) == k:
                    adjacency[i, j] = True
        degree = adjacency.sum(axis=1)
        cliques = set()
        for a in range(count):
            neighbours = np.flatnonzero(adjacency[a])
            for trio in combinations(neighbours[neighbours > a], 3):
                if all(adjacency[x, y] for x, y in combinations(trio, 2)):
                    cliques.add(tuple(sorted((a,) + trio)))
        membership = Counter(h for clique in cliques for h in clique)
        good = (len(cliques) == count // 2
                and set(membership.values()) == {2}
                and len(membership) == count)
        print(f"  share {k} sites: degree {degree.min()}-{degree.max()}, "
              f"{len(cliques)} 4-cliques, membership "
              f"{sorted(set(membership.values()))}"
              f"  {'<-- MATCHES dual diamond' if good else ''}")


def main() -> int:
    for name, spec in (("cubic16", ("cubic", (1, 1, 1))),
                       ("fcc32", ("fcc", (2, 2, 2)))):
        report(geometry.build_cluster(*spec), name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
