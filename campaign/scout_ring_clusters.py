"""Which pyrochlore cluster can actually host a dispersing ring-exchange photon?

FCC-32 cannot: the connected part of its zero-flux sector is an exactly
8-regular graph, so the uniform state is an exact eigenvector, the RK potential
is a constant, and the spectrum is integer and non-dispersive (omega(X)/omega(L)
= 2 against 1.155 for a linear photon).  That is a property of the cluster's
ice-manifold graph, not of the physics.

This scouts alternatives WITHOUT diagonalising anything.  For each candidate it
reports the two things that decide usability:

  * REGULARITY.  Spread of n_flippable over the ice states of the dominant
    sector.  A constant means the cluster is at an RK-like point for every V
    and has no photon.  We want a broad distribution.
  * MOMENTUM RESOLUTION.  Number of primitive cells = number of cluster
    momenta.  FCC-32 gives 8 (Gamma + 3 X + 4 L, all zone boundary).  More
    momenta is the only way to see dispersion.

Also reports ice-manifold size so the diagonalisation cost is known before
committing.  The pure ring-exchange model needs no virtual space, so the cost
is set by the ice manifold alone -- far cheaper than the XXZ fold.

Usage: scout_ring_clusters.py            (runs the default candidate list)
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "notes"))
sys.path.insert(0, str(ROOT / "campaign"))

import recompute_finite_size_artifact as geometry              # noqa: E402
from qsi_campaign.protocol import polarization_sector_labels   # noqa: E402

CANDIDATES = [
    ("cubic", (1, 1, 1)),      # 16 sites, the reference small case
    ("fcc",   (2, 2, 2)),      # 32, the known-degenerate one
    ("fcc",   (2, 2, 3)),      # 48
    ("cubic", (2, 1, 1)),      # 32, different shape
    ("fcc",   (2, 2, 4)),      # 64
    ("cubic", (2, 2, 1)),      # 64
]


def contractible(cluster):
    return [[int(s) for s in e[0]] for e in cluster.hexes
            if tuple(int(w) for w in e[1]) == (0, 0, 0)]


def flippable_counts(states, hexes):
    """n_flippable per ice state; the regularity diagnostic."""
    counts = np.zeros(len(states), dtype=np.int32)
    for path in hexes:
        for i, state in enumerate(states):
            bits = [(int(state) >> s) & 1 for s in path]
            if all(bits[j] != bits[(j + 1) % 6] for j in range(6)):
                counts[i] += 1
    return counts


def main() -> int:
    for basis, shape in CANDIDATES:
        t0 = time.perf_counter()
        print("=" * 70, flush=True)
        print(f"{basis}{shape}", flush=True)
        try:
            cluster = geometry.build_cluster(basis, shape)
        except Exception as exc:
            print(f"  build FAILED: {exc}", flush=True)
            continue
        n_sites = int(cluster.n_sites)
        ice = np.asarray(cluster.ice_states, dtype=np.uint64)
        hexes = contractible(cluster)
        n_cells = int(np.prod(shape)) * (4 if basis == "cubic" else 1)
        print(f"  {n_sites} sites, {len(ice):,} ice states, "
              f"{len(cluster.hexes)} six-cycles / {len(hexes)} contractible, "
              f"~{n_cells} momenta  ({time.perf_counter()-t0:.0f} s)",
              flush=True)

        if len(ice) > 400_000:
            print("  ice manifold too large to profile here; "
                  "sizes reported only", flush=True)
            continue

        labels = polarization_sector_labels(cluster)
        sizes = np.bincount(labels)
        big = int(np.argmax(sizes))
        sel = labels == big
        print(f"  {len(sizes)} transport sectors, largest = {sizes[big]} "
              f"(sector {big})", flush=True)

        counts = flippable_counts(ice[sel], hexes)
        uniq, mult = np.unique(counts, return_counts=True)
        spread = counts.max() - counts.min()
        print(f"  n_flippable in largest sector: mean {counts.mean():.3f}, "
              f"min {counts.min()}, max {counts.max()}, spread {spread}")
        print("    distribution: " + ", ".join(
            f"{u}:{m}" for u, m in zip(uniq, mult)))
        verdict = ("REGULAR -> RK-degenerate, no photon"
                   if spread == 0 else
                   f"irregular (spread {spread}) -> usable")
        print(f"  VERDICT: {verdict}", flush=True)
        print(f"  ({time.perf_counter()-t0:.0f} s total)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
