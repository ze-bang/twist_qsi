#!/usr/bin/env python3
"""Does the hexagon ring exchange stay inside a polarization sector?

The whole masked construction rests on it: the transport-masked Hamiltonian is
the sector-block-diagonal part, and that is only the right object if the
physical ring exchange does not connect sectors.  The flux diagnostic measured
10-28% of flippable ice weight landing outside its own sector block, which is
either a real violation or a bookkeeping error.  This settles it combinatorially
-- no eigenvectors, no truncation, just labels before and after a flip.

It also checks a specific hazard.  `polarization_sector_labels` documents that
its labels "follow the ordering of `cluster.ice_states`", but the consumers do

    labels = polarization_sector_labels(cluster)
    ice_states = np.sort(np.asarray(cluster.ice_states, dtype=np.uint64))
    ice_rows = np.searchsorted(space, ice_states)
    rows = np.concatenate([ice_rows[labels == sector], non_ice])

which pairs the labels with a SORTED copy.  If `cluster.ice_states` is not
already sorted those two orderings disagree and every sector block is built
from the wrong states -- silently, since the blocks stay the right size.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "notes"))
sys.path.insert(0, str(ROOT / "campaign"))

import recompute_finite_size_artifact as geometry  # noqa: E402
from qsi_campaign.protocol import polarization_sector_labels  # noqa: E402

CLUSTERS = {"cubic16": ("cubic", (1, 1, 1)), "fcc32": ("fcc", (2, 2, 2))}


def main() -> int:
    for name in ("cubic16", "fcc32"):
        cluster = geometry.build_cluster(*CLUSTERS[name])
        raw = np.asarray(cluster.ice_states, dtype=np.uint64)
        labels = polarization_sector_labels(cluster)
        sorted_states = np.sort(raw)
        is_sorted = bool(np.array_equal(raw, sorted_states))
        print(f"\n=== {name}: {len(raw)} ice states, "
              f"{len(np.unique(labels))} sectors")
        print(f"  cluster.ice_states already sorted? {is_sorted}")
        if not is_sorted:
            moved = int(np.sum(raw != sorted_states))
            print(f"  !! {moved}/{len(raw)} entries differ between "
                  f"cluster.ice_states and its sort -- any consumer pairing "
                  f"`labels` with the sorted array is mislabelling states")

        # labels keyed by the state itself, which is order-independent
        label_of = {int(s): int(v) for s, v in zip(raw, labels)}

        leaks = 0
        flips = 0
        per_hexagon = []
        for path, _ in cluster.hexes:
            sites = [int(v) for v in path]
            mask = 0
            for site in sites:
                mask |= 1 << site
            local = 0
            local_flips = 0
            for state in raw:
                bits = [(int(state) >> site) & 1 for site in sites]
                if any(bits[k] == bits[(k + 1) % len(bits)]
                       for k in range(len(bits))):
                    continue                       # not flippable
                local_flips += 1
                target = int(state) ^ mask
                if target not in label_of:
                    local += 1                     # left the ice manifold
                elif label_of[target] != label_of[int(state)]:
                    local += 1                     # changed sector
            flips += local_flips
            leaks += local
            per_hexagon.append((local, local_flips))

        print(f"  {len(cluster.hexes)} hexagons, {flips} flippable "
              f"(state, hexagon) pairs")
        print(f"  sector-changing or ice-leaving flips: {leaks}")
        if leaks:
            worst = max(per_hexagon, key=lambda row: row[0])
            print(f"  !! ring exchange does NOT close on a sector; "
                  f"worst hexagon leaks {worst[0]}/{worst[1]}")
        else:
            print("  ring exchange closes on every polarization sector: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
