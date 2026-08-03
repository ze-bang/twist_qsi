#!/usr/bin/env python3
"""Compare the counterterm band with the exact masked roots, without a shift.

The cached comparison spectra were assembled by
``full_hilbert_counterterm_spectrum``, which applies ``match_trace``: it slides
the masked band rigidly so that its mean coincides with the mean of the lowest
ninety periodic levels.  That is a defensible convention for splicing a band
into a spectrum whose absolute placement is otherwise undetermined, but it makes
the cached band useless for judging absolute energies -- it introduced an
apparent ground-state offset of up to 13% of the band width.

Here the masked roots are solved directly and compared unshifted.  Two things
are then measured:

  * the ground energy.  At the Brillouin-Wigner fixed point the counterterm's
    ground energy should *be* the lowest masked root, since
    F_{H+C}(z_0) = Delta[F(z_0)] and E_0(H+C) = z_0 together put z_0 in the
    spectrum of the masked map.  This is a sharp, parameter-free prediction.
  * the whole band, level by level, in units of the band width.

Usage: compare_to_reference.py [<jpm> ...]   (default: every cached coupling)
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "notes"))
sys.path.insert(0, str(ROOT / "campaign"))

import recompute_finite_size_artifact as geometry  # noqa: E402
from qsi_campaign.exact_band import (  # noqa: E402
    fixed_magnetization_basis,
    microscopic_character_hamiltonian,
)
from qsi_campaign.downfold import build_blocks, solve_roots  # noqa: E402

SPINE = ROOT / "campaign" / "outputs" / "counterterm_spine"
N_ICE = 90


def main() -> None:
    requested = [float(v) for v in sys.argv[1:] if not v.startswith("--")]
    cluster = geometry.build_cluster("cubic", (1, 1, 1))
    states = fixed_magnetization_basis(cluster.n_sites, cluster.n_sites // 2)
    index = {int(s): i for i, s in enumerate(states)}
    ice_rows = np.asarray([index[int(s)] for s in cluster.ice_states])

    records = []
    print(f"{'jpm':>7} {'roots':>6} {'E0 counterterm':>16} {'E0 reference':>14} "
          f"{'diff':>10} {'band dev':>9} {'shape dev':>10} {'width ratio':>11}")
    for path in sorted(SPINE.glob("spine_*.npz"),
                       key=lambda p: -float(np.load(p)["jpm"])):
        data = np.load(path)
        jpm = float(data["jpm"])
        if requested and not any(abs(jpm - v) < 1e-9 for v in requested):
            continue
        started = time.perf_counter()
        band = np.sort(np.asarray(data["band"]))

        hamiltonian = microscopic_character_hamiltonian(
            cluster, states, jpm, np.zeros(3))
        blocks = build_blocks(cluster, states, ice_rows, hamiltonian)
        result = solve_roots(blocks, masked=True)
        roots = np.sort(result.energies[result.converged])

        width = float(band[-1] - band[0])
        record = {"jpm": jpm, "n_roots": int(len(roots)),
                  "ground_counterterm": float(band[0]),
                  "ground_reference": float(roots[0]),
                  "ground_difference": float(band[0] - roots[0]),
                  "band_width": width,
                  "runtime": time.perf_counter() - started}
        if len(roots) == N_ICE:
            record["band_deviation"] = float(
                np.max(np.abs(band - roots)) / width)
            record["shape_deviation"] = float(
                np.max(np.abs((band - band[0]) - (roots - roots[0]))) / width)
            record["width_ratio"] = width / float(roots[-1] - roots[0])
        records.append(record)
        show = lambda k, f: ("---".rjust(9) if record.get(k) is None
                             else format(record[k], f))
        print(f"{jpm:>7.3f} {len(roots):>6d} {band[0]:>16.9f} "
              f"{roots[0]:>14.9f} {record['ground_difference']:>+10.2e} "
              f"{show('band_deviation', '9.2%')} "
              f"{show('shape_deviation', '10.2%')} "
              f"{show('width_ratio', '11.5f')}", flush=True)

    records.sort(key=lambda r: r["jpm"])
    with open(SPINE / "reference_comparison.json", "w") as handle:
        json.dump(records, handle, indent=1, default=float)
    complete = [r for r in records if r["n_roots"] == N_ICE]
    if complete:
        print(f"\nover {len(complete)} couplings with a complete reference:")
        print(f"  |E_0 difference| max = "
              f"{max(abs(r['ground_difference']) for r in complete):.2e}")
        print(f"  band deviation   max = "
              f"{max(r['band_deviation'] for r in complete) * 100:.2f}% "
              f"of the band width")
    print(f"wrote {SPINE / 'reference_comparison.json'}")


if __name__ == "__main__":
    main()
