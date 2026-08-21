#!/usr/bin/env python3
"""How many distinct excitations actually carry longitudinal weight, and where.

The path figure shows several horizontal features and the obvious question is
whether they are photon branches.  Two things have to be separated before that
can be answered on a 32-site cluster:

  * a finite cluster has DISCRETE levels, not bands.  Whether a feature
    disperses is decided by comparing the weighted excitation energy between
    the eight genuine cluster momenta -- not by looking at the interpolation
    between them, which carries no independent information.
  * the emergent photon has TWO transverse polarisations, so two branches is
    the expectation, not one; they are degenerate along symmetry directions.
    Any feature beyond two is something else.

For each of the eight momenta this prints the excitations carrying at least
`--floor` of that momentum's total weight, so the branch count is read off the
data rather than off a colour scale.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DSSF = ROOT / "campaign" / "outputs" / "wp3_dssf"


def star(momentum) -> str:
    magnitudes = sorted(abs(float(v)) for v in momentum)
    if np.allclose(magnitudes, [0.0, 0.0, 0.0]):
        return "Gamma"
    if np.allclose(magnitudes, [0.0, 0.0, 1.0]):
        return "X"
    if np.allclose(magnitudes, [0.5, 0.5, 0.5]):
        return "L"
    return "?"


def cluster_peaks(path: Path, floor: float, merge: float) -> None:
    record = json.loads(path.read_text())
    print(f"\n=== {path.name}   jpm={record['jpm']:+.3f} "
          f"states={record.get('nstates_computed', '?')}")
    for entry in record["momenta"]:
        excitation = np.asarray(entry["excitation"], dtype=float)
        weight = np.asarray(entry["weight"], dtype=float)
        total = weight.sum()
        # A superselected momentum has total weight at the 1e-30 noise floor.
        # Reporting "99.9% of it sits at omega=x" would dress numerical dust up
        # as a spectral feature, so those are reported as the zero they are.
        if total <= 1.0e-12:
            print(f"  {star(entry['momentum']):5s} "
                  f"{str([round(float(v), 2) for v in entry['momentum']]):22s} "
                  f"total={total:.2e}  (superselected: no spectral weight)")
            continue
        # merge numerically degenerate levels into one line before counting
        order = np.argsort(excitation)
        excitation, weight = excitation[order], weight[order]
        groups: list[list[float]] = []
        for e, w in zip(excitation, weight):
            if groups and abs(e - groups[-1][0]) < merge:
                groups[-1][1] += w
            else:
                groups.append([float(e), float(w)])
        kept = [(e, w) for e, w in groups if w >= floor * total and e > 1e-9]
        kept.sort(key=lambda row: -row[1])
        # Degeneracy is the discriminator.  The two transverse photon
        # polarisations sit in a 2D irrep of the little group at both X and L,
        # so they are exactly degenerate and must show up as ONE level of
        # multiplicity two -- never as two split lines.  A pair of separated
        # peaks is therefore the photon plus something else, and the
        # multiplicity says which is which.
        pieces = []
        for e, w in kept:
            degeneracy = int(np.sum(np.abs(excitation - e) < merge))
            carrying = int(np.sum((np.abs(excitation - e) < merge)
                                  & (weight > 1e-6 * total)))
            pieces.append(f"w={w / total:5.1%}@{e:.5f}"
                          f"[deg={degeneracy},lit={carrying}]")
        print(f"  {star(entry['momentum']):5s} "
              f"{str([round(float(v), 2) for v in entry['momentum']]):22s} "
              f"total={total:.4f} n_peaks={len(kept):2d}  " + "  ".join(pieces))


def main() -> int:
    floor, merge = 0.02, 1.0e-6
    files = []
    for token in sys.argv[1:]:
        if token.startswith("--floor="):
            floor = float(token.split("=", 1)[1])
        elif token.startswith("--merge="):
            merge = float(token.split("=", 1)[1])
        else:
            files.append(Path(token))
    if not files:
        files = [
            DSSF / "dssf_fcc32_L1_+0p046_pp+0p000.json",
            DSSF / "dssf_periodic_fcc32_L1_+0p046.json",
            DSSF / "dssf_fcc32_L1_-0p200_pp+0p000.json",
            DSSF / "dssf_periodic_fcc32_L1_-0p200.json",
        ]
    print(f"peaks holding >= {floor:.0%} of the per-momentum weight; "
          f"levels merged below {merge:g}")
    for path in files:
        if path.exists():
            cluster_peaks(path, floor, merge)
        else:
            print(f"missing {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
