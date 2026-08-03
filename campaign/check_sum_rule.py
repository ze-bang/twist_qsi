#!/usr/bin/env python3
"""The entropy sum rule, with the integration range converged.

The sum rule integral runs to infinity, and C/N ~ Var(E)/(N T^2) at large T, so
truncating it at T_max leaves a deficit falling as 1/T_max^2.  On cubic-16 that
deficit is 2.6e-5 at T_max = 60 J_zz -- comparable to the differences one might
want to resolve -- and the natural mistake is to read it as a lost level.  It is
not: refining the grid at fixed T_max does not move it (it converges to
-2.58e-5), while extending T_max does, as

    T_max =  60  ->  -2.58e-05
    T_max = 200  ->  -2.32e-06     (ratio 11.1, (200/60)^2 = 11.1)
    T_max = 600  ->  -2.23e-07

This script recomputes the rule on a converged range for every coupling and for
each construction, so the check quoted in the paper is the strong one.

Usage: check_sum_rule.py [--tmax 600] [--n 40000]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SPINE = ROOT / "campaign" / "outputs" / "counterterm_spine"
N_SITES = 16
DEGENERACY_TOL = 1.0e-9


# The full spectrum carries ~42000 level pairs split below 1e-8, essentially all
# at machine precision (down to 5e-17).  Integrating from below that scale picks
# up spurious sub-picokelvin Schottky structure and the quadrature becomes noisy:
# at Jpm = -0.13 it produced a +1.3e-6 outlier of the wrong sign.  T_MIN sits
# above the noise and far below the smallest physical gap (the ring scale, ~1e-5
# at the weakest coupling); 1e-8 and 1e-6 agree to 1.3e-8, confirming nothing
# physical lives in between.
T_MIN = 1.0e-8


def released_entropy(levels, tmax, n, chunk=256):
    energies = np.sort(np.asarray(levels, float))
    energies = energies - energies[0]
    degeneracy = int(np.sum(energies < DEGENERACY_TOL))
    grid = np.geomspace(T_MIN, tmax, n)
    integrand = np.empty(len(grid))
    for start in range(0, len(grid), chunk):
        temperature = grid[start:start + chunk]
        weight = np.exp(-energies[None, :] / temperature[:, None])
        partition = weight.sum(1)
        mean = (weight * energies[None, :]).sum(1) / partition
        second = (weight * energies[None, :] ** 2).sum(1) / partition
        heat = (second - mean * mean) / (N_SITES * temperature ** 2)
        integrand[start:start + chunk] = heat / temperature
    return (float(np.trapezoid(integrand, grid))
            + np.log(degeneracy) / N_SITES, degeneracy)


def main() -> None:
    tmax = 600.0
    n = 40000
    if "--tmax" in sys.argv:
        tmax = float(sys.argv[sys.argv.index("--tmax") + 1])
    if "--n" in sys.argv:
        n = int(sys.argv[sys.argv.index("--n") + 1])

    target = float(np.log(2.0))
    print(f"entropy sum rule, T in [{T_MIN:g}, {tmax:g}], {n} points; target "
          f"ln 2 = {target:.9f}")
    print(f"{'jpm':>7} {'counterterm':>13} {'err':>11} {'g0':>3} | "
          f"{'periodic':>13} {'err':>11} {'g0':>3}")
    rows = []
    worst = 0.0
    for path in sorted(SPINE.glob("spine_*.npz"),
                       key=lambda p: -float(np.load(p)["jpm"])):
        data = np.load(path)
        jpm = float(data["jpm"])
        row = {"jpm": jpm}
        line = f"{jpm:>7.3f}"
        for key, tag in (("levels", "counterterm"),):
            value, degeneracy = released_entropy(data[key], tmax, n)
            row[tag] = value
            row[f"{tag}_g0"] = degeneracy
            worst = max(worst, abs(value - target))
            line += f" {value:13.9f} {value - target:+11.2e} {degeneracy:>3d} |"
        # the periodic spectrum shares every S^z != 0 sector, so it is the
        # independent control on the splice
        periodic = np.load(ROOT / "campaign" / "outputs" / "method_comparison"
                           / f"compare_{format(jpm, '+.3f').replace('.', 'p')}"
                             ".npz")["periodic_levels"]
        value, degeneracy = released_entropy(periodic, tmax, n)
        row["periodic"] = value
        row["periodic_g0"] = degeneracy
        worst = max(worst, abs(value - target))
        line += f" {value:13.9f} {value - target:+11.2e} {degeneracy:>3d}"
        print(line, flush=True)
        rows.append(row)

    with open(SPINE / "sum_rule.json", "w") as handle:
        json.dump({"tmax": tmax, "n": n, "target": target, "rows": rows},
                  handle, indent=1, default=float)
    print(f"\nworst deviation from ln 2 across all couplings and both "
          f"constructions: {worst:.2e}")
    print(f"wrote {SPINE / 'sum_rule.json'}")


if __name__ == "__main__":
    main()
