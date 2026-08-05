#!/usr/bin/env python3
"""Test the reflection the flux map is built on, by solving both sides.

The map solves only Jx >= Jy and mirrors the result, on the argument that
the pi/2 rotation sending Jpmpm -> -Jpmpm leaves the hexagon operator
invariant: three S+ and three S- pick up (i)^3 (-i)^3 = 1.  That argument
is worth exactly as much as its verification, and the assembled figure
cannot check it -- it writes both cells from one solve, so any self-check
there returns zero by construction.

So solve both members of several mirror pairs from scratch and compare.
The gap and the ice violation should agree to machine precision; the ice
weight is an eigenvector quantity and may drift where the spectrum is
clustered, which is worth seeing separately.

Usage: check_flux_symmetry.py
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
from qsi_campaign.exact_band import full_spin_basis  # noqa: E402
from scan_xy_phase_plane import build_pieces  # noqa: E402
from scan_xy_points import (DEGEN_TOL, loop_expectation,  # noqa: E402
                            loop_operator, solve)

PAIRS = [(0.50, -0.50), (0.30, -0.70), (0.60, 0.20),
         (0.10, -0.35), (0.85, 0.40), (0.05, -0.05)]


def measure(pieces, hex_ops, wind_ops, ice_rows, diagonal, guess, jx, jy):
    ising, exchange, pair = pieces
    jpm, jpmpm = -(jx + jy) / 4.0, (jx - jy) / 4.0
    hamiltonian = (ising - jpm * exchange + jpmpm * pair).tocsr()
    values, vectors, _, _, converged = solve(hamiltonian, guess)
    degeneracy = int(np.sum(values <= values[0] + DEGEN_TOL))
    multiplet = vectors[:, :degeneracy]
    return {
        "energy": float(values[0]),
        "gap": float(values[1] - values[0]),
        "degeneracy": degeneracy,
        "ice_weight": float(np.mean(
            np.sum(np.abs(multiplet[ice_rows, :]) ** 2, axis=0))),
        "ice_violation": float(np.sum(np.abs(vectors[:, 0]) ** 2 * diagonal)),
        "hexagon_flux": loop_expectation(hex_ops, multiplet),
        "winding4": loop_expectation(wind_ops, multiplet),
        "converged": converged,
    }


def main() -> None:
    cluster = geometry.build_cluster("cubic", (1, 1, 1))
    states = full_spin_basis(cluster.n_sites)
    ice_rows = np.searchsorted(states, np.sort(
        np.asarray(cluster.ice_states, dtype=np.uint64)))
    ising, exchange, pair, diagonal = build_pieces(cluster, states)
    hexagons = {tuple(sorted(sites)): sites
                for sites, wind in cluster.hexes if tuple(wind) == (0, 0, 0)}
    windings = {tuple(sorted(sites)): sites for sites, _ in cluster.loops4}
    hex_ops = [loop_operator(sites) for sites in hexagons.values()]
    wind_ops = [loop_operator(sites) for sites in windings.values()]
    guess = np.zeros(len(states))
    guess[ice_rows] = 1.0 / np.sqrt(len(ice_rows))
    pieces = (ising, exchange, pair)

    worst = {}
    for jx, jy in PAIRS:
        forward = measure(pieces, hex_ops, wind_ops, ice_rows, diagonal,
                          guess, jx, jy)
        mirror = measure(pieces, hex_ops, wind_ops, ice_rows, diagonal,
                         guess, jy, jx)
        print(f"({jx:+.2f}, {jy:+.2f}) vs mirror:")
        for name in ("energy", "gap", "ice_weight", "ice_violation",
                     "hexagon_flux", "winding4"):
            difference = abs(forward[name] - mirror[name])
            worst[name] = max(worst.get(name, 0.0), difference)
            print(f"    {name:14s} {forward[name]:+.12f} vs "
                  f"{mirror[name]:+.12f}   diff {difference:.3e}")
        if forward["degeneracy"] != mirror["degeneracy"]:
            print(f"    DEGENERACY MISMATCH {forward['degeneracy']} "
                  f"vs {mirror['degeneracy']}")

    print("\nworst discrepancy over all pairs:")
    for name, value in worst.items():
        print(f"  {name:14s} {value:.3e}")
    verdict = "PASS" if worst["hexagon_flux"] < 1e-9 else "FAIL"
    print(f"hexagon flux reflection: {verdict} "
          f"(max {worst['hexagon_flux']:.3e})")


if __name__ == "__main__":
    main()
