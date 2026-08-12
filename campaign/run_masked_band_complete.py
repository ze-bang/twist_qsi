#!/usr/bin/env python3
"""Recover the masked map's escaped branches: the complete winding-free band.

The reference solver brackets every root in [floor, edge] with
edge = min(poles) - 1e-9, so a branch whose root has migrated above the lowest
pole of the self-energy is reported as unconverged.  That is why the reference
has 90 roots out to Jpm = -0.22, 89 at -0.25 and 77 at -0.28: past that point
the winding-free spectrum was being read off an incomplete band, and every
comparison against it inherited the hole.

Those states are not missing.  Two structural facts locate them.

1.  The mask makes Delta[F(z)] block diagonal in the transport sectors, and the
    s-th block is

        F_s(z) = (PHP)_ss + (PHQ)_s (z - QHQ)^{-1} (QHP)_s,

    which is exactly the Feshbach map of the Hermitian bordered matrix

        M_s = [[ (PHP)_ss , (PHQ)_s ],
               [ (QHP)_s  ,   QHQ   ]].

    So each sector's winding-free spectrum is the spectrum of a genuine
    Hermitian operator.  Nothing forbids eigenvalues above the continuum edge,
    and there is no sense in which a branch "ceases to exist" there.

    (This also explains the earlier failure of a global root count: summing over
    sectors gives sum_s (d_s + n_Q) = 90 + 25 * 12780 roots, because each sector
    carries its own copy of the complement.  The masked map is a direct sum of
    25 Schur complements, not the Schur complement of one 90-dimensional
    operator, so counting arguments that assume the latter are invalid.)

2.  Between two consecutive poles each sorted branch runs monotonically from
    +infinity to -infinity, so g_n(z) = lambda_n(z) - z has exactly one root per
    pole interval.  Bisection anywhere above the edge is therefore as safe as
    below it; what is *not* safe is assuming the root you land on is the band
    state, because every interval has one.

The band state is identified by its ice weight.  A bound state in the continuum
carries O(1) weight on the ice manifold while the dressed spinon roots
surrounding it carry almost none, so scanning above the edge and keeping the
largest-Z root per branch recovers the branch that escaped -- and Z itself is
the diagnostic that says whether the recovery is meaningful.

Validation: at couplings where the reference is complete this must reproduce it
exactly, since every root then lies below the edge and no scanning happens.

Usage: run_masked_band_complete.py <jpm> [<jpm> ...] [--window W] [--grid N]
                                   [--jpmpm PP] [--full-basis]

``--jpmpm`` selects the full 65,536-state cubic-16 spin basis (Sz is no longer
conserved) and adds the S+S+/S-S- bond terms at theta=0; the transport-sector
mask is untouched because sector labels are properties of the ice
configurations alone.  ``--full-basis`` forces the full basis at Jpmpm = 0:
there every coupling to an Sz != 0 pole vanishes identically, so the result
must reproduce the fixed-Sz calculation exactly -- the structural gate for the
full-basis path.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.linalg import eigvalsh, eigh

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "notes"))
sys.path.insert(0, str(ROOT / "campaign"))

import recompute_finite_size_artifact as geometry  # noqa: E402
from qsi_campaign.exact_band import (  # noqa: E402
    cubic16_translation_permutations,
    fixed_magnetization_basis,
    full_spin_basis,
    microscopic_character_hamiltonian,
    translation_sector_bases,
)
from qsi_campaign.downfold import (  # noqa: E402
    build_blocks,
    build_blocks_klein,
    solve_roots,
)

OUTPUT = ROOT / "campaign" / "outputs" / "masked_band_complete"
OUTPUT.mkdir(parents=True, exist_ok=True)
N_ICE = 90
COUPLING_TOL = 1.0e-12


class SectorMap:
    """The masked map restricted to one transport sector."""

    def __init__(self, blocks, rows):
        self.rows = np.asarray(rows)
        self.php = blocks.php[np.ix_(self.rows, self.rows)]
        coupling = blocks.coupling[:, self.rows]          # (nq, d)
        # Poles decoupled from this sector are not poles of F_s at all; keeping
        # them would manufacture spurious intervals.
        live = np.linalg.norm(coupling, axis=1) > COUPLING_TOL
        self.coupling = coupling[live]
        self.poles = blocks.poles[live]
        self.dimension = len(self.rows)

    def matrix(self, z):
        weights = 1.0 / (z - self.poles)
        return self.php + (self.coupling * weights[:, None]).T @ self.coupling

    def branches(self, z):
        return eigvalsh(self.matrix(z))

    def residue(self, z, vector):
        amplitude = self.coupling @ vector
        slope = -float(np.sum(np.abs(amplitude) ** 2 / (z - self.poles) ** 2))
        return 1.0 / (1.0 - slope)

    def root_data(self, z):
        values, vectors = eigh(self.matrix(z))
        return values, vectors


def bisect_branch(sector, branch, low, high, tolerance=1e-13, steps=200):
    """The unique root of lambda_branch(z) - z in a pole-free interval."""
    for _ in range(steps):
        middle = 0.5 * (low + high)
        if sector.branches(middle)[branch] - middle > 0.0:
            low = middle
        else:
            high = middle
        if high - low < tolerance:
            break
    return 0.5 * (low + high)


def scan_above_edge(sector, branch, edge, window, grid_points):
    """Every downward zero crossing of g_n above the edge, with its residue.

    A genuine root has g decreasing through zero; a pole crossing has g jumping
    from -infinity to +infinity, i.e. increasing.  Only the former is kept.
    """
    grid = np.linspace(edge, edge + window, grid_points)
    values = np.empty(len(grid))
    for index, z in enumerate(grid):
        values[index] = sector.branches(z)[branch] - z
    found = []
    for index in range(len(grid) - 1):
        if values[index] > 0.0 >= values[index + 1]:
            z = bisect_branch(sector, branch, grid[index], grid[index + 1])
            eigenvalues, eigenvectors = sector.root_data(z)
            weight = sector.residue(z, eigenvectors[:, branch])
            found.append((weight, z))
    return found


def main() -> None:
    # Option values are not couplings: filtering on a leading "--" silently
    # swallowed 1.5 and 25000 as couplings on the first run of this script.
    window, grid_points, couplings = 0.40, 6000, []
    jpmpm, full_basis, direct = 0.0, False, False
    gpu, tag_suffix = False, ""
    tokens = list(sys.argv[1:])
    while tokens:
        token = tokens.pop(0)
        if token == "--window":
            window = float(tokens.pop(0))
        elif token == "--grid":
            grid_points = int(tokens.pop(0))
        elif token == "--jpmpm":
            jpmpm = float(tokens.pop(0))
        elif token == "--full-basis":
            full_basis = True
        elif token == "--direct":
            direct = True
        elif token == "--gpu":
            gpu = True
        elif token == "--tag-suffix":
            tag_suffix = tokens.pop(0)
        else:
            couplings.append(float(token))

    cluster = geometry.build_cluster("cubic", (1, 1, 1))
    if jpmpm != 0.0 or full_basis:
        states = full_spin_basis(cluster.n_sites)
    else:
        states = fixed_magnetization_basis(cluster.n_sites, cluster.n_sites // 2)
    index = {int(s): i for i, s in enumerate(states)}
    ice_rows = np.asarray([index[int(s)] for s in cluster.ice_states])

    # the Klein translation decomposition cuts the QHQ eigendecomposition
    # ~16x on the full basis; --direct keeps the single dense solve as the
    # cross-validation reference
    blocked = len(states) > 20000 and not direct
    sector_bases = None
    if blocked:
        sector_bases = translation_sector_bases(
            states, cubic16_translation_permutations(cluster))

    for jpm in couplings:
        started = time.perf_counter()
        hamiltonian = microscopic_character_hamiltonian(
            cluster, states, jpm, np.zeros(3), jpmpm=jpmpm)
        if blocked:
            blocks = build_blocks_klein(cluster, states, ice_rows,
                                        hamiltonian, sector_bases, gpu=gpu)
        else:
            blocks = build_blocks(cluster, states, ice_rows, hamiltonian)
        labels = blocks.sector_labels

        # the reference, for validation and for the count it misses
        baseline = solve_roots(blocks, masked=True)
        n_baseline = int(baseline.converged.sum())

        levels, weights, origin, sector_of = [], [], [], []
        for sector_id in np.unique(labels):
            rows = np.flatnonzero(labels == sector_id)
            sector = SectorMap(blocks, rows)
            edge = float(sector.poles.min()) - 1e-9
            floor = min(float(np.linalg.eigvalsh(sector.php).min()), edge) - 2.0
            for _ in range(12):
                if np.all(sector.branches(floor) - floor > 0.0):
                    break
                floor -= 4.0
            gap_at_edge = sector.branches(edge) - edge

            for branch in range(sector.dimension):
                if gap_at_edge[branch] <= 0.0:
                    z = bisect_branch(sector, branch, floor, edge)
                    values, vectors = sector.root_data(z)
                    levels.append(z)
                    weights.append(sector.residue(z, vectors[:, branch]))
                    origin.append("below")
                else:
                    candidates = scan_above_edge(sector, branch, edge,
                                                 window, grid_points)
                    if not candidates:
                        levels.append(np.nan)
                        weights.append(np.nan)
                        origin.append("lost")
                    else:
                        weight, z = max(candidates)
                        levels.append(z)
                        weights.append(weight)
                        origin.append("above")
                sector_of.append(int(sector_id))

        levels = np.asarray(levels)
        weights = np.asarray(weights)
        origin = np.asarray(origin)
        order = np.argsort(np.where(np.isnan(levels), np.inf, levels))
        levels, weights, origin = levels[order], weights[order], origin[order]
        sector_of = np.asarray(sector_of)[order]

        n_below = int((origin == "below").sum())
        n_above = int((origin == "above").sum())
        n_lost = int((origin == "lost").sum())

        record = {
            "jpm": jpm,
            "jpmpm": jpmpm,
            "n_states": len(states),
            "n_reference": n_baseline,
            "n_below_edge": n_below,
            "n_recovered_above": n_above,
            "n_lost": n_lost,
            "levels": levels.tolist(),
            "ice_weight": weights.tolist(),
            "origin": origin.tolist(),
            "sector": sector_of.tolist(),
            "window": window,
            "grid_points": grid_points,
        }

        # validation: where the reference is complete, we must reproduce it
        if n_baseline == N_ICE and n_lost == 0:
            reference = np.sort(baseline.energies[baseline.converged])
            record["max_deviation_from_reference"] = float(
                np.max(np.abs(np.sort(levels) - reference)))

        record["method"] = ("klein-gpu" if gpu else "klein") if blocked \
            else "direct"
        record["runtime"] = time.perf_counter() - started
        tag = f"{jpm:+.3f}".replace(".", "p")
        if jpmpm != 0.0:
            tag += f"_pp{jpmpm:+.3f}".replace(".", "p")
            if not blocked:
                tag += "_direct"
        elif full_basis:
            tag += "_fullbasis" + ("_klein" if blocked else "")
        tag += tag_suffix
        with open(OUTPUT / f"band_{tag}.json", "w") as handle:
            json.dump(record, handle, indent=1, default=float)

        deviation = record.get("max_deviation_from_reference")
        print(f"jpm={jpm:+.3f} pp={jpmpm:+.3f} reference={n_baseline:3d}/90 -> "
              f"complete={n_below + n_above:3d}/90 "
              f"(below {n_below}, recovered {n_above}, lost {n_lost}) | "
              f"Z of recovered: "
              f"{'-' if not n_above else f'{np.nanmin(weights[origin == chr(97)+chr(98)+chr(111)+chr(118)+chr(101)]):.3f}'}"
              f"..{'-' if not n_above else f'{np.nanmax(weights[origin == chr(97)+chr(98)+chr(111)+chr(118)+chr(101)]):.3f}'} | "
              f"validation={'-' if deviation is None else f'{deviation:.1e}'} "
              f"({record['runtime']:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
