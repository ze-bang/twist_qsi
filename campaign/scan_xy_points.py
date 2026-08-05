#!/usr/bin/env python3
"""Solve an explicit list of (Jx, Jy) points and record the flux with them.

Two jobs use this: filling the handful of cells the column array could not
reach, and mapping the hexagon flux over the plane.  Both want the same
thing -- a converged ground vector at a named point -- so they share one
script and one output schema.

Why the column array stalled, and what is different here.  The original
scan called eigsh(k=10) with the default Krylov space, ncv = 21.  Twenty-one
basis vectors is very tight for ten wanted eigenvalues, and the points that
died are exactly the ones whose lowest levels are clustered to 1e-4 and
below -- ARPACK restarts forever without ever isolating them.  The fix is
room, not patience: ncv = 80 and up, escalating if that still fails, with
the ice-uniform vector as v0 (the ground state is ice-dominated in the
region that stalls) and a warm start from the previous point of the same
task.  Partial convergence is caught and reported rather than thrown away.

Recorded per point: the low levels, gap, degeneracy, ice weight, ice
violation, and the two loop amplitudes -- the contractible hexagon flux
<O_hex + h.c.>, whose sign is the 0-flux / pi-flux label, and the winding
four-loop amplitude, which is the finite-size artefact the mask removes.

Usage: scan_xy_points.py <points.json> <task_id> <n_tasks> <out_prefix>
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.sparse.linalg import eigsh
from scipy.sparse.linalg import ArpackNoConvergence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "notes"))
sys.path.insert(0, str(ROOT / "campaign"))

import recompute_finite_size_artifact as geometry  # noqa: E402
from qsi_campaign.exact_band import full_spin_basis  # noqa: E402
from scan_xy_phase_plane import build_pieces  # noqa: E402

OUTPUT = ROOT / "campaign" / "outputs" / "xy_phase_plane"
OUTPUT.mkdir(parents=True, exist_ok=True)

N_LEVELS = 10
DEGEN_TOL = 1e-8
LADDER = ((10, 80, 1e-10), (20, 180, 1e-9), (32, 300, 1e-8))


def loop_operator(sites):
    """Row indices and their partners for O_loop + h.c. on the full basis.

    The basis is arange(2**n), so a configuration is its own index and the
    flipped partner is a bare XOR.  A loop of even length is flippable in
    exactly two patterns (alternating up/down and its complement); including
    both is what makes the operator Hermitian.
    """
    mask = 0
    pattern = 0
    for position, site in enumerate(sites):
        mask |= 1 << int(site)
        if position % 2 == 0:
            pattern |= 1 << int(site)
    other = mask ^ pattern
    configurations = np.arange(1 << 16, dtype=np.int64)
    occupied = configurations & mask
    active = np.where((occupied == pattern) | (occupied == other))[0]
    return active, active ^ mask


def loop_expectation(operators, vectors):
    """Mean of <O + h.c.> over the supplied loops and over the multiplet.

    Averaging over the degenerate ground multiplet, not over one ARPACK
    vector, is what makes this comparable with the counterterm runs -- and
    it is the only well-defined choice, since a single vector inside a
    degenerate subspace is whatever the solver happened to return.
    """
    if not operators:
        return float("nan")
    vectors = np.atleast_2d(vectors.T).T
    return float(np.mean([
        float(np.real(np.sum(vectors[active, column].conj()
                             * vectors[partners, column])))
        for active, partners in operators
        for column in range(vectors.shape[1])]))


def solve(hamiltonian, v0):
    """Lowest levels with escalating Krylov room; report what converged."""
    last = None
    for k, ncv, tol in LADDER:
        try:
            values, vectors = eigsh(hamiltonian, k=k, which="SA", v0=v0,
                                    ncv=ncv, tol=tol, maxiter=100000)
            order = np.argsort(values)
            return values[order], vectors[:, order], k, ncv, True
        except ArpackNoConvergence as failure:
            values = np.asarray(failure.eigenvalues)
            vectors = np.asarray(failure.eigenvectors)
            print(f"    ncv={ncv}: only {values.size}/{k} converged",
                  flush=True)
            if values.size >= 2:
                order = np.argsort(values)
                last = (values[order], vectors[:, order], k, ncv, False)
    if last is None:
        raise RuntimeError("no eigenpairs converged at any Krylov size")
    return last


def main() -> None:
    points_file, task_id, n_tasks, prefix = (
        sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4])
    requested = json.loads(Path(points_file).read_text())["points"]
    mine = sorted(requested[task_id::n_tasks],
                  key=lambda point: (point["jx"], point["jy"]))
    print(f"task {task_id}/{n_tasks}: {len(mine)} of {len(requested)} points",
          flush=True)
    if not mine:
        return

    cluster = geometry.build_cluster("cubic", (1, 1, 1))
    states = full_spin_basis(cluster.n_sites)
    ice_rows = np.searchsorted(states, np.sort(
        np.asarray(cluster.ice_states, dtype=np.uint64)))

    started = time.perf_counter()
    ising, exchange, pair, diagonal = build_pieces(cluster, states)
    hexagons = {tuple(sorted(sites)): sites
                for sites, wind in cluster.hexes if tuple(wind) == (0, 0, 0)}
    windings = {tuple(sorted(sites)): sites for sites, _ in cluster.loops4}
    hex_ops = [loop_operator(sites) for sites in hexagons.values()]
    wind_ops = [loop_operator(sites) for sites in windings.values()]
    print(f"pieces built in {time.perf_counter() - started:.0f}s; "
          f"{len(hex_ops)} contractible hexagons, {len(wind_ops)} 4-loops",
          flush=True)

    guess = np.zeros(len(states))
    guess[ice_rows] = 1.0 / np.sqrt(len(ice_rows))

    records = []
    for point in mine:
        point_started = time.perf_counter()
        jx, jy = float(point["jx"]), float(point["jy"])
        jpm, jpmpm = -(jx + jy) / 4.0, (jx - jy) / 4.0
        hamiltonian = (ising - jpm * exchange + jpmpm * pair).tocsr()
        values, vectors, k, ncv, full = solve(hamiltonian, guess)
        ground = vectors[:, 0]
        guess = ground.copy()          # warm start for the next point
        degeneracy = int(np.sum(values <= values[0] + DEGEN_TOL))
        multiplet = vectors[:, :degeneracy]
        records.append({
            "jx": jx, "jy": jy, "jpm": jpm, "jpmpm": jpmpm,
            "levels": values[:N_LEVELS].tolist(),
            "energy": float(values[0]),
            # "gap" keeps the existing columns' definition so panel (b) stays
            # one quantity across the plane; the multiplet gap sits beside it
            "gap": float(values[1] - values[0]),
            "gap_multiplet": float(values[degeneracy] - values[0])
                             if degeneracy < values.size else float("nan"),
            "degeneracy": degeneracy,
            # vector 0 for parity with the existing columns, multiplet mean
            # beside it because that is the basis-independent number
            "ice_weight": float(np.sum(np.abs(ground[ice_rows]) ** 2)),
            "ice_weight_multiplet": float(np.mean(
                np.sum(np.abs(multiplet[ice_rows, :]) ** 2, axis=0))),
            "ice_violation": float(np.sum(np.abs(ground) ** 2 * diagonal)),
            "hexagon_flux": loop_expectation(hex_ops, multiplet),
            "hexagon_flux_ground": loop_expectation(hex_ops, ground),
            "winding4_amplitude": loop_expectation(wind_ops, multiplet),
            "converged": bool(full),
            "krylov": [k, ncv],
            "runtime": time.perf_counter() - point_started,
        })
        entry = records[-1]
        print(f"jx={jx:+.3f} jy={jy:+.3f} E0={entry['energy']:+.6f} "
              f"gap={entry['gap']:.3e} deg={entry['degeneracy']} "
              f"Zice={entry['ice_weight']:.4f} "
              f"flux6={entry['hexagon_flux']:+.5f} "
              f"wind4={entry['winding4_amplitude']:+.5f} "
              f"{'ok' if full else 'PARTIAL'} ({entry['runtime']:.1f}s)",
              flush=True)

    with open(OUTPUT / f"{prefix}_{task_id:03d}.json", "w") as handle:
        json.dump({"points": records}, handle, indent=1, default=float)
    print(f"wrote {OUTPUT}/{prefix}_{task_id:03d}.json", flush=True)


if __name__ == "__main__":
    main()
