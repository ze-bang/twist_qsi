#!/usr/bin/env python3
"""Ground-state map of the cubic-16 XYZ model over the (Jx, Jy) plane at Jz = 1.

The campaign's plane is parametrised by (Jpm, Jpmpm), which covers only a
narrow diagonal band of the XYZ couplings.  This scans the full square
Jx, Jy in [-1, 1] with the Ising axis fixed at Jz = 1, using the exact
65,536-state spectrum -- no ice projection, no mask.  That is deliberate:
the mask cleans the *gauge band* of a finite-size artefact, but the phase a
finite cluster sits in is a property of its true ground state, so the
diagnostics here are all raw.

Mapping (SIMULATION_PLAN, dominant-a convention), with Jz = Ja:

    Jpm = -(Jx + Jy)/4,   Jpmpm = (Jx - Jy)/4.

At theta = 0 the Hamiltonian is linear in the couplings --- diagonal Ising
part, exchange part carrying -Jpm, pair-flip part carrying +Jpmpm --- so the
three pieces are built once and recombined per grid point.  That turns a
20 s Hamiltonian assembly into a matrix add, which is what makes a fine grid
affordable.

Recorded per point: the ten lowest levels (so degeneracy structure can be
re-analysed without rescanning), the ice weight of the ground state
<psi|P_ice|psi>, and the mean ice violation <sum_t Q_t^2>.

Jx <-> Jy is the exact symmetry Jpmpm -> -Jpmpm, verified to 6e-14 on the
campaign grid; the map must be symmetric about the diagonal, which is the
scan's own internal check.

Usage: scan_xy_phase_plane.py <n_grid> <column-index>
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import eigsh

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "notes"))

import recompute_finite_size_artifact as geometry  # noqa: E402
from qsi_campaign.exact_band import full_spin_basis  # noqa: E402

OUTPUT = ROOT / "campaign" / "outputs" / "xy_phase_plane"
OUTPUT.mkdir(parents=True, exist_ok=True)

N_LEVELS = 10
DEGEN_TOL = 1e-8


def build_pieces(cluster, states):
    """Ising diagonal, exchange, and pair-flip operators on the full basis.

    Every bond term is a bit-flip of one pair, so each piece is assembled
    from vectorised searchsorted lookups rather than a Python loop over
    states -- the whole point being that this runs once per job, not once
    per grid point.
    """
    ups = np.bitwise_count(states[:, None] & cluster.tet_masks[None, :])
    diagonal = 0.5 * ((ups.astype(np.int16) - 2) ** 2).sum(axis=1).astype(float)

    one = np.uint64(1)
    rows_ex, cols_ex, rows_pr, cols_pr = [], [], [], []
    index = np.arange(len(states))
    for (left, right) in cluster.bonds:
        left_bit = one << np.uint64(int(left))
        right_bit = one << np.uint64(int(right))
        left_up = (states & left_bit) != 0
        right_up = (states & right_bit) != 0
        flipped = states ^ (left_bit | right_bit)
        target = np.searchsorted(states, flipped)
        differ = left_up != right_up
        rows_ex.append(target[differ])
        cols_ex.append(index[differ])
        same = ~differ
        rows_pr.append(target[same])
        cols_pr.append(index[same])

    size = len(states)
    exchange = coo_matrix(
        (np.ones(sum(len(r) for r in rows_ex)),
         (np.concatenate(rows_ex), np.concatenate(cols_ex))),
        shape=(size, size)).tocsr()
    pair = coo_matrix(
        (np.ones(sum(len(r) for r in rows_pr)),
         (np.concatenate(rows_pr), np.concatenate(cols_pr))),
        shape=(size, size)).tocsr()
    ising = csr_matrix(
        (diagonal, (index, index)), shape=(size, size))
    for name, matrix in (("exchange", exchange), ("pair", pair)):
        asymmetry = abs(matrix - matrix.T).max()
        if asymmetry > 1e-12:
            raise RuntimeError(f"{name} piece is not symmetric: {asymmetry:.3e}")
    return ising, exchange, pair, diagonal


def main() -> None:
    n_grid = int(sys.argv[1])
    column = int(sys.argv[2])

    cluster = geometry.build_cluster("cubic", (1, 1, 1))
    states = full_spin_basis(cluster.n_sites)
    ice_rows = np.searchsorted(states, np.sort(
        np.asarray(cluster.ice_states, dtype=np.uint64)))

    started = time.perf_counter()
    ising, exchange, pair, diagonal = build_pieces(cluster, states)
    print(f"pieces built in {time.perf_counter() - started:.0f}s "
          f"(exchange nnz={exchange.nnz}, pair nnz={pair.nnz})", flush=True)

    axis = np.linspace(-1.0, 1.0, n_grid)
    jx = float(axis[column])
    records = []
    for jy in axis:
        point_started = time.perf_counter()
        jpm = -(jx + jy) / 4.0
        jpmpm = (jx - jy) / 4.0
        hamiltonian = (ising - jpm * exchange + jpmpm * pair).tocsr()
        values, vectors = eigsh(hamiltonian, k=N_LEVELS, which="SA",
                                tol=1e-11, maxiter=50000)
        order = np.argsort(values)
        values, vectors = values[order], vectors[:, order]
        ground = vectors[:, 0]
        weight = float(np.sum(np.abs(ground[ice_rows]) ** 2))
        violation = float(np.sum(np.abs(ground) ** 2 * diagonal))
        degeneracy = int(np.sum(values <= values[0] + DEGEN_TOL))
        records.append({
            "jx": jx, "jy": jy, "jpm": jpm, "jpmpm": jpmpm,
            "levels": values.tolist(),
            "gap": float(values[1] - values[0]),
            "degeneracy": degeneracy,
            "ice_weight": weight,
            "ice_violation": violation,
            "runtime": time.perf_counter() - point_started,
        })
        print(f"jx={jx:+.3f} jy={jy:+.3f} E0={values[0]:+.6f} "
              f"gap={values[1] - values[0]:.3e} deg={degeneracy} "
              f"Zice={weight:.4f} viol={violation:.4f} "
              f"({records[-1]['runtime']:.1f}s)", flush=True)

    tag = f"col{column:03d}_n{n_grid}"
    with open(OUTPUT / f"xyscan_{tag}.json", "w") as handle:
        json.dump({"n_grid": n_grid, "column": column, "jx": jx,
                   "points": records}, handle, indent=1, default=float)
    print(f"wrote {OUTPUT}/xyscan_{tag}.json", flush=True)


if __name__ == "__main__":
    main()
