#!/usr/bin/env python3
"""Spinon-number-truncated Feshbach map: the same measurement on a bigger cluster.

The exact Feshbach map needs the whole complement Q of the ice manifold, which
on cubic-16 is the full 12870-state fixed-magnetization space but on the FCC-32
cluster would be C(32,16) = 6.0e8 states.  The dressing that generates the ring
exchange is however short: every ice-to-ice process at order n passes through
states with at most n/2 spinon pairs.  We therefore restrict Q to the states
reachable from the ice manifold in at most ``levels`` exchange hops (levels = 1
keeps one spinon pair, levels = 2 keeps two), build the sparse Hamiltonian on
that space, and evaluate

    F(z)_{ba} = H_{ba} - H_{bQ} (QHQ - z)^{-1} H_{Qa} ,   z = E_0 ,

by one conjugate-gradient solve per ice column.  The truncation is variational
and its error is measured directly on cubic-16, where the untruncated answer is
available; the same code then runs on FCC-32.

Reported per coupling: the dressed contractible-hexagon amplitude g, the dressed
winding amplitudes, the Rokhsar-Kivelson potential V from the diagonal, the
two-spinon threshold (lowest eigenvalue of QHQ), and the ground-state energy.

Usage: run_truncated_feshbach.py <cubic16|fcc32> <levels> <columns> <jpm> ...
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix, identity
from scipy.sparse.linalg import cg, eigsh

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "notes"))

import recompute_finite_size_artifact as geometry  # noqa: E402

OUTPUT = ROOT / "campaign" / "outputs" / "truncated_feshbach"
OUTPUT.mkdir(parents=True, exist_ok=True)

CLUSTERS = {"cubic16": ("cubic", (1, 1, 1)), "fcc32": ("fcc", (2, 2, 2))}


def path_mask(path) -> int:
    mask = 0
    for site in path:
        mask |= 1 << int(site)
    return mask


def bfs_space(cluster, levels: int, jpmpm: float = 0.0) -> np.ndarray:
    """States reachable from the ice manifold in at most ``levels`` hops.

    With ``jpmpm`` nonzero the pair-flip moves ``S+S+`` / ``S-S-`` are hops too,
    and they must be in the BFS or the truncated space is not closed under the
    Hamiltonian: a pair flip out of the space would silently drop amplitude
    rather than raise.  Pair moves change S^z by +-2, so the reachable space at
    a given ``levels`` is several-fold larger than the XXZ one -- which is the
    cost WP2 has to budget for.
    """
    current = np.asarray(cluster.ice_states, dtype=np.uint64)
    seen = current.copy()
    for _ in range(levels):
        images = []
        for (left, right) in cluster.bonds:
            left_bit = np.uint64(1) << np.uint64(left)
            right_bit = np.uint64(1) << np.uint64(right)
            left_up = (current & left_bit) != 0
            right_up = (current & right_bit) != 0
            exchange = left_up != right_up
            images.append(current[exchange] ^ (left_bit | right_bit))
            if jpmpm != 0.0:
                # S+S+ acts on both-down, S-S- on both-up; the flipped bit
                # pattern is the same XOR either way
                pair = left_up == right_up
                images.append(current[pair] ^ (left_bit | right_bit))
        fresh = np.unique(np.concatenate(images))
        seen = np.union1d(seen, fresh)
        current = fresh
    return np.sort(seen)


def build_hamiltonian(cluster, space: np.ndarray, jpm: float,
                      jpmpm: float = 0.0, strict: bool = False) -> csr_matrix:
    """Sparse XXZ/XYZ Hamiltonian restricted to ``space``.

    Conventions follow ``exact_band.microscopic_character_hamiltonian`` at
    ``theta = 0`` exactly, since the cubic-16 plane grid built with that
    function is the calibration oracle for this one:

    ```text
    H = H0 - Jpm sum_<ij> (S+_i S-_j + S-_i S+_j)
           + Jpmpm sum_<ij> (S+_i S+_j + S-_i S-_j).
    ```

    Note the sign asymmetry -- the exchange term enters with `-Jpm` and the
    pair term with `+Jpmpm`.  Both are real at `theta = 0`, so the matrix stays
    real and the block-CG machinery downstream is unchanged.

    Out-of-space targets are DROPPED, for both channels.  That is not a
    shortcut: restricting an operator to a subspace *is* ``P_sub H P_sub``, so
    discarding the matrix elements that leave the space is exactly the
    variational truncation this whole construction rests on, and it is what the
    exchange channel has always done.  A truncated BFS space is never closed --
    its boundary states always have images one hop further out -- so raising
    here would make every genuinely truncated run impossible.

    ``strict=True`` instead raises on any dropped element.  Use it only to
    ASSERT closure, as WP2's Gate A does after growing the BFS to saturation:
    there the space really is closed, and a dropped element would mean the
    builder disagrees with the reference Hamiltonian.
    """
    ups = np.bitwise_count(space[:, None] & cluster.tet_masks[None, :])
    diagonal = 0.5 * ((ups.astype(np.int16) - 2) ** 2).sum(axis=1)
    rows = [np.arange(len(space))]
    columns = [np.arange(len(space))]
    values = [diagonal.astype(float)]
    for (left, right) in cluster.bonds:
        left_bit = np.uint64(1) << np.uint64(left)
        right_bit = np.uint64(1) << np.uint64(right)
        left_up = (space & left_bit) != 0
        right_up = (space & right_bit) != 0

        source = np.where(left_up != right_up)[0]
        target_states = space[source] ^ (left_bit | right_bit)
        position = np.clip(np.searchsorted(space, target_states),
                           0, len(space) - 1)
        inside = space[position] == target_states
        rows.append(position[inside])
        columns.append(source[inside])
        values.append(np.full(int(inside.sum()), -jpm))

        if jpmpm == 0.0:
            continue
        source = np.where(left_up == right_up)[0]
        target_states = space[source] ^ (left_bit | right_bit)
        position = np.clip(np.searchsorted(space, target_states),
                           0, len(space) - 1)
        inside = space[position] == target_states
        if strict and not inside.all():
            raise ValueError(
                f"{int((~inside).sum())} pair-flip targets fall outside the "
                "space, but closure was asserted: rebuild the space with "
                "bfs_space(..., jpmpm=...) grown to saturation")
        rows.append(position[inside])
        columns.append(source[inside])
        values.append(np.full(int(inside.sum()), jpmpm))

    hamiltonian = csr_matrix(
        (np.concatenate(values),
         (np.concatenate(rows), np.concatenate(columns))),
        shape=(len(space),) * 2)
    return hamiltonian


def loop_partner(state: int, sites) -> int | None:
    mask = path_mask(sites)
    pattern = 0
    for position, site in enumerate(sites):
        if position % 2 == 0:
            pattern |= 1 << int(site)
    occupied = state & mask
    if occupied == pattern or occupied == (mask ^ pattern):
        return state ^ mask
    return None


def main() -> None:
    name = sys.argv[1]
    levels = int(sys.argv[2])
    n_columns = int(sys.argv[3])
    couplings = [float(v) for v in sys.argv[4:]]
    basis, shape = CLUSTERS[name]
    cluster = geometry.build_cluster(basis, shape)

    space = bfs_space(cluster, levels)
    ice_states = np.sort(np.asarray(cluster.ice_states, dtype=np.uint64))
    ice_rows = np.searchsorted(space, ice_states)
    assert np.all(space[ice_rows] == ice_states)
    complement = np.setdiff1d(np.arange(len(space)), ice_rows)

    contractible = [sites for sites, wind in cluster.hexes
                    if tuple(wind) == (0, 0, 0)]
    wrapping = [sites for sites, wind in cluster.hexes
                if tuple(wind) != (0, 0, 0)]
    winding4 = [sites for sites, _ in cluster.loops4]
    ice_index = {int(s): i for i, s in enumerate(ice_states)}

    n_flip = np.zeros(len(ice_states))
    for sites in contractible:
        for row, state in enumerate(ice_states):
            if loop_partner(int(state), sites) is not None:
                n_flip[row] += 1.0

    # columns spanning the flippability range, deterministically chosen
    picks = np.argsort(n_flip)[np.linspace(
        0, len(ice_states) - 1, min(n_columns, len(ice_states))).astype(int)]
    picks = np.unique(picks)

    print(f"{name}: {len(space)} states in {levels} hops, {len(ice_states)} ice,"
          f" {len(picks)} columns, n_flip in [{n_flip.min():.0f},"
          f" {n_flip.max():.0f}]", flush=True)

    for jpm in couplings:
        started = time.perf_counter()
        hamiltonian = build_hamiltonian(cluster, space, jpm).tocsc()
        energy = float(eigsh(hamiltonian, k=1, which="SA",
                             tol=1e-10, return_eigenvectors=False)[0])
        block = hamiltonian[complement][:, complement].tocsc()
        edge = float(eigsh(block, k=1, which="SA", tol=1e-9,
                           return_eigenvectors=False)[0])
        shifted = (block - energy * identity(block.shape[0],
                                             format="csc")).tocsc()
        ice_block = hamiltonian[ice_rows][:, ice_rows].toarray()
        coupling_block = hamiltonian[complement][:, ice_rows].tocsc()

        columns = {}
        for column in picks:
            rhs = np.asarray(coupling_block[:, [column]].todense()).ravel()
            solution, info = cg(shifted, rhs, rtol=1e-10, atol=0.0,
                                maxiter=20000)
            if info != 0:
                raise RuntimeError(f"CG failed at column {column}: {info}")
            columns[int(column)] = ice_block[:, column] \
                - coupling_block.T @ solution

        hexagon, wind_axial, wind_diagonal, wrap = [], [], [], []
        diagonal = []
        for column, values in columns.items():
            state = int(ice_states[column])
            diagonal.append((n_flip[column], float(values[column])))
            for sites in contractible:
                partner = loop_partner(state, sites)
                if partner is not None:
                    hexagon.append(float(values[ice_index[partner]]))
            for sites in winding4:
                partner = loop_partner(state, sites)
                if partner is not None:
                    windings = [w for s, w in cluster.loops4 if s == sites][0]
                    target = wind_axial if sorted(map(abs, windings)) \
                        == [0, 0, 1] else wind_diagonal
                    target.append(float(values[ice_index[partner]]))
            for sites in wrapping:
                partner = loop_partner(state, sites)
                if partner is not None:
                    wrap.append(float(values[ice_index[partner]]))

        diagonal = np.asarray(diagonal)
        design = np.column_stack([np.ones(len(diagonal)), diagonal[:, 0]])
        solution, *_ = np.linalg.lstsq(design, diagonal[:, 1], rcond=None)
        residual = diagonal[:, 1] - design @ solution
        v_eff = float(solution[1])
        v_r2 = float(1.0 - residual.var() / max(diagonal[:, 1].var(), 1e-300))
        g_eff = -float(np.mean(hexagon)) if hexagon else np.nan

        record = {
            "cluster": name, "levels": levels, "jpm": jpm,
            "space": len(space), "n_columns": len(picks),
            "energy": energy, "energy_per_site": energy / cluster.n_sites,
            "continuum_edge": edge, "two_spinon_gap": edge - energy,
            "g_eff": g_eff, "g_bare": 12.0 * abs(jpm) ** 3,
            "g_over_g6": g_eff / (12.0 * abs(jpm) ** 3),
            "g_spread": float(np.std(hexagon)) if hexagon else np.nan,
            "v_eff": v_eff, "v_r2": v_r2,
            "rk_ratio": v_eff / g_eff if g_eff else np.nan,
            "wind_axial": float(np.mean(wind_axial)) if wind_axial else np.nan,
            "wind_diagonal": float(np.mean(wind_diagonal))
            if wind_diagonal else np.nan,
            "wind_over_bare": (float(np.mean(np.abs(wind_axial)))
                               / (4.0 * jpm * jpm)) if wind_axial else np.nan,
            "wrapping_hexagon": float(np.mean(wrap)) if wrap else np.nan,
            "runtime": time.perf_counter() - started,
        }
        tag = f"{name}_L{levels}_{jpm:+.3f}".replace(".", "p")
        with open(OUTPUT / f"trunc_{tag}.json", "w") as handle:
            json.dump(record, handle, indent=1, default=float)
        print(f"{name} L{levels} jpm={jpm:+.3f} E/N={record['energy_per_site']:+.6f} "
              f"gap2s={record['two_spinon_gap']:.4f} "
              f"g={g_eff:+.4e} ({record['g_over_g6']:.3f} g6) "
              f"V={v_eff:+.4e} V/g={record['rk_ratio']:+.3f} "
              f"c4={record['wind_axial']:+.4e} "
              f"({record['wind_over_bare']:.3f} of 4J^2) "
              f"({record['runtime']:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
