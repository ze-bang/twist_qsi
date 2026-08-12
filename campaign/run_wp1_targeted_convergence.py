#!/usr/bin/env python3
"""Process-targeted FCC-32 loop amplitudes for a depth convergence test.

Unlike the historical sampled calculation, this solver does not discard loop
partners outside a principal ice submatrix.  It chooses a deterministic,
uniform sample of ice columns, folds each column through the full depth-d
microscopic space, and retains every transport-preserving simple-loop partner
in the complete 2970-state ice manifold.  The same columns and the same anchor
are used at every depth, so d is the only changing numerical parameter.

Usage:
  run_wp1_targeted_convergence.py <depth> <jpm> <anchor|auto> \
      [--ncolumns=32] [--chunk=1] [--resume]

With ``auto`` the anchor is the lowest eigenvalue of the same depth-truncated
Hamiltonian.  This is the appropriate mode for physical depth convergence;
a numerical anchor is retained for fixed-resolvent diagnostics.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "notes"))
sys.path.insert(0, str(ROOT / "campaign"))

import recompute_finite_size_artifact as geometry  # noqa: E402
from qsi_campaign.protocol import polarization_sector_labels  # noqa: E402
from qsi_campaign.tomography import (  # noqa: E402
    classify_flip,
    graph_distance_matrix,
    minimum_image_distance_matrix,
)
from run_multi_anchor_band import block_cg  # noqa: E402
from run_truncated_feshbach import bfs_space, build_hamiltonian  # noqa: E402

OUTPUT = ROOT / "campaign" / "outputs" / "wp1_tomography"
PERIMETERS = (6, 8, 10, 12)
SEED = 20260811
BOOTSTRAPS = 4000


def build_hamiltonian_compact(cluster, space: np.ndarray, jpm: float) -> csr_matrix:
    """Build the restricted XXZ Hamiltonian without a giant COO transient.

    The standard builder collects one row/column/value array per bond and then
    concatenates them.  At depth four that temporary representation exceeds a
    249-GiB node even though the final CSR matrix fits.  Here we count the
    allowed degree of every row, allocate CSR storage once, and fill it bond by
    bond.  The resulting operator is algebraically identical.
    """
    nstates = len(space)
    degree = np.zeros(nstates, dtype=np.uint8)

    def targets(left: int, right: int) -> tuple[np.ndarray, np.ndarray]:
        flip = (np.uint64(1) << np.uint64(left)) | (
            np.uint64(1) << np.uint64(right))
        exchange = ((space >> np.uint64(left)) ^
                    (space >> np.uint64(right))) & np.uint64(1)
        source = np.flatnonzero(exchange).astype(np.int64, copy=False)
        target_states = space[source] ^ flip
        position = np.searchsorted(space, target_states).astype(
            np.int64, copy=False)
        inside = position < nstates
        inside[inside] &= space[position[inside]] == target_states[inside]
        return source[inside], position[inside]

    print("  compact CSR pass 1/2: counting row degrees", flush=True)
    for bond_index, (left, right) in enumerate(cluster.bonds, start=1):
        source, _ = targets(int(left), int(right))
        degree[source] += np.uint8(1)
        if bond_index % 8 == 0:
            print(f"    bonds {bond_index}/{len(cluster.bonds)}", flush=True)

    indptr = np.empty(nstates + 1, dtype=np.int64)
    indptr[0] = 0
    np.cumsum(degree.astype(np.int64) + 1, out=indptr[1:])
    nnz = int(indptr[-1])
    print(f"  allocating compact CSR with {nnz:,} entries", flush=True)
    indices = np.empty(nnz, dtype=np.int64)
    data = np.empty(nnz, dtype=float)
    cursor = indptr[:-1].copy()

    diagonal = np.zeros(nstates, dtype=float)
    for mask in np.asarray(cluster.tet_masks, dtype=np.uint64):
        charge = np.bitwise_count(space & mask).astype(np.int8) - 2
        diagonal += 0.5 * charge.astype(float) ** 2
    row = np.arange(nstates, dtype=np.int64)
    indices[cursor] = row
    data[cursor] = diagonal
    cursor += 1
    del diagonal, row

    print("  compact CSR pass 2/2: filling exchange edges", flush=True)
    for bond_index, (left, right) in enumerate(cluster.bonds, start=1):
        source, target = targets(int(left), int(right))
        location = cursor[source]
        indices[location] = target
        data[location] = -jpm
        cursor[source] += 1
        if bond_index % 8 == 0:
            print(f"    bonds {bond_index}/{len(cluster.bonds)}", flush=True)
    if not np.array_equal(cursor, indptr[1:]):
        raise RuntimeError("compact CSR fill count is inconsistent")
    return csr_matrix((data, indices, indptr), shape=(nstates, nstates),
                      copy=False)


def main() -> int:
    positional = [token for token in sys.argv[1:] if not token.startswith("--")]
    if len(positional) != 3:
        raise SystemExit(__doc__)
    depth, jpm = int(positional[0]), float(positional[1])
    automatic_anchor = positional[2].lower() == "auto"
    anchor = np.nan if automatic_anchor else float(positional[2])
    ncolumns = 32
    selection_pool = None
    chunk = 1
    resume = "--resume" in sys.argv
    for token in sys.argv[1:]:
        if token.startswith("--ncolumns="):
            ncolumns = int(token.split("=", 1)[1])
        elif token.startswith("--selection-pool="):
            selection_pool = int(token.split("=", 1)[1])
        elif token.startswith("--chunk="):
            chunk = int(token.split("=", 1)[1])

    started = time.perf_counter()
    cluster = geometry.build_cluster("fcc", (2, 2, 2))
    ice_states = np.asarray(cluster.ice_states, dtype=np.uint64)
    labels = polarization_sector_labels(cluster)
    rng = np.random.default_rng(SEED)
    pool = ncolumns if selection_pool is None else selection_pool
    if pool < ncolumns:
        raise ValueError("selection pool must be at least ncolumns")
    selected = np.sort(rng.choice(len(ice_states), size=pool, replace=False))[
        :ncolumns]

    graph_distance = graph_distance_matrix(cluster)
    real_distance = minimum_image_distance_matrix(cluster)
    partners: dict[int, list[tuple[int, int]]] = {length: [] for length in PERIMETERS}
    for local_column, ice_column in enumerate(selected):
        state = int(ice_states[ice_column])
        for ice_row, other in enumerate(ice_states):
            if ice_row == ice_column or labels[ice_row] != labels[ice_column]:
                continue
            mask = state ^ int(other)
            length = mask.bit_count()
            if length not in partners:
                continue
            perimeter, components, simple, _, _ = classify_flip(
                cluster, mask, graph_distance, real_distance)
            if perimeter == length and components == 1 and simple:
                partners[length].append((ice_row, local_column))
    print(
        f"target columns={ncolumns}, pair counts="
        + ", ".join(f"L{length}:{len(partners[length])}" for length in PERIMETERS),
        flush=True,
    )
    if any(not partners[length] for length in PERIMETERS):
        raise RuntimeError("target sample misses a required loop perimeter")

    print(f"building FCC-32 depth-{depth} space", flush=True)
    space = bfs_space(cluster, depth)
    ice_rows = np.searchsorted(space, ice_states)
    if not np.array_equal(space[ice_rows], ice_states):
        raise RuntimeError("ice manifold is not contained in the BFS space")
    print(f"space={len(space):,}; building sparse Hamiltonian", flush=True)
    if "--compact" in sys.argv:
        hamiltonian = build_hamiltonian_compact(cluster, space, jpm)
    else:
        hamiltonian = build_hamiltonian(cluster, space, jpm).tocsr()
    print(f"sparse Hamiltonian ready: nnz={hamiltonian.nnz:,}", flush=True)
    if automatic_anchor:
        print("determining depth-matched ground-energy anchor", flush=True)
        anchor = float(eigsh(
            hamiltonian, k=1, which="SA", tol=1.0e-9,
            return_eigenvectors=False)[0])
        print(f"depth-matched anchor={anchor:.15g}", flush=True)
    ice_to_all = hamiltonian[ice_rows, :].tocsr()
    print("ice-to-microscopic coupling rows extracted", flush=True)

    tag = (
        f"target_fcc32_d{depth}_{jpm:+.3f}_n{ncolumns}_z{anchor:+.6f}"
        .replace(".", "p")
    )
    checkpoint = OUTPUT / f"ckpt_{tag}.npz"
    folded_columns = np.zeros((len(ice_states), ncolumns), dtype=float)
    done = 0
    if resume and checkpoint.exists():
        with np.load(checkpoint) as saved:
            if (
                int(saved["depth"]) == depth
                and float(saved["jpm"]) == jpm
                and float(saved["anchor"]) == anchor
                and np.array_equal(saved["selected"], selected)
            ):
                folded_columns = saved["folded_columns"]
                done = int(saved["done"])
                print(f"resumed {done}/{ncolumns} columns", flush=True)

    def shifted_q_matvec(vectors: np.ndarray) -> np.ndarray:
        output = hamiltonian @ vectors - anchor * vectors
        output[ice_rows, :] = 0.0
        return output

    for start in range(done, ncolumns, chunk):
        stop = min(start + chunk, ncolumns)
        # H is real symmetric.  Reading the corresponding sparse P rows and
        # transposing them is identical to H[:, P], but avoids a full scan of
        # all 3.8 billion CSR column indices at depth four.
        rhs = np.asarray(
            ice_to_all[selected[start:stop], :].T.todense())
        rhs[ice_rows, :] = 0.0
        solution = block_cg(
            shifted_q_matvec, rhs, np.zeros_like(rhs), tol=1.0e-9)
        folded_columns[:, start:stop] = -(ice_to_all @ solution)
        np.savez_compressed(
            checkpoint,
            folded_columns=folded_columns,
            done=stop,
            depth=depth,
            jpm=jpm,
            anchor=anchor,
            selected=selected,
        )
        print(f"columns {stop}/{ncolumns}", flush=True)

    # Per-column sufficient statistics preserve correlations between the
    # different perimeters in the paired bootstrap ratios.
    sums = np.zeros((ncolumns, len(PERIMETERS)), dtype=float)
    counts = np.zeros((ncolumns, len(PERIMETERS)), dtype=np.int64)
    for perimeter_index, length in enumerate(PERIMETERS):
        for ice_row, local_column in partners[length]:
            sums[local_column, perimeter_index] += abs(
                folded_columns[ice_row, local_column])
            counts[local_column, perimeter_index] += 1

    total_count = counts.sum(axis=0)
    amplitudes = sums.sum(axis=0) / total_count
    bootstrap_rng = np.random.default_rng(SEED + 1)
    bootstrap = np.empty((BOOTSTRAPS, len(PERIMETERS)), dtype=float)
    for sample_index in range(BOOTSTRAPS):
        pick = bootstrap_rng.integers(0, ncolumns, size=ncolumns)
        sample_count = counts[pick].sum(axis=0)
        bootstrap[sample_index] = sums[pick].sum(axis=0) / sample_count
    ratios = amplitudes[1:] / amplitudes[0]
    bootstrap_ratios = bootstrap[:, 1:] / bootstrap[:, :1]

    record: dict[str, object] = {
        "cluster": "fcc32",
        "method": "targeted_depth_convergence",
        "virtual_depth": depth,
        "jpm": jpm,
        "common_anchor": anchor,
        "anchor_mode": "depth_matched_ground" if automatic_anchor else "fixed",
        "space_size": int(len(space)),
        "n_ice": int(len(ice_states)),
        "n_columns": ncolumns,
        "column_seed": SEED,
        "selected_columns": selected.tolist(),
        "runtime": time.perf_counter() - started,
    }
    for index, length in enumerate(PERIMETERS):
        record[f"K::{length}"] = {
            "K": float(amplitudes[index]),
            "count": int(total_count[index]),
            "bootstrap_ci95": np.percentile(
                bootstrap[:, index], [2.5, 97.5]).tolist(),
        }
        if length != 6:
            ratio_index = index - 1
            record[f"k{length}_over_k6"] = float(ratios[ratio_index])
            record[f"k{length}_over_k6_ci95"] = np.percentile(
                bootstrap_ratios[:, ratio_index], [2.5, 97.5]).tolist()

    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / f"{tag}.json").write_text(json.dumps(record, indent=1))
    np.savez_compressed(
        OUTPUT / f"{tag}.npz",
        folded_columns=folded_columns,
        selected_columns=selected,
        sums=sums,
        counts=counts,
    )
    checkpoint.unlink(missing_ok=True)
    print(json.dumps(record, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
