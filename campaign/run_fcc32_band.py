#!/usr/bin/env python3
"""The winding-free band of the FCC-32 cluster from the truncated Feshbach map.

Cubic-16 is a poor host for the winding-free dynamics: its 90 ice states split
into 25 transport sectors of size at most 12, and every ice configuration has
either zero or exactly four flippable hexagons.  FCC-32 has 2970 ice states in
85 sectors, the largest of dimension 396, with flippabilities 0, 4, 6, 7, 8.

The full complement of the ice manifold is out of reach there, but the spinon
truncation validated on cubic-16 is not: restricting Q to the states reachable
in ``levels`` exchange hops, one conjugate-gradient solve per ice column builds
the entire ice-block map F(z = E_0).  Masking it (discarding matrix elements
between distinct polarization sectors) gives the winding-free effective
Hamiltonian, whose spectrum, ring heat-capacity peak and ground-multiplet
hexagon flux are reported here.

Usage: run_fcc32_band.py <cubic16|fcc32> <levels> <jpm> [<jpm> ...]
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.sparse import identity
from scipy.sparse.linalg import cg, eigsh

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "notes"))
sys.path.insert(0, str(ROOT / "campaign"))

import recompute_finite_size_artifact as geometry  # noqa: E402
from qsi_campaign.protocol import polarization_sector_labels  # noqa: E402
from run_truncated_feshbach import (  # noqa: E402
    CLUSTERS,
    bfs_space,
    build_hamiltonian,
    loop_partner,
    path_mask,
)
from run_feshbach_anatomy import band_peak  # noqa: E402

OUTPUT = ROOT / "campaign" / "outputs" / "fcc32_band"
OUTPUT.mkdir(parents=True, exist_ok=True)


def main() -> None:
    name = sys.argv[1]
    levels = int(sys.argv[2])
    couplings = [float(v) for v in sys.argv[3:]]
    basis, shape = CLUSTERS[name]
    cluster = geometry.build_cluster(basis, shape)

    space = bfs_space(cluster, levels)
    ice_states = np.sort(np.asarray(cluster.ice_states, dtype=np.uint64))
    ice_rows = np.searchsorted(space, ice_states)
    complement = np.setdiff1d(np.arange(len(space)), ice_rows)

    # polarization labels follow cluster.ice_states, so relabel for the sorted
    # ordering used here
    order = np.argsort(np.asarray(cluster.ice_states, dtype=np.uint64))
    labels = polarization_sector_labels(cluster)[order]
    same = labels[:, None] == labels[None, :]

    contractible = [sites for sites, wind in cluster.hexes
                    if tuple(wind) == (0, 0, 0)]
    index = {int(s): i for i, s in enumerate(ice_states)}
    hexagon = np.zeros((len(ice_states),) * 2)
    for sites in contractible:
        for row, state in enumerate(ice_states):
            partner = loop_partner(int(state), sites)
            if partner is not None:
                hexagon[row, index[partner]] += 1.0

    print(f"{name}: {len(space)} states, {len(ice_states)} ice, "
          f"{len(set(labels.tolist()))} transport sectors, "
          f"{len(contractible)} contractible hexagons", flush=True)

    for jpm in couplings:
        started = time.perf_counter()
        hamiltonian = build_hamiltonian(cluster, space, jpm).tocsc()
        energy = float(eigsh(hamiltonian, k=1, which="SA", tol=1e-10,
                             return_eigenvectors=False)[0])
        block = hamiltonian[complement][:, complement].tocsc()
        shifted = (block - energy * identity(block.shape[0],
                                             format="csc")).tocsc()
        coupling = hamiltonian[complement][:, ice_rows].tocsc()
        ice_block = hamiltonian[ice_rows][:, ice_rows].toarray()

        f_matrix = np.zeros_like(ice_block)
        for column in range(len(ice_rows)):
            rhs = np.asarray(coupling[:, [column]].todense()).ravel()
            solution, info = cg(shifted, rhs, rtol=1e-10, atol=0.0,
                                maxiter=20000)
            if info != 0:
                raise RuntimeError(f"CG failed on column {column}: {info}")
            f_matrix[:, column] = ice_block[:, column] - coupling.T @ solution
        f_matrix = 0.5 * (f_matrix + f_matrix.T)
        masked = np.where(same, f_matrix, 0.0)

        raw_levels = np.linalg.eigvalsh(f_matrix)
        values, vectors = np.linalg.eigh(masked)
        multiplet = vectors[:, values - values[0] < 1e-9]
        flux = float(np.real(np.einsum("in,ij,jn->", multiplet.conj(),
                                       hexagon, multiplet))
                     / (multiplet.shape[1] * len(contractible)))

        record = {
            "cluster": name, "levels": levels, "jpm": jpm,
            "space": len(space), "ice": len(ice_states),
            "energy": energy,
            "band_width": float(values[-1] - values[0]),
            "raw_band_width": float(raw_levels[-1] - raw_levels[0]),
            "ring_peak": band_peak(values),
            "ring_peak_over_g6": band_peak(values) / (12.0 * abs(jpm) ** 3),
            "raw_peak": band_peak(raw_levels),
            "raw_peak_over_g6": band_peak(raw_levels) / (12.0 * abs(jpm) ** 3),
            "gs_flux": flux,
            "degeneracy": int(multiplet.shape[1]),
            "runtime": time.perf_counter() - started,
        }
        tag = f"{name}_L{levels}_{jpm:+.3f}".replace(".", "p")
        with open(OUTPUT / f"band_{tag}.json", "w") as handle:
            json.dump(record, handle, indent=1, default=float)
        np.savez_compressed(OUTPUT / f"band_{tag}.npz", jpm=jpm,
                            masked_levels=values, raw_levels=raw_levels)
        print(f"{name} L{levels} jpm={jpm:+.3f} E={energy:+.5f} "
              f"peak/g6={record['ring_peak_over_g6']:.4f} "
              f"(raw {record['raw_peak_over_g6']:.4f}) "
              f"width={record['band_width']:.5f} flux={flux:+.4f} "
              f"deg={record['degeneracy']} ({record['runtime']:.0f}s)",
              flush=True)


if __name__ == "__main__":
    main()
