#!/usr/bin/env python3
"""Which part of the corner-coherence loss is the artifact, and which the dressing.

Two independent effects stop a low-energy manifold from being corner-closed:

  (a) the winding processes themselves, which connect ice configurations in
      different transport sectors and therefore hybridise the sectors inside
      the band -- this scrambles individual eigenvectors while leaving the
      manifold closed;

  (b) virtual spinon pairs, whose polarization is a fraction of a period, which
      leak weight out of the manifold entirely.

Measuring the per-vector return |<v|U_c|v>|^2 and the manifold return
<v|U_c^dag P U_c|v> for the SAME band of H and of the winding-free Hamiltonian
H + C separates them: removing the winding restores the per-vector quantum
number, and what is left of the decay with coupling is the dressing.

Usage: run_corner_decomposition.py <jpm> [<jpm> ...]
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.linalg import eigh

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "notes"))
sys.path.insert(0, str(ROOT / "campaign"))

import recompute_finite_size_artifact as geometry  # noqa: E402
from qsi_campaign.exact_band import (  # noqa: E402
    basis_character_phases,
    fixed_magnetization_basis,
    microscopic_character_hamiltonian,
    translation_sector_bases,
    cubic16_translation_permutations,
    uniform_character_grid,
)
from qsi_campaign.protocol import polarization_sector_labels  # noqa: E402
from run_winding_residual import feshbach_matrix  # noqa: E402
from run_exact_counterterm import embed  # noqa: E402

OUTPUT = ROOT / "campaign" / "outputs" / "corner_decomposition"
OUTPUT.mkdir(parents=True, exist_ok=True)
N_BAND = 90


def transport_basis(levels, band, polarization, tolerance=1e-9):
    """Rotate each degenerate multiplet to eigenvectors of the polarization.

    Inside an exactly degenerate multiplet ``eigh`` returns an arbitrary basis,
    which can mix transport sectors that happen to be degenerate.  The physical
    per-vector question ("does this state have a transport label?") is only
    meaningful in the basis that diagonalizes the polarization operator within
    each multiplet, which is what this routine builds.
    """
    rotated = band.copy()
    start = 0
    while start < len(levels):
        stop = start + 1
        while stop < len(levels) and levels[stop] - levels[start] < tolerance:
            stop += 1
        if stop - start > 1:
            block = band[:, start:stop]
            operator = block.conj().T @ (polarization[:, None] * block)
            _, rotation = np.linalg.eigh(0.5 * (operator + operator.conj().T))
            rotated[:, start:stop] = block @ rotation
        start = stop
    return rotated


def returns(band, corner_phases):
    manifold, vector = [], []
    for phase in corner_phases:
        overlap = band.conj().T @ (phase[:, None] * band)
        manifold.append(np.real(np.einsum("nm,nm->n", overlap, overlap.conj())))
        vector.append(np.abs(np.diag(overlap)) ** 2)
    return float(np.mean(manifold)), float(np.mean(vector))


def main() -> None:
    couplings = [float(v) for v in sys.argv[1:]]
    cluster = geometry.build_cluster("cubic", (1, 1, 1))
    states = fixed_magnetization_basis(cluster.n_sites, cluster.n_sites // 2)
    index = {int(s): i for i, s in enumerate(states)}
    sectors = translation_sector_bases(
        states, cubic16_translation_permutations(cluster))
    ice_rows = np.asarray([index[int(s)] for s in cluster.ice_states])
    outside = np.ones(len(states), dtype=bool)
    outside[ice_rows] = False
    complement = np.where(outside)[0]
    labels = polarization_sector_labels(cluster)
    cross = labels[:, None] != labels[None, :]
    corner_phases = [basis_character_phases(cluster, states, np.asarray(t))
                     for t in uniform_character_grid(2)]
    bits = ((np.asarray(states, dtype=np.uint64)[:, None]
             >> np.arange(cluster.n_sites, dtype=np.uint64)) & np.uint64(1))
    mixed = (bits.astype(float) - 0.5) @ np.asarray(
        cluster.positions, dtype=float) @ np.array([1.0, np.sqrt(2.0),
                                                    np.sqrt(3.0)])

    def band_of(hamiltonian, count):
        values, vectors = [], []
        for basis in sectors.values():
            block = (basis.conj().T @ (hamiltonian @ basis)).toarray()
            block = np.ascontiguousarray(
                (0.5 * (block + block.conj().T)).real)
            val, vec = eigh(block, subset_by_index=(0, count - 1))
            values.append(val)
            vectors.append(np.asarray(basis @ vec))
        values = np.concatenate(values)
        vectors = np.concatenate(vectors, axis=1)
        order = np.argsort(values)[:count]
        return values[order], vectors[:, order]

    for jpm in couplings:
        started = time.perf_counter()
        bare = microscopic_character_hamiltonian(
            cluster, states, jpm, np.zeros(3)).tocsc().real
        levels, _ = band_of(bare, 1)
        anchor = float(levels[0])
        for _ in range(3):
            f_matrix = feshbach_matrix(bare, ice_rows, complement, anchor)
            corrected = (bare + embed(-np.where(cross, f_matrix, 0.0),
                                      ice_rows, len(states))).tocsc()
            new_levels, band_wf = band_of(corrected, N_BAND)
            if abs(new_levels[0] - anchor) < 1e-10:
                break
            anchor = float(new_levels[0])
        raw_levels, band_raw = band_of(bare, N_BAND)
        band_raw = transport_basis(raw_levels, band_raw, mixed)
        band_wf = transport_basis(new_levels, band_wf, mixed)

        raw_manifold, raw_vector = returns(band_raw, corner_phases)
        wf_manifold, wf_vector = returns(band_wf, corner_phases)
        raw_ice = float(np.mean(np.sum(np.abs(band_raw[ice_rows, :]) ** 2,
                                       axis=0)))
        wf_ice = float(np.mean(np.sum(np.abs(band_wf[ice_rows, :]) ** 2,
                                      axis=0)))

        record = {
            "jpm": jpm,
            "raw_manifold_return": raw_manifold,
            "raw_vector_return": raw_vector,
            "raw_band_ice": raw_ice,
            "wf_manifold_return": wf_manifold,
            "wf_vector_return": wf_vector,
            "wf_band_ice": wf_ice,
            "runtime": time.perf_counter() - started,
        }
        tag = f"{jpm:+.3f}".replace(".", "p")
        with open(OUTPUT / f"decomposition_{tag}.json", "w") as handle:
            json.dump(record, handle, indent=1, default=float)
        print(f"jpm={jpm:+.3f} raw: manifold={raw_manifold:.4f} "
              f"vector={raw_vector:.4f} ice={raw_ice:.4f} | "
              f"H+C: manifold={wf_manifold:.4f} vector={wf_vector:.4f} "
              f"ice={wf_ice:.4f} ({record['runtime']:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
