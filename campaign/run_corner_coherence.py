#!/usr/bin/env python3
"""What corner coherence measures, and why transport quantization is collective.

The counting field enters as the diagonal unitary U(theta) = exp(-2i theta.p),
p = sum_i S^z_i r_i.  The eight gauge corners theta in {0,pi}^3 generate,
together with their products, the diagonal parity operators

    Lambda_alpha = exp(4 pi i p_alpha)   (alpha = x, y, z),

which are exactly +-1 on every basis state because 4 p_alpha is an integer:
they are the polarization-parity operators whose joint eigenspaces the M = 2
character average projects onto.

For a low-energy manifold with projector P = Psi Psi^dagger this script
measures, at each coupling:

  * the corner-closure (coherence) operator K = (1/8) sum_c A_c A_c^dagger,
    A_c = Psi^dagger U_c Psi.  An eigenvalue of K is the fraction of the
    corner image of that mode that stays inside the manifold: 1 means the
    corner action is a symmetry of the manifold, 0 means it maps the mode out.

  * the COLLECTIVE quantization: singular values of P Lambda_alpha P.  A
    manifold that is a direct sum of transport sectors has P Lambda P unitary,
    i.e. all singular values equal to one, whatever the individual vectors do.

  * the INDIVIDUAL quantization: |<v|Lambda_alpha|v>| for each manifold vector
    together with <p> and Var(p).  Virtual spinon pairs give every dressed
    vector a fractional polarization spread, so these are never quantized.

  * the greedy-selection experiment: remove the k vectors with the largest
    transport variance and recompute the coherence of the remainder, which is
    the quantitative form of "per-vector selection destroys the structure".

Usage: run_corner_coherence.py <jpm> [<jpm> ...]
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.linalg import eigh, svdvals

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "notes"))

import recompute_finite_size_artifact as geometry  # noqa: E402
from qsi_campaign.exact_band import (  # noqa: E402
    basis_character_phases,
    fixed_magnetization_basis,
    microscopic_character_hamiltonian,
    translation_sector_bases,
    cubic16_translation_permutations,
    uniform_character_grid,
)

OUTPUT = ROOT / "campaign" / "outputs" / "corner_coherence"
OUTPUT.mkdir(parents=True, exist_ok=True)
N_POOL = 150
N_BAND = 90


def main() -> None:
    couplings = [float(v) for v in sys.argv[1:]]
    cluster = geometry.build_cluster("cubic", (1, 1, 1))
    states = fixed_magnetization_basis(cluster.n_sites, cluster.n_sites // 2)
    sectors = translation_sector_bases(
        states, cubic16_translation_permutations(cluster))
    state_index = {int(s): i for i, s in enumerate(states)}

    bits = ((np.asarray(states, dtype=np.uint64)[:, None]
             >> np.arange(cluster.n_sites, dtype=np.uint64)) & np.uint64(1))
    polarization = (bits.astype(float) - 0.5) @ np.asarray(
        cluster.positions, dtype=float)
    parity = np.exp(4j * np.pi * polarization)          # (dim, 3), = +-1
    parity_error = float(np.max(np.abs(parity.imag)))
    parity = parity.real
    corner_phases = [basis_character_phases(cluster, states, np.asarray(t))
                     for t in uniform_character_grid(2)]

    for jpm in couplings:
        started = time.perf_counter()
        hamiltonian = microscopic_character_hamiltonian(
            cluster, states, jpm, np.zeros(3))
        values, vectors = [], []
        for basis in sectors.values():
            block = (basis.conj().T @ (hamiltonian @ basis)).toarray()
            block = 0.5 * (block + block.conj().T)
            assert np.max(np.abs(block.imag)) < 1e-12
            block = np.ascontiguousarray(block.real)
            val, vec = eigh(block, subset_by_index=(0, N_POOL - 1))
            values.append(val)
            vectors.append(np.asarray(basis @ vec))
        values = np.concatenate(values)
        vectors = np.concatenate(vectors, axis=1)
        order = np.argsort(values)[:N_POOL]
        energies = values[order]
        pool = vectors[:, order]
        band = pool[:, :N_BAND]

        def coherence(psi):
            overlaps = [psi.conj().T @ (phase[:, None] * psi)
                        for phase in corner_phases]
            operator = np.mean([o @ o.conj().T for o in overlaps], axis=0)
            return (np.sort(np.linalg.eigvalsh(
                0.5 * (operator + operator.conj().T)))[::-1],
                min(float(svdvals(o).min()) for o in overlaps))

        pool_spectrum, _ = coherence(pool)
        band_spectrum, sigma_min = coherence(band)

        # collective versus individual return under the corner action:
        # manifold_return_n = <v_n| U_c^dag P U_c |v_n> averaged over corners
        # (the diagonal of the closure operator: the fraction of the corner
        # image that stays in the manifold), against
        # vector_return_n = |<v_n| U_c |v_n>|^2 (the fraction that stays on
        # the same vector).  Their gap is the reshuffling.
        manifold_return, vector_return = [], []
        for phase in corner_phases:
            overlap = band.conj().T @ (phase[:, None] * band)
            manifold_return.append(np.real(np.einsum(
                "nm,nm->n", overlap, overlap.conj())))
            vector_return.append(np.abs(np.diag(overlap)) ** 2)
        manifold_return = np.mean(manifold_return, axis=0)
        vector_return = np.mean(vector_return, axis=0)
        ice_rows = np.asarray([state_index[int(s)]
                               for s in cluster.ice_states])
        band_ice = np.sum(np.abs(band[ice_rows, :]) ** 2, axis=0)

        # collective vs individual transport quantization on the band
        collective, individual = [], []
        for axis in range(3):
            projected = band.conj().T @ (parity[:, axis][:, None] * band)
            collective.append(svdvals(projected))
            individual.append(np.abs(np.einsum(
                "in,i,in->n", band.conj(), parity[:, axis], band)))
        collective = np.asarray(collective)
        individual = np.asarray(individual)

        weights = np.array([1.0, np.sqrt(2.0), np.sqrt(3.0)])
        mixed = polarization @ weights
        mean_p = np.einsum("in,i,in->n", band.conj(), mixed, band).real
        var_p = np.einsum("in,i,in->n", band.conj(), mixed**2, band).real \
            - mean_p**2

        # greedy per-vector selection: drop the most fluctuating vectors
        greedy = []
        drop_order = np.argsort(var_p)[::-1]
        for dropped in (0, 1, 2, 5, 10, 20):
            keep = np.sort(drop_order[dropped:])
            spectrum, sigma = coherence(band[:, keep])
            greedy.append({
                "dropped": dropped,
                "rank": len(keep),
                "coherence_min": float(spectrum[-1]),
                "coherence_mean": float(spectrum.mean()),
                "sigma_min": sigma,
            })

        record = {
            "jpm": jpm,
            "parity_imaginary_residue": parity_error,
            "band_gap": float(energies[N_BAND] - energies[N_BAND - 1]),
            "coherence_rank_0p8": int((pool_spectrum > 0.8).sum()),
            "coherence_rank_0p5": int((pool_spectrum > 0.5).sum()),
            "band_coherence_min": float(band_spectrum[-1]),
            "band_coherence_mean": float(band_spectrum.mean()),
            "corner_sigma_min": sigma_min,
            "collective_sigma_min": float(collective.min()),
            "collective_sigma_mean": float(collective.mean()),
            "individual_min": float(individual.min()),
            "individual_mean": float(individual.mean()),
            "manifold_return_mean": float(manifold_return.mean()),
            "vector_return_mean": float(vector_return.mean()),
            "vector_return_max": float(vector_return.max()),
            "band_ice_weight": float(band_ice.mean()),
            "band_coherence_mean_over_ice": float(band_spectrum.mean()
                                                  / band_ice.mean()),
            "transport_variance_mean": float(var_p.mean()),
            "transport_variance_max": float(var_p.max()),
            "greedy": greedy,
            "runtime": time.perf_counter() - started,
        }
        tag = f"{jpm:+.3f}".replace(".", "p")
        with open(OUTPUT / f"coherence_{tag}.json", "w") as handle:
            json.dump(record, handle, indent=1, default=float)
        np.savez_compressed(
            OUTPUT / f"coherence_{tag}.npz", jpm=jpm, energies=energies,
            pool_spectrum=pool_spectrum, band_spectrum=band_spectrum,
            collective=collective, individual=individual,
            mean_p=mean_p, var_p=var_p, manifold_return=manifold_return,
            vector_return=vector_return, band_ice=band_ice)
        print(f"jpm={jpm:+.3f} rank(>0.8)={record['coherence_rank_0p8']} "
              f"band_coh_min={record['band_coherence_min']:.4f} "
              f"sigma_min={sigma_min:.4f} "
              f"collective_sv_min={record['collective_sigma_min']:.4f} "
              f"ret_manifold={record['manifold_return_mean']:.4f} "
              f"ret_vector={record['vector_return_mean']:.4f} "
              f"band_ice={record['band_ice_weight']:.4f} "
              f"var_p_mean={record['transport_variance_mean']:.4f} "
              f"({record['runtime']:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
