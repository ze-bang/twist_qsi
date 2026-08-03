#!/usr/bin/env python3
"""Vison observables for the low spectrum as a function of Jpm.

For each coupling, diagonalizes H(0) (dense per translation sector) and
computes, for the lowest levels:

  * the per-hexagon flux fingerprint  f_h(n) = <n| O_h + O_h^dag |n>  over
    the 16 contractible hexagons (O_h the oriented ring product, the lattice
    realization of e^{iB_h}), reduced to its mean and its standard deviation
    over hexagons (flux roughness);
  * the vison-insertion spectral weights |<n| V |GS>|^2, where
    V = exp(i sum_i alpha_i S_i^z) is the diagonal Dirac-string unitary whose
    discrete curl threads 2pi flux through one chosen hexagon (a tight
    monopole pair on the dual lattice); alpha solves the curl equation in
    minimum norm.

Usage: run_jpm_vison.py <jpm> [<jpm> ...]
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from scipy.linalg import eigh
from scipy.sparse import csr_matrix

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "notes"))

import recompute_finite_size_artifact as geometry  # noqa: E402
from qsi_campaign.exact_band import (  # noqa: E402
    fixed_magnetization_basis,
    microscopic_character_hamiltonian,
    translation_sector_bases,
    cubic16_translation_permutations,
)

OUTPUT = ROOT / "campaign" / "outputs" / "jpm_levels"
OUTPUT.mkdir(parents=True, exist_ok=True)
N_KEEP = 150


def contractible_hexagons(cluster):
    return [sites for sites, winding in cluster.hexes
            if tuple(winding) == (0, 0, 0)]


def ring_operator(states, index, sites):
    """Sparse O_h + O_h^dag on the fixed-Sz basis for one hexagon."""
    mask = np.uint64(0)
    pattern_a = np.uint64(0)
    for position, site in enumerate(sites):
        mask |= np.uint64(1) << np.uint64(site)
        if position % 2 == 0:
            pattern_a |= np.uint64(1) << np.uint64(site)
    pattern_b = mask ^ pattern_a
    rows, cols = [], []
    for row, state in enumerate(states):
        occupied = np.uint64(state) & mask
        if occupied == pattern_a or occupied == pattern_b:
            rows.append(row)
            cols.append(index[int(np.uint64(state) ^ mask)])
    data = np.ones(len(rows))
    return csr_matrix((data, (rows, cols)), shape=(len(states),) * 2)


def dirac_string_phases(cluster, states, hexagons):
    """Diagonal phases of V = exp(i sum alpha_i S_i^z), curl alpha = 2pi e_h0."""
    curl = np.zeros((len(hexagons), cluster.n_sites))
    for row, sites in enumerate(hexagons):
        for position, site in enumerate(sites):
            curl[row, site] = 1.0 if position % 2 == 0 else -1.0
    # On a torus a single-valued string alpha can only realize monopole-free
    # flux shifts (rank(curl) = 6 of 16: the global twist-like channels): the
    # 2*pi Dirac string of a genuine vison pair is observable to the
    # half-integer charges, so an exact pair insertion is obstructed.  We
    # therefore insert the closest realizable approximation to a +-2*pi pair
    # (least-squares projection onto the realizable set) and report the
    # residual as the smearing of the inserted texture.
    target = np.zeros(len(hexagons))
    target[0] = 2.0 * np.pi
    target[8] = -2.0 * np.pi
    alpha, *_ = np.linalg.lstsq(curl, target, rcond=None)
    residual = float(np.max(np.abs(curl @ alpha - target)))
    bits = ((np.asarray(states, dtype=np.uint64)[:, None]
             >> np.arange(cluster.n_sites, dtype=np.uint64)) & np.uint64(1))
    spins = bits.astype(float) - 0.5
    return np.exp(1j * (spins @ alpha)), residual


def main() -> None:
    couplings = [float(v) for v in sys.argv[1:]]
    cluster = geometry.build_cluster("cubic", (1, 1, 1))
    states = fixed_magnetization_basis(cluster.n_sites, cluster.n_sites // 2)
    index = {int(s): i for i, s in enumerate(states)}
    sectors = translation_sector_bases(
        states, cubic16_translation_permutations(cluster))
    hexagons = contractible_hexagons(cluster)
    rings = [ring_operator(states, index, sites) for sites in hexagons]
    vison_phases, curl_residual = dirac_string_phases(cluster, states, hexagons)

    for jpm in couplings:
        started = time.perf_counter()
        hamiltonian = microscopic_character_hamiltonian(
            cluster, states, jpm, np.zeros(3))
        values, vectors = [], []
        for basis in sectors.values():
            block = (basis.conj().T @ (hamiltonian @ basis)).toarray()
            block = 0.5 * (block + block.conj().T)
            vals, vecs = eigh(block, subset_by_index=(0, N_KEEP - 1))
            values.append(vals)
            vectors.append(np.asarray(basis @ vecs))
        values = np.concatenate(values)
        vectors = np.concatenate(vectors, axis=1)
        order = np.argsort(values)[:N_KEEP]
        levels = values[order]
        psi = vectors[:, order]

        flux = np.stack([
            np.real(np.einsum("in,in->n", psi.conj(), ring @ psi))
            for ring in rings])
        ground = psi[:, 0]
        inserted = vison_phases * ground
        weights = np.abs(psi.conj().T @ inserted) ** 2

        tag = f"{jpm:+.3f}".replace(".", "p")
        np.savez_compressed(
            OUTPUT / f"vison_{tag}.npz",
            jpm=jpm, levels=levels,
            flux_mean=flux.mean(axis=0), flux_std=flux.std(axis=0),
            vison_weight=weights, captured=float(weights.sum()),
            curl_residual=curl_residual)
        print(f"jpm={jpm:+.3f} GSflux={flux[:,0].mean():+.4f} "
              f"captured={weights.sum():.3f} curl_res={curl_residual:.1e} "
              f"({time.perf_counter()-started:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
