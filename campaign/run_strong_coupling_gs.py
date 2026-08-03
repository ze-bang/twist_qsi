#!/usr/bin/env python3
"""What the cubic-16 ground state actually does at large |J_pm|.

Frame-free, protocol-free ground-state characterization across the coupling
range where the winding-free band construction dies.  Per coupling we take the
ground multiplet of the exact Hamiltonian (translation-sector resolved, the
degenerate multiplet averaged) and measure

  * ice weight and mean number of charged tetrahedra (the dressing),
  * the hexagon flux <O_hex> and the winding four-cycle amplitude <O_4>,
    i.e. how much of the ground state's ring dynamics is physical and how
    much is the torus artifact,
  * the transverse one-body density matrix M_ij = <S^+_i S^-_j>, whose
    largest eigenvalue divided by N is the off-diagonal long-range order
    (condensate) fraction of the XY channel,
  * the ground-state energy and, by finite differences on the grid, its
    curvature -- the thermodynamic signature of a transition,
  * the low-level structure resolved by translation sector, so that level
    crossings are visible.

Usage: run_strong_coupling_gs.py <jpm> [<jpm> ...]
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

import recompute_finite_size_artifact as geometry  # noqa: E402
from qsi_campaign.exact_band import (  # noqa: E402
    fixed_magnetization_basis,
    microscopic_character_hamiltonian,
    translation_sector_bases,
    cubic16_translation_permutations,
)

OUTPUT = ROOT / "campaign" / "outputs" / "strong_coupling"
OUTPUT.mkdir(parents=True, exist_ok=True)
N_LEVELS = 24


def loop_expectation(states, index, sites, vectors):
    """<O_loop + h.c.> per loop, averaged over the supplied vectors."""
    mask = np.uint64(0)
    pattern = np.uint64(0)
    for position, site in enumerate(sites):
        mask |= np.uint64(1) << np.uint64(site)
        if position % 2 == 0:
            pattern |= np.uint64(1) << np.uint64(site)
    other = mask ^ pattern
    occupied = np.asarray(states, dtype=np.uint64) & mask
    active = np.where((occupied == pattern) | (occupied == other))[0]
    if active.size == 0:
        return 0.0
    partners = np.asarray([index[int(np.uint64(states[row]) ^ mask)]
                           for row in active])
    total = 0.0
    for column in range(vectors.shape[1]):
        vector = vectors[:, column]
        total += float(np.real(np.sum(vector[active].conj()
                                      * vector[partners])))
    return total / vectors.shape[1]


def transverse_density_matrix(states, index, n_sites, vectors):
    """M_ij = <S^+_i S^-_j> averaged over the supplied vectors."""
    states = np.asarray(states, dtype=np.uint64)
    matrix = np.zeros((n_sites, n_sites), dtype=complex)
    one = np.uint64(1)
    for i in range(n_sites):
        bit_i = one << np.uint64(i)
        up_i = (states & bit_i) != 0
        matrix[i, i] = float(np.mean([
            np.sum(np.abs(vectors[up_i, column]) ** 2)
            for column in range(vectors.shape[1])]))
        for j in range(n_sites):
            if i == j:
                continue
            bit_j = one << np.uint64(j)
            selection = np.where((~up_i) & ((states & bit_j) != 0))[0]
            if selection.size == 0:
                continue
            targets = np.asarray([index[int(states[row] ^ (bit_i | bit_j))]
                                  for row in selection])
            value = 0.0
            for column in range(vectors.shape[1]):
                vector = vectors[:, column]
                value += np.sum(vector[targets].conj() * vector[selection])
            matrix[i, j] = value / vectors.shape[1]
    return 0.5 * (matrix + matrix.conj().T)


def main() -> None:
    couplings = [float(v) for v in sys.argv[1:]]
    cluster = geometry.build_cluster("cubic", (1, 1, 1))
    states = fixed_magnetization_basis(cluster.n_sites, cluster.n_sites // 2)
    index = {int(s): i for i, s in enumerate(states)}
    sectors = translation_sector_bases(
        states, cubic16_translation_permutations(cluster))
    ice_rows = np.asarray([index[int(s)] for s in cluster.ice_states])
    defects = (np.bitwise_count(
        np.asarray(states, dtype=np.uint64)[:, None]
        & cluster.tet_masks[None, :]) != 2).sum(axis=1).astype(float)
    contractible = [sites for sites, wind in cluster.hexes
                    if tuple(wind) == (0, 0, 0)]
    winding4 = [sites for sites, _ in cluster.loops4]

    for jpm in couplings:
        started = time.perf_counter()
        hamiltonian = microscopic_character_hamiltonian(
            cluster, states, jpm, np.zeros(3))
        levels, vectors, tags = [], [], []
        for name, basis in sectors.items():
            block = (basis.conj().T @ (hamiltonian @ basis)).toarray()
            block = 0.5 * (block + block.conj().T)
            assert np.max(np.abs(block.imag)) < 1e-12
            block = np.ascontiguousarray(block.real)
            values, states_block = eigh(block,
                                        subset_by_index=(0, N_LEVELS - 1))
            levels.append(values)
            vectors.append(np.asarray(basis @ states_block))
            tags.extend([name] * len(values))
        levels = np.concatenate(levels)
        vectors = np.concatenate(vectors, axis=1)
        order = np.argsort(levels)
        levels = levels[order]
        vectors = vectors[:, order]
        tags = [tags[k] for k in order]

        multiplet = vectors[:, levels - levels[0] < 1e-9]
        gap = float(levels[multiplet.shape[1]] - levels[0])

        ice_weight = float(np.mean([
            np.sum(np.abs(multiplet[ice_rows, column]) ** 2)
            for column in range(multiplet.shape[1])]))
        defect_number = float(np.mean([
            np.sum(defects * np.abs(multiplet[:, column]) ** 2)
            for column in range(multiplet.shape[1])]))
        flux = float(np.mean([
            loop_expectation(states, index, sites, multiplet)
            for sites in contractible]))
        winding = float(np.mean([
            loop_expectation(states, index, sites, multiplet)
            for sites in winding4]))
        density = transverse_density_matrix(states, index, cluster.n_sites,
                                            multiplet)
        occupations = np.linalg.eigvalsh(density)[::-1]

        record = {
            "jpm": jpm,
            "energy": float(levels[0]),
            "degeneracy": int(multiplet.shape[1]),
            "gap": gap,
            "sector_tags": tags[:8],
            "ice_weight": ice_weight,
            "defect_number": defect_number,
            "hexagon_flux": flux,
            "winding4_amplitude": winding,
            "condensate_fraction": float(occupations[0] / cluster.n_sites),
            "occupation_spectrum": occupations.tolist(),
            "runtime": time.perf_counter() - started,
        }
        tag = f"{jpm:+.3f}".replace(".", "p")
        with open(OUTPUT / f"gs_{tag}.json", "w") as handle:
            json.dump(record, handle, indent=1, default=float)
        np.savez_compressed(OUTPUT / f"gs_{tag}.npz", jpm=jpm, levels=levels,
                            occupations=occupations, density=density)
        print(f"jpm={jpm:+.3f} E={levels[0]:+.6f} deg={record['degeneracy']} "
              f"gap={gap:.5f} ice={ice_weight:.4f} ndef={defect_number:.3f} "
              f"flux6={flux:+.4f} wind4={winding:+.4f} "
              f"n0/N={record['condensate_fraction']:.4f} "
              f"({record['runtime']:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
