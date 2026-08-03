#!/usr/bin/env python3
"""Ground-state observables and gauge coherence of the counterterm Hamiltonian.

``run_counterterm3.py`` tunes microscopic loop counterterms so that the EXACT
Feshbach amplitude of every winding cycle vanishes.  The result is a genuine
Hermitian Hamiltonian H_c whose ice-to-ice dynamics contains no boundary
transport at any coupling -- no band, no frame, no Gram matrix anywhere.  This
script asks what that Hamiltonian does where the frame-based constructions die:

  * the same ground-state observables as ``run_strong_coupling_gs.py``
    (ice weight, defect number, hexagon flux, winding amplitude, transverse
    condensate fraction), so that raw and winding-free ED can be compared
    observable by observable;

  * the corner-coherence spectrum of its lowest 90 states, i.e. whether a
    gauge-coherent manifold exists in the winding-free spectrum at couplings
    where the raw spectrum has none;

  * band isolation (the gap above the 90th level) and the ring heat-capacity
    peak.

Usage: run_counterterm_observables.py <jpm> [<jpm> ...]
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
from run_counterterm3 import (  # noqa: E402
    exact_winding_amplitude,
    loop_operator,
    winding_pair,
)
from run_strong_coupling_gs import (  # noqa: E402
    loop_expectation,
    transverse_density_matrix,
)
from run_feshbach_anatomy import band_peak  # noqa: E402

OUTPUT = ROOT / "campaign" / "outputs" / "counterterm_observables"
OUTPUT.mkdir(parents=True, exist_ok=True)
N_POOL = 150
N_BAND = 90


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

    axial = [sites for sites, wind in cluster.loops4
             if sorted(map(abs, wind)) == [0, 0, 1]]
    diagonal = [sites for sites, wind in cluster.loops4
                if sorted(map(abs, wind)) == [0, 1, 1]]
    wrapping = [sites for sites, wind in cluster.hexes
                if tuple(wind) != (0, 0, 0)]
    contractible = [sites for sites, wind in cluster.hexes
                    if tuple(wind) == (0, 0, 0)]
    class_operators = [sum(loop_operator(states, index, sites)
                           for sites in group)
                       for group in (axial, diagonal, wrapping)]
    pairs = [winding_pair(cluster, index, group[0])
             for group in (axial, diagonal, wrapping)]
    keep = np.ones(len(states), dtype=bool)
    keep[ice_rows] = False
    complement = np.where(keep)[0]
    corner_phases = [basis_character_phases(cluster, states, np.asarray(t))
                     for t in uniform_character_grid(2)]

    for jpm in couplings:
        started = time.perf_counter()
        bare = microscopic_character_hamiltonian(
            cluster, states, jpm, np.zeros(3)).tocsc().real

        anchor = min(
            float(eigh(np.ascontiguousarray(
                (basis.conj().T @ (bare @ basis)).toarray().real),
                subset_by_index=(0, 0), eigvals_only=True)[0])
            for basis in sectors.values())

        coefficients = np.array([4.0 * jpm * jpm, 4.0 * jpm * jpm,
                                 12.0 * abs(jpm) ** 3 * np.sign(-jpm)])
        for _ in range(40):
            trial = (bare + sum(value * operator for value, operator
                                in zip(coefficients, class_operators))).tocsc()
            amplitudes = np.array([
                exact_winding_amplitude(trial, *pair, anchor, complement)
                for pair in pairs])
            if np.max(np.abs(amplitudes)) < 1e-11:
                break
            coefficients = coefficients - amplitudes
        hamiltonian = (bare + sum(value * operator for value, operator
                                  in zip(coefficients,
                                         class_operators))).tocsc()

        values, vectors = [], []
        for basis in sectors.values():
            block = (basis.conj().T @ (hamiltonian @ basis)).toarray()
            block = 0.5 * (block + block.conj().T)
            block = np.ascontiguousarray(block.real)
            val, vec = eigh(block, subset_by_index=(0, N_POOL - 1))
            values.append(val)
            vectors.append(np.asarray(basis @ vec))
        values = np.concatenate(values)
        vectors = np.concatenate(vectors, axis=1)
        order = np.argsort(values)[:N_POOL]
        levels = values[order]
        pool = vectors[:, order]

        multiplet = pool[:, levels - levels[0] < 1e-9]
        band = pool[:, :N_BAND]

        ice_weight = float(np.mean([
            np.sum(np.abs(multiplet[ice_rows, column]) ** 2)
            for column in range(multiplet.shape[1])]))
        band_ice = float(np.mean([
            np.sum(np.abs(band[ice_rows, column]) ** 2)
            for column in range(N_BAND)]))
        defect_number = float(np.mean([
            np.sum(defects * np.abs(multiplet[:, column]) ** 2)
            for column in range(multiplet.shape[1])]))
        flux = float(np.mean([
            loop_expectation(states, index, sites, multiplet)
            for sites in contractible]))
        winding = float(np.mean([
            loop_expectation(states, index, sites, multiplet)
            for sites in axial + diagonal]))
        density = transverse_density_matrix(states, index, cluster.n_sites,
                                            multiplet)
        occupations = np.linalg.eigvalsh(density)[::-1]

        overlaps = [band.conj().T @ (phase[:, None] * band)
                    for phase in corner_phases]
        closure = np.mean([o @ o.conj().T for o in overlaps], axis=0)
        coherence = np.sort(np.linalg.eigvalsh(
            0.5 * (closure + closure.conj().T)))[::-1]
        sigma_min = min(float(svdvals(o).min()) for o in overlaps)

        record = {
            "jpm": jpm,
            "coefficients": coefficients.tolist(),
            "energy": float(levels[0]),
            "degeneracy": int(multiplet.shape[1]),
            "gap": float(levels[multiplet.shape[1]] - levels[0]),
            "band_gap": float(levels[N_BAND] - levels[N_BAND - 1]),
            "ice_weight": ice_weight,
            "band_ice_weight": band_ice,
            "defect_number": defect_number,
            "hexagon_flux": flux,
            "winding4_amplitude": winding,
            "condensate_fraction": float(occupations[0] / cluster.n_sites),
            "ring_peak": band_peak(levels[:N_BAND]),
            "ring_peak_over_g6": band_peak(levels[:N_BAND])
            / (12.0 * abs(jpm) ** 3),
            "coherence_min": float(coherence[-1]),
            "coherence_mean": float(coherence.mean()),
            "coherence_rank_0p8": int((coherence > 0.8).sum()),
            "corner_sigma_min": sigma_min,
            "runtime": time.perf_counter() - started,
        }
        tag = f"{jpm:+.3f}".replace(".", "p")
        with open(OUTPUT / f"ctobs_{tag}.json", "w") as handle:
            json.dump(record, handle, indent=1, default=float)
        np.savez_compressed(OUTPUT / f"ctobs_{tag}.npz", jpm=jpm,
                            levels=levels, coherence=coherence,
                            occupations=occupations,
                            coefficients=coefficients)
        print(f"jpm={jpm:+.3f} E={levels[0]:+.6f} ice={ice_weight:.4f} "
              f"band_ice={band_ice:.4f} flux6={flux:+.4f} "
              f"wind4={winding:+.4f} n0/N={record['condensate_fraction']:.4f} "
              f"peak/g6={record['ring_peak_over_g6']:.3f} "
              f"coh(>0.8)={record['coherence_rank_0p8']} "
              f"coh_min={record['coherence_min']:.3f} "
              f"bandgap={record['band_gap']:.4f} "
              f"({record['runtime']:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
