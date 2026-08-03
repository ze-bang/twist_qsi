#!/usr/bin/env python3
"""How much boundary transport is left after the counterterms, order by order.

Two frame-free winding-free protocols exist at arbitrary coupling:

  (a) the polarization mask applied to the exact Feshbach map, which annihilates
      every transporting matrix element of the ice block exactly, to all orders,
      but lives on the ice block and therefore inherits the residue Z;

  (b) the microscopic counterterm Hamiltonian H_c, which is a genuine operator
      on the whole Hilbert space -- observables, thermodynamics and spectra are
      all computable from it -- but whose counterterms cancel the transport of
      one loop class at a time.

This script quantifies the gap between them.  For H and for H_c it assembles
the full exact ice-block Feshbach matrix F(z=E_0) by one conjugate-gradient
solve per ice column, splits it into the transport-diagonal (physical) and
transport-changing (winding) parts, and reports

    residual = || cross-sector part of F || / || off-diagonal part of F ||

together with the change the counterterms induce in the physical part, which is
the price paid for the cancellation.

Usage: run_winding_residual.py <jpm> [<jpm> ...]
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.linalg import eigh
from scipy.sparse import identity
from scipy.sparse.linalg import cg

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "notes"))
sys.path.insert(0, str(ROOT / "campaign"))

import recompute_finite_size_artifact as geometry  # noqa: E402
from qsi_campaign.exact_band import (  # noqa: E402
    fixed_magnetization_basis,
    microscopic_character_hamiltonian,
    translation_sector_bases,
    cubic16_translation_permutations,
)
from qsi_campaign.protocol import polarization_sector_labels  # noqa: E402
from run_counterterm3 import (  # noqa: E402
    exact_winding_amplitude,
    loop_operator,
    winding_pair,
)

OUTPUT = ROOT / "campaign" / "outputs" / "winding_residual"
OUTPUT.mkdir(parents=True, exist_ok=True)


def feshbach_matrix(hamiltonian, ice_rows, complement, z):
    """Full ice-block F(z) by one CG solve per ice column."""
    block = hamiltonian[complement][:, complement].tocsc()
    shifted = (block - z * identity(block.shape[0], format="csc")).tocsc()
    coupling = hamiltonian[complement][:, ice_rows].tocsc()
    ice_block = hamiltonian[ice_rows][:, ice_rows].toarray()
    result = np.zeros_like(ice_block)
    for column in range(len(ice_rows)):
        rhs = np.asarray(coupling[:, [column]].todense()).ravel()
        solution, info = cg(shifted, rhs, rtol=1e-11, atol=0.0, maxiter=20000)
        if info != 0:
            raise RuntimeError(f"CG failed on column {column}: {info}")
        result[:, column] = ice_block[:, column] - coupling.T @ solution
    return 0.5 * (result + result.T)


def split(matrix, labels):
    same = labels[:, None] == labels[None, :]
    off = matrix.copy()
    np.fill_diagonal(off, 0.0)
    return np.where(same, off, 0.0), np.where(same, 0.0, off)


def main() -> None:
    couplings = [float(v) for v in sys.argv[1:]]
    cluster = geometry.build_cluster("cubic", (1, 1, 1))
    states = fixed_magnetization_basis(cluster.n_sites, cluster.n_sites // 2)
    index = {int(s): i for i, s in enumerate(states)}
    sectors = translation_sector_bases(
        states, cubic16_translation_permutations(cluster))
    ice_rows = np.asarray([index[int(s)] for s in cluster.ice_states])
    mask = np.ones(len(states), dtype=bool)
    mask[ice_rows] = False
    complement = np.where(mask)[0]
    labels = polarization_sector_labels(cluster)

    axial = [sites for sites, wind in cluster.loops4
             if sorted(map(abs, wind)) == [0, 0, 1]]
    diagonal = [sites for sites, wind in cluster.loops4
                if sorted(map(abs, wind)) == [0, 1, 1]]
    wrapping = [sites for sites, wind in cluster.hexes
                if tuple(wind) != (0, 0, 0)]
    class_operators = [sum(loop_operator(states, index, sites)
                           for sites in group)
                       for group in (axial, diagonal, wrapping)]
    pairs = [winding_pair(cluster, index, group[0])
             for group in (axial, diagonal, wrapping)]

    for jpm in couplings:
        started = time.perf_counter()
        bare = microscopic_character_hamiltonian(
            cluster, states, jpm, np.zeros(3)).tocsc().real
        energy = min(
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
                exact_winding_amplitude(trial, *pair, energy, complement)
                for pair in pairs])
            if np.max(np.abs(amplitudes)) < 1e-11:
                break
            coefficients = coefficients - amplitudes
        corrected = (bare + sum(value * operator for value, operator
                                in zip(coefficients, class_operators))).tocsc()

        f_bare = feshbach_matrix(bare, ice_rows, complement, energy)
        f_corrected = feshbach_matrix(corrected, ice_rows, complement, energy)
        physical_bare, winding_bare = split(f_bare, labels)
        physical_corrected, winding_corrected = split(f_corrected, labels)

        record = {
            "jpm": jpm,
            "energy": energy,
            "coefficients": coefficients.tolist(),
            "winding_fraction_bare": float(
                np.linalg.norm(winding_bare)
                / np.linalg.norm(winding_bare + physical_bare)),
            "winding_fraction_corrected": float(
                np.linalg.norm(winding_corrected)
                / np.linalg.norm(winding_corrected + physical_corrected)),
            "winding_suppression": float(
                np.linalg.norm(winding_corrected)
                / np.linalg.norm(winding_bare)),
            "physical_change": float(
                np.linalg.norm(physical_corrected - physical_bare)
                / np.linalg.norm(physical_bare)),
            "winding_norm_bare": float(np.linalg.norm(winding_bare)),
            "winding_norm_corrected": float(np.linalg.norm(winding_corrected)),
            "physical_norm": float(np.linalg.norm(physical_bare)),
            "masked_band_bare": np.linalg.eigvalsh(
                physical_bare + np.diag(np.diag(f_bare))).tolist(),
            "masked_band_corrected": np.linalg.eigvalsh(
                physical_corrected + np.diag(np.diag(f_corrected))).tolist(),
            "runtime": time.perf_counter() - started,
        }
        tag = f"{jpm:+.3f}".replace(".", "p")
        with open(OUTPUT / f"residual_{tag}.json", "w") as handle:
            json.dump(record, handle, indent=1, default=float)
        print(f"jpm={jpm:+.3f} winding frac {record['winding_fraction_bare']:.4f}"
              f" -> {record['winding_fraction_corrected']:.4f} "
              f"(suppression {record['winding_suppression']:.4f}, "
              f"physical part moved {record['physical_change']:.4f}) "
              f"({record['runtime']:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
