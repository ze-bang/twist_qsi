#!/usr/bin/env python3
"""Compute cubic-16 periodic and winding-free longitudinal DSSF."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.sparse.linalg import eigsh


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "notes"))

import recompute_finite_size_artifact as geometry  # noqa: E402
from qsi_campaign.exact_band import (  # noqa: E402
    cubic16_translation_permutations,
    fixed_magnetization_basis,
    microscopic_character_hamiltonian,
    translation_sector_bases,
)
from qsi_campaign.protocol import inverse_sqrt_hermitian  # noqa: E402


OUTPUT = ROOT / "campaign" / "outputs"
INPUT_STEM = OUTPUT / "nonperturbative_cubic16_p0p046000"


def zero_source_frame(cluster, states, ice_indices, jpm, report):
    """Reconstruct the exact canonical map from the ice basis to the band."""
    hamiltonian = microscopic_character_hamiltonian(
        cluster, states, jpm, np.zeros(3)
    )
    sectors = translation_sector_bases(
        states, cubic16_translation_permutations(cluster)
    )
    requested = report["zero_source"]["sector_eigenpairs"]
    candidates = []
    for name, sector in sectors.items():
        sector_hamiltonian = (sector.conj().T @ (hamiltonian @ sector)).tocsc()
        count = int(requested[name])
        values, vectors = eigsh(
            sector_hamiltonian.real,
            k=count,
            which="SA",
            tol=1.0e-11,
            ncv=min(sector_hamiltonian.shape[0], max(2 * count + 1, 80)),
        )
        order = np.argsort(values)
        values = np.asarray(values[order], dtype=float)
        vectors = np.asarray(vectors[:, order], dtype=np.complex128)
        full_vectors = np.asarray(sector @ vectors)
        candidates.extend(
            (float(value), name, full_vectors[:, index])
            for index, value in enumerate(values)
        )

    candidates.sort(key=lambda item: item[0])
    n_band = cluster.n_ice
    selected = candidates[: n_band + 1]
    if len(selected) != n_band + 1:
        raise RuntimeError("insufficient zero-source eigenpairs")
    energies = np.asarray([item[0] for item in selected[:n_band]])
    eigenvectors = np.column_stack([item[2] for item in selected[:n_band]])
    projected = eigenvectors[np.asarray(ice_indices), :]
    gram = projected.conj().T @ projected
    model_eigenvectors = projected @ inverse_sqrt_hermitian(gram)
    isometry = eigenvectors @ model_eigenvectors.conj().T
    operator = model_eigenvectors @ np.diag(energies) @ model_eigenvectors.conj().T
    operator = 0.5 * (operator + operator.conj().T)
    diagnostics = {
        "band_gap": float(selected[n_band][0] - selected[n_band - 1][0]),
        "model_overlap_min": float(np.linalg.eigvalsh(gram).min()),
        "isometry_error": float(
            np.linalg.norm(isometry.conj().T @ isometry - np.eye(n_band))
        ),
        "sector_counts": {
            name: sum(item[1] == name for item in selected[:n_band])
            for name in sectors
        },
    }
    return operator, isometry, diagnostics


def pulled_back_spin_sources(cluster, states, isometry, momenta):
    bits = np.uint64(1) << np.arange(cluster.n_sites, dtype=np.uint64)
    spins = ((states[:, None] & bits[None, :]) != 0).astype(float) - 0.5
    sublattices = np.arange(cluster.n_sites) % 4
    sources = np.empty(
        (len(momenta), 4, cluster.n_ice, cluster.n_ice), dtype=np.complex128
    )
    for momentum_index, momentum in enumerate(momenta):
        phase = np.exp(-2j * np.pi * (cluster.positions @ momentum))
        for sublattice in range(4):
            site_weights = phase * (sublattices == sublattice)
            diagonal = spins @ site_weights
            source = isometry.conj().T @ (diagonal[:, None] * isometry)
            sources[momentum_index, sublattice] = source
    return sources


def spectral_lines(operator, sources, n_sites, ground_tolerance=1.0e-9):
    energies, eigenvectors = np.linalg.eigh(operator)
    excitation = energies - energies[0]
    ground = np.flatnonzero(excitation <= ground_tolerance)
    weights = np.zeros((sources.shape[0], len(energies)), dtype=float)
    for momentum_index in range(sources.shape[0]):
        for sublattice in range(4):
            source_eigenbasis = (
                eigenvectors.conj().T
                @ sources[momentum_index, sublattice]
                @ eigenvectors
            )
            weights[momentum_index] += np.mean(
                np.abs(source_eigenbasis[:, ground]) ** 2, axis=1
            ) / n_sites
        weights[momentum_index, ground] = 0.0
    return excitation, weights, len(ground)


def broaden(excitation, weights, frequency, eta):
    distance = (frequency[:, None] - excitation[None, :]) / eta
    gaussian = np.exp(-0.5 * distance**2) / (np.sqrt(2.0 * np.pi) * eta)
    return gaussian @ weights.T


def main() -> None:
    report = json.loads(INPUT_STEM.with_suffix(".json").read_text())
    exact = np.load(INPUT_STEM.with_suffix(".npz"))
    jpm = float(report["jpm"])
    cluster = geometry.build_cluster("cubic", (1, 1, 1))
    states = fixed_magnetization_basis(cluster.n_sites, cluster.n_sites // 2)
    state_index = {int(state): index for index, state in enumerate(states)}
    ice_indices = np.asarray(
        [state_index[int(state)] for state in cluster.ice_states], dtype=np.int64
    )

    periodic_operator, isometry, frame_diagnostics = zero_source_frame(
        cluster, states, ice_indices, jpm, report
    )
    cached_periodic = np.asarray(exact["M1_operator"])
    reconstruction_error = float(
        np.linalg.norm(periodic_operator - cached_periodic)
        / np.linalg.norm(cached_periodic)
    )
    if reconstruction_error > 1.0e-8:
        raise RuntimeError(
            f"zero-source operator reconstruction failed: {reconstruction_error:.3e}"
        )

    momenta = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        )
    )
    labels = np.asarray((r"$\Gamma$", r"$X_x$", r"$X_y$", r"$X_z$"))
    sources = pulled_back_spin_sources(cluster, states, isometry, momenta)
    winding_free_operator = np.asarray(exact["M4_operator"])
    periodic_excitation, periodic_weight, periodic_ground = spectral_lines(
        periodic_operator, sources, cluster.n_sites
    )
    winding_free_excitation, winding_free_weight, winding_free_ground = spectral_lines(
        winding_free_operator, sources, cluster.n_sites
    )

    frequency = np.linspace(0.0, 0.14, 900)
    eta = 6.0e-4
    periodic_spectrum = broaden(
        periodic_excitation, periodic_weight, frequency, eta
    )
    winding_free_spectrum = broaden(
        winding_free_excitation, winding_free_weight, frequency, eta
    )

    def channel_summary(excitation, weight):
        summary = []
        for momentum_index in range(len(momenta)):
            active = weight[momentum_index] > 1.0e-10
            total = float(weight[momentum_index].sum())
            summary.append(
                {
                    "integrated_weight": total,
                    "first_active_frequency": float(
                        excitation[active].min() if np.any(active) else np.nan
                    ),
                    "centroid": float(
                        np.dot(excitation, weight[momentum_index]) / total
                        if total > 0.0
                        else np.nan
                    ),
                }
            )
        return summary

    output = OUTPUT / "dssf_cubic16_p0p046000.npz"
    np.savez_compressed(
        output,
        frequency=frequency,
        eta=np.asarray(eta),
        momenta=momenta,
        momentum_labels=labels,
        periodic_excitation=periodic_excitation,
        periodic_weights=periodic_weight,
        winding_free_excitation=winding_free_excitation,
        winding_free_weights=winding_free_weight,
        periodic_spectrum=periodic_spectrum,
        winding_free_spectrum=winding_free_spectrum,
    )
    summary = {
        "method": "zero-temperature four-sublattice-traced Szz in the exact cubic-16 ice-connected band",
        "probe": "same W0-dressed microscopic Sz operator for periodic and winding-free Hamiltonians",
        "coupling": jpm,
        "broadening_eta_over_Jzz": eta,
        "momenta_reciprocal_cubic_units": momenta.tolist(),
        "momentum_labels": labels.tolist(),
        "periodic_ground_multiplicity": periodic_ground,
        "winding_free_ground_multiplicity": winding_free_ground,
        "periodic_channels": channel_summary(periodic_excitation, periodic_weight),
        "winding_free_channels": channel_summary(
            winding_free_excitation, winding_free_weight
        ),
        "frame_diagnostics": {
            **frame_diagnostics,
            "cached_operator_relative_error": reconstruction_error,
        },
    }
    output.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
