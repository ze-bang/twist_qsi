"""Nonperturbative twisted-band extraction for the cubic-16 QSI cluster."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, product
from math import ceil
from typing import Iterable

import numpy as np
from scipy.sparse import csc_matrix, coo_matrix
from scipy.sparse.linalg import eigsh

from .protocol import (
    band_operator_from_eigenvectors,
    band_operator_from_projected_vectors,
)


@dataclass(frozen=True)
class ExactBandResult:
    operator: np.ndarray
    eigenvalues: np.ndarray
    fixed_sz_gap_above_band: float
    diagnostics: dict[str, object]


def fixed_magnetization_basis(n_sites: int, n_up: int) -> np.ndarray:
    """Return bit states with exactly ``n_up`` set bits."""
    if not 0 <= n_up <= n_sites:
        raise ValueError("n_up must lie between zero and n_sites")
    states = []
    for occupied in combinations(range(n_sites), n_up):
        state = 0
        for site in occupied:
            state |= 1 << site
        states.append(state)
    return np.asarray(states, dtype=np.uint64)


def cubic16_translation_permutations(cluster) -> list[np.ndarray]:
    """Site permutations for the four FCC translations in the cubic cell."""
    if cluster.n_sites != 16:
        raise ValueError("the current translation decomposition is cubic-16 specific")
    translations = (
        np.array((0.0, 0.0, 0.0)),
        np.array((0.0, 0.5, 0.5)),
        np.array((0.5, 0.0, 0.5)),
        np.array((0.5, 0.5, 0.0)),
    )
    site_by_position = {
        tuple(np.rint(4.0 * np.mod(position, 1.0)).astype(int) % 4): site
        for site, position in enumerate(cluster.positions)
    }
    permutations = []
    for translation in translations:
        permutation = []
        for position in cluster.positions:
            key = tuple(
                np.rint(4.0 * np.mod(position + translation, 1.0)).astype(int) % 4
            )
            permutation.append(site_by_position[key])
        permutations.append(np.asarray(permutation, dtype=np.int64))
    return permutations


def _translate_state(state: int, permutation: np.ndarray) -> int:
    translated = 0
    for site, target in enumerate(permutation):
        if state & (1 << site):
            translated |= 1 << int(target)
    return translated


def translation_sector_bases(
    states: np.ndarray,
    permutations: Iterable[np.ndarray],
) -> dict[str, csc_matrix]:
    """Build orthonormal bases for the four irreps of the FCC Klein group."""
    permutations = list(permutations)
    if len(permutations) != 4:
        raise ValueError("exactly four Klein-group translations are required")
    characters = {
        "++": np.array((1, 1, 1, 1), dtype=np.complex128),
        "+-": np.array((1, 1, -1, -1), dtype=np.complex128),
        "-+": np.array((1, -1, 1, -1), dtype=np.complex128),
        "--": np.array((1, -1, -1, 1), dtype=np.complex128),
    }
    index = {int(state): position for position, state in enumerate(states)}
    transformed = np.empty((4, len(states)), dtype=np.int64)
    for group_index, permutation in enumerate(permutations):
        for state_index, state in enumerate(states):
            transformed[group_index, state_index] = index[
                _translate_state(int(state), permutation)
            ]

    sectors = {}
    for name, character in characters.items():
        visited = np.zeros(len(states), dtype=bool)
        rows: list[int] = []
        columns: list[int] = []
        values: list[complex] = []
        column = 0
        for state_index in range(len(states)):
            if visited[state_index]:
                continue
            orbit = set(int(transformed[g, state_index]) for g in range(4))
            visited[list(orbit)] = True
            coefficient_by_state: dict[int, complex] = {}
            for group_index, coefficient in enumerate(character):
                target = int(transformed[group_index, state_index])
                coefficient_by_state[target] = (
                    coefficient_by_state.get(target, 0.0j) + coefficient
                )
            norm = np.sqrt(sum(abs(value) ** 2 for value in coefficient_by_state.values()))
            if norm < 1.0e-13:
                continue
            for row, value in coefficient_by_state.items():
                rows.append(row)
                columns.append(column)
                values.append(value / norm)
            column += 1
        sectors[name] = coo_matrix(
            (values, (rows, columns)),
            shape=(len(states), column),
            dtype=np.complex128,
        ).tocsc()
    if sum(sector.shape[1] for sector in sectors.values()) != len(states):
        raise RuntimeError("translation sectors do not span the fixed-magnetization basis")
    return sectors


def uniform_character_grid(size: int) -> list[np.ndarray]:
    """Return the complete Z_M^3 character grid in radians."""
    if size < 1:
        raise ValueError("character-grid size must be positive")
    points = 2.0 * np.pi * np.arange(size, dtype=float) / size
    return [np.asarray(theta) for theta in product(points, repeat=3)]


def basis_character_phases(cluster, states: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Diagonal gauge unitary for the q=2*delta character source."""
    theta = np.asarray(theta, dtype=float)
    bits = np.uint64(1) << np.arange(cluster.n_sites, dtype=np.uint64)
    spins = ((states[:, None] & bits[None, :]) != 0).astype(float) - 0.5
    polarization = spins @ np.asarray(cluster.positions, dtype=float)
    return np.exp(-2j * (polarization @ theta))


def microscopic_character_hamiltonian(
    cluster,
    states: np.ndarray,
    jpm: float,
    theta: np.ndarray,
) -> csc_matrix:
    """Build H(theta) with the primitive ice-band character q=2*delta.

    A completed virtual path acquires ``exp(-i theta.q)``. For
    ``S_i^+ S_j^-``, delta = r_i-r_j+nL = -d_ij, so its microscopic phase is
    ``exp(+i theta.(2 d_ij))``. Individual hops can carry half-integer
    components; completed paths between ice states have integer q.
    """
    theta = np.asarray(theta, dtype=float)
    if theta.shape != (3,):
        raise ValueError("theta must have three components")
    state_index = {int(state): index for index, state in enumerate(states)}
    rows: list[int] = list(range(len(states)))
    columns: list[int] = list(range(len(states)))
    bits_on_tetrahedra = states[:, None] & cluster.tet_masks[None, :]
    tetrahedron_ups = np.bitwise_count(bits_on_tetrahedra).astype(np.int16)
    values: list[complex] = list(
        (0.5 * ((tetrahedron_ups - 2) ** 2).sum(axis=1)).astype(np.complex128)
    )

    one = np.uint64(1)
    for column, state in enumerate(states):
        for (left, right), image in zip(cluster.bonds, cluster.bond_wrap):
            left_bit = one << np.uint64(left)
            right_bit = one << np.uint64(right)
            left_up = bool(state & left_bit)
            right_up = bool(state & right_bit)
            if left_up == right_up:
                continue
            displacement = (
                cluster.positions[right]
                - cluster.positions[left]
                - np.asarray(image) @ cluster.Lvecs
            )
            phase = np.exp(1j * float(theta @ (2.0 * displacement)))
            amplitude = -jpm * (phase if (not left_up and right_up) else phase.conjugate())
            target_state = int(state ^ (left_bit | right_bit))
            rows.append(state_index[target_state])
            columns.append(column)
            values.append(amplitude)
    hamiltonian = coo_matrix(
        (np.asarray(values), (rows, columns)),
        shape=(len(states), len(states)),
        dtype=np.complex128,
    ).tocsc()
    return 0.5 * (hamiltonian + hamiltonian.conj().T)


def _lowest_eigenpairs(matrix: csc_matrix, count: int, tolerance: float):
    dimension = matrix.shape[0]
    count = min(count, dimension - 2)
    if count < 1:
        raise ValueError("sector is too small for sparse band extraction")
    imaginary_scale = np.max(np.abs(matrix.data.imag), initial=0.0)
    operator = matrix.real if imaginary_scale < 1.0e-14 else matrix
    values, vectors = eigsh(
        operator,
        k=count,
        which="SA",
        tol=tolerance,
        ncv=min(dimension, max(2 * count + 1, 80)),
    )
    order = np.argsort(values)
    return np.asarray(values[order], dtype=float), np.asarray(vectors[:, order])


def extract_exact_band(
    hamiltonian: csc_matrix,
    sectors: dict[str, csc_matrix],
    ice_indices: np.ndarray,
    n_band: int,
    *,
    tolerance: float = 1.0e-10,
    sector_margin: int = 8,
) -> ExactBandResult:
    """Extract and polar-pull back the lowest isolated microscopic band."""
    n_required = n_band + 1
    initial_count = ceil(n_required / len(sectors)) + sector_margin
    requested = {
        name: min(initial_count, sector.shape[1] - 2)
        for name, sector in sectors.items()
    }
    results: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    while True:
        for name, sector in sectors.items():
            if name in results and results[name][0].size == requested[name]:
                continue
            sector_hamiltonian = (sector.conj().T @ (hamiltonian @ sector)).tocsc()
            values, vectors = _lowest_eigenpairs(
                sector_hamiltonian, requested[name], tolerance
            )
            projected_vectors = sector[np.asarray(ice_indices), :] @ vectors
            results[name] = (values, np.asarray(projected_vectors))

        candidates = []
        for name, (values, projected) in results.items():
            candidates.extend(
                (float(value), name, projected[:, index])
                for index, value in enumerate(values)
            )
        candidates.sort(key=lambda item: item[0])
        selected = candidates[:n_required]
        if len(selected) < n_required:
            raise RuntimeError("not enough sector eigenpairs to identify the band gap")
        selected_counts = {
            name: sum(item[1] == name for item in selected) for name in sectors
        }
        saturated = [
            name for name in sectors if selected_counts[name] >= requested[name]
        ]
        if not saturated:
            break
        for name in saturated:
            maximum = sectors[name].shape[1] - 2
            enlarged = min(maximum, max(requested[name] + sector_margin, 2 * requested[name]))
            if enlarged == requested[name]:
                raise RuntimeError(f"sector {name} is saturated at its numerical limit")
            requested[name] = enlarged

    band_candidates = selected[:n_band]
    eigenvalues = np.asarray([item[0] for item in band_candidates])
    projected = np.column_stack([item[2] for item in band_candidates])
    operator, polar_diagnostics = band_operator_from_projected_vectors(
        eigenvalues, projected
    )
    diagnostics: dict[str, object] = {
        **polar_diagnostics,
        "sector_counts": {
            name: sum(item[1] == name for item in band_candidates) for name in sectors
        },
        "sector_eigenpairs": requested,
        "band_minimum": float(eigenvalues[0]),
        "band_maximum": float(eigenvalues[-1]),
    }
    return ExactBandResult(
        operator=operator,
        eigenvalues=eigenvalues,
        fixed_sz_gap_above_band=float(
            selected[n_band][0] - selected[n_band - 1][0]
        ),
        diagnostics=diagnostics,
    )


def extract_exact_band_full(
    hamiltonian: csc_matrix,
    ice_indices: np.ndarray,
    n_band: int,
    *,
    tolerance: float = 1.0e-10,
    max_iterations: int = 3000,
) -> ExactBandResult:
    """Extract a generic-source band without assuming translation symmetry."""
    n_required = n_band + 1
    values, vectors = eigsh(
        hamiltonian,
        k=n_required,
        which="SA",
        tol=tolerance,
        ncv=min(hamiltonian.shape[0], max(3 * n_required + 1, 300)),
        maxiter=max_iterations,
    )
    order = np.argsort(values)
    values = np.asarray(values[order], dtype=float)
    vectors = np.asarray(vectors[:, order], dtype=np.complex128)
    band_values = values[:n_band]
    band_vectors = vectors[:, :n_band]
    operator, polar_diagnostics = band_operator_from_eigenvectors(
        band_values, band_vectors, np.asarray(ice_indices)
    )
    residual = hamiltonian @ band_vectors - band_vectors * band_values[None, :]
    diagnostics: dict[str, object] = {
        **polar_diagnostics,
        "band_minimum": float(band_values[0]),
        "band_maximum": float(band_values[-1]),
        "maximum_ritz_residual": float(
            np.max(np.linalg.norm(residual, axis=0), initial=0.0)
        ),
    }
    return ExactBandResult(
        operator=operator,
        eigenvalues=band_values,
        fixed_sz_gap_above_band=float(values[n_band] - values[n_band - 1]),
        diagnostics=diagnostics,
    )
