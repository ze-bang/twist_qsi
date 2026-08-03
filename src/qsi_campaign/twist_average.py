"""Observable-level twist averaging of the full cubic-16 XXZ spectrum.

This is the direct exploration of the original idea: solve the microscopic
Hamiltonian exactly under a smooth-gauge twist ``theta`` (every ``S+S-`` hop
carries ``exp(2i theta . d)`` with ``d`` the lifted bond displacement), compute
the observable per twist, and average over the twist grid.  No band isolation,
no ice Gram, no stitching — defined at every coupling.

Symmetry facts used (each is verified by ``campaign/run_twist_average.py
--verify`` before production):

* ``Jpmpm = 0`` conserves total ``S^z`` and the twisted Hamiltonian still
  commutes with the four Klein translations, so the 65,536-state spectrum
  splits into ``17 x 4`` dense Hermitian blocks (largest ~3,218).
* Corner shifts are pure gauge: ``H(theta + c) = V(c) H(theta) V(c)^dag`` for
  ``c in {0, pi}^3`` (Thm A extended to the microscopic level), so distinct
  spectra live in ``theta in [0, pi)^3``.
* ``H(-theta) = H(theta)*`` gives equal spectra, pairing ``theta`` with its
  reflection in the reduced cube.
* The global spin flip maps the ``n_up`` sector onto ``16 - n_up`` with a
  conjugated (isospectral) block, halving the sector work.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix

from .exact_band import (
    cubic16_translation_permutations,
    fixed_magnetization_basis,
    translation_sector_bases,
)

N_SITES = 16

try:  # dense complex eigvalsh is 8x faster on the GPU for the big blocks
    import cupy as _cupy

    _cupy.cuda.runtime.getDeviceCount()
except Exception:  # pragma: no cover - GPU is an optional accelerator
    _cupy = None

_GPU_MIN_DIMENSION = 512


@dataclass(frozen=True)
class SectorStructure:
    """Twist-independent geometry of one fixed-``S^z`` sector."""

    n_up: int
    states: np.ndarray
    diagonal: np.ndarray
    hop_rows: np.ndarray
    hop_columns: np.ndarray
    hop_displacements: np.ndarray  # lifted r_target_up - r_source_up per hop
    sector_bases: list[csr_matrix]


def _sector_structure(cluster, n_up: int) -> SectorStructure:
    states = np.sort(fixed_magnetization_basis(cluster.n_sites, n_up))
    bits = np.uint64(1) << np.arange(cluster.n_sites, dtype=np.uint64)

    tet_bits = states[:, None] & np.asarray(cluster.tet_masks, dtype=np.uint64)[None, :]
    tet_ups = np.bitwise_count(tet_bits).astype(np.int64)
    diagonal = 0.5 * ((tet_ups - 2) ** 2).sum(axis=1).astype(np.float64)

    rows: list[np.ndarray] = []
    columns: list[np.ndarray] = []
    displacements: list[np.ndarray] = []
    for (left, right), image in zip(cluster.bonds, cluster.bond_wrap):
        lifted_right = cluster.positions[right] - np.asarray(image) @ cluster.Lvecs
        displacement = lifted_right - cluster.positions[left]
        left_up = (states & bits[left]) != 0
        right_up = (states & bits[right]) != 0
        flip = np.uint64(int(bits[left]) | int(bits[right]))
        for source_mask, hop_vector in (
            (~left_up & right_up, displacement),  # S+_left S-_right
            (left_up & ~right_up, -displacement),
        ):
            sources = np.flatnonzero(source_mask)
            if sources.size == 0:
                continue
            targets = np.searchsorted(states, states[sources] ^ flip)
            rows.append(targets)
            columns.append(sources)
            displacements.append(np.broadcast_to(hop_vector, (sources.size, 3)))

    bases = [
        basis.tocsr()
        for basis in translation_sector_bases(
            states, cubic16_translation_permutations(cluster)
        ).values()
    ]
    return SectorStructure(
        n_up=n_up,
        states=states,
        diagonal=diagonal,
        hop_rows=np.concatenate(rows) if rows else np.empty(0, dtype=np.int64),
        hop_columns=np.concatenate(columns) if columns else np.empty(0, dtype=np.int64),
        hop_displacements=(
            np.concatenate(displacements) if displacements else np.empty((0, 3))
        ),
        sector_bases=bases,
    )


def build_structures(cluster) -> list[SectorStructure]:
    """Structures for ``n_up = 0..8``; the flip mirror covers ``9..16``."""
    if cluster.n_sites != N_SITES:
        raise ValueError("the Klein-sector decomposition is cubic-16 specific")
    return [_sector_structure(cluster, n_up) for n_up in range(N_SITES // 2 + 1)]


def _eigvalsh(block: np.ndarray) -> np.ndarray:
    if _cupy is not None and block.shape[0] >= _GPU_MIN_DIMENSION:
        return _cupy.asnumpy(_cupy.linalg.eigvalsh(_cupy.asarray(block)))
    return np.linalg.eigvalsh(block)


def sector_spectrum(structure: SectorStructure, jpm: float, theta: np.ndarray) -> np.ndarray:
    """All eigenvalues of one fixed-``S^z`` sector of the twisted Hamiltonian."""
    theta = np.asarray(theta, dtype=float)
    dimension = len(structure.states)
    amplitudes = -jpm * np.exp(2j * (structure.hop_displacements @ theta))
    hamiltonian = coo_matrix(
        (
            np.concatenate([structure.diagonal.astype(np.complex128), amplitudes]),
            (
                np.concatenate([np.arange(dimension), structure.hop_rows]),
                np.concatenate([np.arange(dimension), structure.hop_columns]),
            ),
        ),
        shape=(dimension, dimension),
    ).tocsr()
    values = []
    for basis in structure.sector_bases:
        block = (basis.conj().T @ (hamiltonian @ basis)).toarray()
        values.append(_eigvalsh(block))
    return np.sort(np.concatenate(values))


def full_spectrum(structures: list[SectorStructure], jpm: float, theta: np.ndarray) -> np.ndarray:
    """The complete 65,536-state spectrum at one coupling and twist."""
    parts = []
    for structure in structures:
        spectrum = sector_spectrum(structure, jpm, theta)
        parts.append(spectrum)
        if 2 * structure.n_up != N_SITES:  # mirror sector 16 - n_up
            parts.append(spectrum)
    return np.sort(np.concatenate(parts))


def reduced_twist_grid(size: int) -> list[tuple[np.ndarray, int]]:
    """Half-step twist grid on ``[0, pi)^3`` with reflection weights.

    A ``size``-per-axis half-step grid on the reduced cube is, through the
    corner gauge identity, equivalent to a ``2 * size`` uniform average over
    the full ``[0, 2pi)^3`` twist torus.  ``H(-theta)`` isospectrality pairs
    ``theta`` with ``pi - theta`` componentwise, so only lexicographically
    canonical representatives are returned, weighted by orbit size.
    """
    if size < 1:
        raise ValueError("twist-grid size must be positive")
    points = np.pi * (np.arange(size) + 0.5) / size
    weights: dict[tuple[int, ...], int] = {}
    for index in np.ndindex(*(size,) * 3):
        partner = tuple(size - 1 - component for component in index)
        canonical = min(index, partner)
        weights[canonical] = weights.get(canonical, 0) + 1
    return [
        (points[np.asarray(index)], weight)
        for index, weight in sorted(weights.items())
    ]
