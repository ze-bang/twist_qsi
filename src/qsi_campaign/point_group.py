"""Space-group reduction of the character grid.

The sourced Hamiltonian is covariant under any space-group operation of the
cluster: if ``R`` is realised as a site permutation ``sigma`` that carries every
lifted bond displacement ``d_ij`` to ``R d_ij``, then

```text
H(R theta) = P_sigma H(theta) P_sigma^dagger,
```

so the whole band construction follows, ``h(R theta) = U_R h(theta) U_R^T``,
with ``U_R`` the induced permutation of the ice reference basis.  Together with
``h(-theta) = h(theta).conj()`` this lets one solved source point supply every
other point in its orbit at the cost of a permutation.

The reduction requires ``Jpmpm = 0``.  The pair-flip phase is set by
``c_ij = r_i + r_j - n_ij^T L``, which involves absolute positions and
transforms as ``c -> R c + 2 t``; it is therefore *not* covariant for the
operations that carry a non-primitive translation ``t``.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations, product
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class SpaceGroupOperation:
    """A cubic rotation together with the site permutation that realises it."""

    rotation: np.ndarray
    sites: np.ndarray


def signed_permutation_group() -> list[np.ndarray]:
    """Return the 48 signed coordinate permutations of the cubic point group."""
    matrices = []
    for order in permutations(range(3)):
        for signs in product((1, -1), repeat=3):
            matrix = np.zeros((3, 3))
            for row in range(3):
                matrix[row, order[row]] = signs[row]
            matrices.append(matrix)
    return matrices


def _position_key(cluster, position: np.ndarray, resolution: int) -> tuple[int, ...]:
    fractional = np.mod(position @ np.linalg.inv(cluster.Lvecs), 1.0)
    return tuple(np.rint(fractional * resolution).astype(int) % resolution)


def _lifted_displacements(cluster) -> dict[tuple[int, int], np.ndarray]:
    displacements: dict[tuple[int, int], np.ndarray] = {}
    for (left, right), image in zip(cluster.bonds, cluster.bond_wrap):
        vector = (
            cluster.positions[right]
            - cluster.positions[left]
            - np.asarray(image, dtype=float) @ cluster.Lvecs
        )
        displacements[(int(left), int(right))] = vector
        displacements[(int(right), int(left))] = -vector
    return displacements


def realised_space_group(
    cluster, *, resolution: int = 8, tolerance: float = 1.0e-9
) -> list[SpaceGroupOperation]:
    """Space-group operations realised on ``cluster`` as site permutations.

    An operation is accepted only when it maps the cluster onto itself *and*
    carries every lifted bond displacement covariantly.  The second condition is
    what makes the character-grid reduction exact rather than approximate.
    """
    positions = cluster.positions
    n_sites = cluster.n_sites
    site_of = {
        _position_key(cluster, position, resolution): site
        for site, position in enumerate(positions)
    }
    if len(site_of) != n_sites:
        raise RuntimeError("position keys collide; raise the resolution")
    displacements = _lifted_displacements(cluster)

    accepted: list[SpaceGroupOperation] = []
    for rotation in signed_permutation_group():
        for anchor in range(n_sites):
            shift = positions[anchor] - rotation @ positions[0]
            permutation: list[int] = []
            for site in range(n_sites):
                key = _position_key(cluster, rotation @ positions[site] + shift, resolution)
                if key not in site_of:
                    break
                permutation.append(site_of[key])
            if len(permutation) != n_sites or len(set(permutation)) != n_sites:
                continue
            if all(
                (permutation[left], permutation[right]) in displacements
                and np.allclose(
                    displacements[(permutation[left], permutation[right])],
                    rotation @ vector,
                    atol=tolerance,
                )
                for (left, right), vector in displacements.items()
            ):
                accepted.append(
                    SpaceGroupOperation(
                        rotation=rotation,
                        sites=np.asarray(permutation, dtype=np.int64),
                    )
                )
                break
    return accepted


def ice_permutations(
    cluster, operations: Iterable[SpaceGroupOperation]
) -> list[np.ndarray]:
    """Induced permutations of the ice reference basis, as index arrays.

    Entry ``p[k]`` is the ice index that basis state ``k`` is carried to.
    """
    ice_states = [int(state) for state in cluster.ice_states]
    position_of = {state: index for index, state in enumerate(ice_states)}
    induced = []
    for operation in operations:
        image_index = np.empty(len(ice_states), dtype=np.int64)
        for index, state in enumerate(ice_states):
            image = 0
            for site in range(cluster.n_sites):
                if state >> site & 1:
                    image |= 1 << int(operation.sites[site])
            image_index[index] = position_of[image]
        induced.append(image_index)
    return induced


def conjugate_ice_operator(operator: np.ndarray, ice_permutation: np.ndarray) -> np.ndarray:
    """Return ``U h U^T`` for the permutation encoded by ``ice_permutation``."""
    inverse = np.empty_like(ice_permutation)
    inverse[ice_permutation] = np.arange(len(ice_permutation))
    return operator[np.ix_(inverse, inverse)]


def is_gauge_corner(index: Sequence[int], grid_size: int) -> bool:
    """True when the source at ``index`` is a pure gauge (self-conjugate) point."""
    return all((2 * value) % grid_size == 0 for value in index)


def rotate_index(
    index: Sequence[int], rotation: np.ndarray, grid_size: int
) -> tuple[int, ...]:
    """Apply a rotation to a character-grid index, modulo the grid size."""
    vector = np.rint(rotation @ np.asarray(index, dtype=float)).astype(int)
    return tuple(int(value) % grid_size for value in vector)


def character_orbits(
    grid_size: int, operations: Sequence[SpaceGroupOperation]
) -> tuple[list[tuple[int, ...]], dict[tuple[int, ...], tuple[tuple[int, ...], int, bool]]]:
    """Reduce the ``Z_M^3`` character grid under the space group and conjugation.

    Returns the orbit representatives and, for every grid index, a recipe
    ``(representative, operation, conjugate)``: apply ``operations[operation]``
    to the representative's operator, then conjugate if requested.
    """
    recipes: dict[tuple[int, ...], tuple[tuple[int, ...], int, bool]] = {}
    representatives: list[tuple[int, ...]] = []
    for index in product(range(grid_size), repeat=3):
        if index in recipes:
            continue
        representatives.append(index)
        for operation, element in enumerate(operations):
            rotated = rotate_index(index, element.rotation, grid_size)
            for conjugate in (False, True):
                image = (
                    tuple((-value) % grid_size for value in rotated)
                    if conjugate
                    else rotated
                )
                recipes.setdefault(image, (index, operation, conjugate))
    if len(recipes) != grid_size**3:
        raise RuntimeError("character orbits do not cover the grid")
    for index, (representative, _, _) in recipes.items():
        if is_gauge_corner(index, grid_size) != is_gauge_corner(representative, grid_size):
            raise RuntimeError(
                "an orbit mixes gauge corners with sourced points; the reduction "
                "assumes the two never share an orbit"
            )
    return representatives, recipes


def reconstruct_operator(
    operator: np.ndarray,
    recipe: tuple[tuple[int, ...], int, bool],
    ice_permutation_by_operation: Sequence[np.ndarray],
) -> np.ndarray:
    """Rebuild one grid point's operator from its orbit representative."""
    _, operation, conjugate = recipe
    image = conjugate_ice_operator(operator, ice_permutation_by_operation[operation])
    return image.conjugate() if conjugate else image
