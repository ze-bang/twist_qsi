"""Tests for the space-group reduction of the character grid."""

from __future__ import annotations

import sys
from itertools import combinations, product
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "notes"))

import recompute_finite_size_artifact as geometry  # noqa: E402
from qsi_campaign.exact_band import microscopic_character_hamiltonian  # noqa: E402
from qsi_campaign.point_group import (  # noqa: E402
    character_orbits,
    conjugate_ice_operator,
    ice_permutations,
    is_gauge_corner,
    realised_space_group,
    reconstruct_operator,
    rotate_index,
)


@pytest.fixture(scope="module")
def cubic():
    return geometry.build_cluster("cubic", (1, 1, 1))


@pytest.fixture(scope="module")
def fcc():
    return geometry.build_cluster("fcc", (2, 2, 2))


def test_full_cubic_group_is_realised(cubic, fcc):
    """Both production clusters carry the whole 48-element point group."""
    assert len(realised_space_group(cubic)) == 48
    assert len(realised_space_group(fcc)) == 48


def test_site_permutations_are_bijections(cubic):
    for operation in realised_space_group(cubic):
        assert sorted(operation.sites.tolist()) == list(range(cubic.n_sites))


def test_hamiltonian_is_covariant(cubic):
    """H(R theta) = P_sigma H(theta) P_sigma^dagger, which is what licenses the reduction."""
    n_up = 2
    states = np.array(
        sorted(sum(1 << i for i in c) for c in combinations(range(cubic.n_sites), n_up)),
        dtype=np.uint64,
    )
    index_of = {int(state): position for position, state in enumerate(states)}
    theta = np.array([0.11, -0.23, 0.37])
    hamiltonian = microscopic_character_hamiltonian(cubic, states, 0.046, theta).toarray()

    for operation in realised_space_group(cubic):
        permutation = np.empty(len(states), dtype=int)
        for position, state in enumerate(states):
            image = 0
            for site in range(cubic.n_sites):
                if int(state) >> site & 1:
                    image |= 1 << int(operation.sites[site])
            permutation[position] = index_of[image]
        rotated = microscopic_character_hamiltonian(
            cubic, states, 0.046, operation.rotation @ theta
        ).toarray()
        moved = hamiltonian[np.ix_(np.argsort(permutation), np.argsort(permutation))]
        assert np.abs(moved - rotated).max() < 1.0e-12


def test_ice_conjugation_matches_dense_product(cubic):
    permutations = ice_permutations(cubic, realised_space_group(cubic))
    rank = len(cubic.ice_states)
    rng = np.random.default_rng(0)
    operator = rng.normal(size=(rank, rank)) + 1j * rng.normal(size=(rank, rank))
    operator = 0.5 * (operator + operator.conj().T)
    for permutation in permutations[:6]:
        dense = np.zeros((rank, rank))
        dense[permutation, np.arange(rank)] = 1.0
        assert np.abs(
            conjugate_ice_operator(operator, permutation) - dense @ operator @ dense.T
        ).max() == 0.0


@pytest.mark.parametrize("grid_size", [2, 3, 4])
def test_orbits_cover_the_grid_and_separate_corners(cubic, grid_size):
    group = realised_space_group(cubic)
    representatives, recipes = character_orbits(grid_size, group)
    grid = list(product(range(grid_size), repeat=3))
    assert set(recipes) == set(grid)
    for index in grid:
        representative, operation, conjugate = recipes[index]
        assert representative in representatives
        rotated = rotate_index(representative, group[operation].rotation, grid_size)
        if conjugate:
            rotated = tuple((-value) % grid_size for value in rotated)
        assert rotated == index
        # a pure-gauge point can never be rebuilt from a sourced one, or vice versa
        assert is_gauge_corner(index, grid_size) == is_gauge_corner(
            representative, grid_size
        )


@pytest.mark.parametrize("grid_size", [3, 4])
def test_reduction_is_smaller_than_conjugation_alone(cubic, grid_size):
    group = realised_space_group(cubic)
    representatives, _ = character_orbits(grid_size, group)
    solved = [i for i in representatives if not is_gauge_corner(i, grid_size)]
    conjugation_only = sum(
        1 for i in product(range(grid_size), repeat=3) if not is_gauge_corner(i, grid_size)
    ) // 2
    assert len(solved) < conjugation_only


def test_reconstruction_round_trips_through_the_identity(cubic):
    group = realised_space_group(cubic)
    permutations = ice_permutations(cubic, group)
    rank = len(cubic.ice_states)
    rng = np.random.default_rng(1)
    operator = rng.normal(size=(rank, rank)) + 1j * rng.normal(size=(rank, rank))
    operator = 0.5 * (operator + operator.conj().T)
    identity = next(
        i for i, o in enumerate(group) if np.array_equal(o.sites, np.arange(cubic.n_sites))
    )
    rebuilt = reconstruct_operator(operator, ((0, 0, 0), identity, False), permutations)
    assert np.abs(rebuilt - operator).max() == 0.0
    flipped = reconstruct_operator(operator, ((0, 0, 0), identity, True), permutations)
    assert np.abs(flipped - operator.conjugate()).max() == 0.0
