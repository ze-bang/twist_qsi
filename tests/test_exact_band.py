from types import SimpleNamespace

import numpy as np
import pytest

from qsi_campaign.exact_band import (
    basis_character_phases,
    fixed_magnetization_basis,
    full_spin_basis,
    microscopic_character_hamiltonian,
    translation_sector_bases,
    uniform_character_grid,
)


def test_fixed_magnetization_basis():
    basis = fixed_magnetization_basis(6, 3)
    assert len(basis) == 20
    assert all(int(state).bit_count() == 3 for state in basis)


def test_full_spin_basis():
    np.testing.assert_array_equal(full_spin_basis(3), np.arange(8))


def test_character_grid_is_complete():
    grid = uniform_character_grid(2)
    assert len(grid) == 8
    assert {tuple(np.rint(point / np.pi).astype(int)) for point in grid} == {
        (x, y, z) for x in (0, 1) for y in (0, 1) for z in (0, 1)
    }


def test_translation_sectors_are_orthonormal_and_complete():
    states = np.array(
        (0b0011, 0b0101, 0b0110, 0b1001, 0b1010, 0b1100), dtype=np.uint64
    )
    identity = np.arange(4)
    swap_pairs = np.array((1, 0, 3, 2))
    reverse = np.array((3, 2, 1, 0))
    cross = np.array((2, 3, 0, 1))
    sectors = translation_sector_bases(
        states, (identity, swap_pairs, reverse, cross)
    )
    assert sum(sector.shape[1] for sector in sectors.values()) == len(states)
    combined = np.hstack([sector.toarray() for sector in sectors.values()])
    np.testing.assert_allclose(
        combined.conj().T @ combined, np.eye(len(states)), atol=1.0e-14
    )


def two_site_cluster():
    return SimpleNamespace(
        n_sites=2,
        positions=np.asarray(((0.0, 0.0, 0.0), (0.25, 0.25, 0.0))),
        bonds=np.asarray(((0, 1),)),
        bond_wrap=np.asarray(((0, 0, 0),)),
        Lvecs=np.eye(3),
        tet_masks=np.asarray((0b11,), dtype=np.uint64),
    )


def test_pair_flip_source_is_exact_polarization_character():
    cluster = two_site_cluster()
    states = full_spin_basis(2)
    theta = np.asarray((0.37, -0.21, 0.13))
    jpmpm = 0.17
    sourced = microscopic_character_hamiltonian(
        cluster, states, 0.11, theta, jpmpm=jpmpm
    ).toarray()
    zero = microscopic_character_hamiltonian(
        cluster, states, 0.11, np.zeros(3), jpmpm=jpmpm
    ).toarray()
    phases = basis_character_phases(cluster, states, theta)
    np.testing.assert_allclose(
        sourced, phases[:, None] * zero * phases.conjugate()[None, :], atol=1.0e-14
    )
    pair_center = cluster.positions.sum(axis=0)
    np.testing.assert_allclose(
        sourced[0b11, 0b00], jpmpm * np.exp(-2j * theta @ pair_center)
    )


def test_pair_flip_rejects_nonclosed_fixed_magnetization_basis():
    with pytest.raises(ValueError, match="not closed under J_pm_pm"):
        microscopic_character_hamiltonian(
            two_site_cluster(),
            fixed_magnetization_basis(2, 0),
            0.11,
            np.zeros(3),
            jpmpm=0.17,
        )
