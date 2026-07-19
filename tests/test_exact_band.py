import numpy as np

from qsi_campaign.exact_band import (
    fixed_magnetization_basis,
    translation_sector_bases,
    uniform_character_grid,
)


def test_fixed_magnetization_basis():
    basis = fixed_magnetization_basis(6, 3)
    assert len(basis) == 20
    assert all(int(state).bit_count() == 3 for state in basis)


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
