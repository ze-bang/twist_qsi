import numpy as np

from qsi_campaign.protocol import (
    band_operator_from_eigenvectors,
    band_operator_from_projected_vectors,
    character_project,
    full_hilbert_counterterm_spectrum,
)


def test_character_project_is_hermitian_average():
    matrices = np.array(
        [
            [[1, 1j], [-1j, 2]],
            [[3, -1j], [1j, 4]],
        ],
        dtype=complex,
    )
    projected = character_project(matrices)
    np.testing.assert_allclose(projected, np.diag([2.0, 3.0]))


def test_band_pullback_recovers_spectrum():
    energies = np.array([-2.0, 0.5])
    psi = np.eye(3, 2, dtype=complex)
    operator, diagnostics = band_operator_from_eigenvectors(energies, psi, np.array([0, 1]))
    np.testing.assert_allclose(np.linalg.eigvalsh(operator), energies)
    assert diagnostics["model_overlap_min"] == 1.0


def test_projected_band_pullback_is_invariant_to_column_phases():
    energies = np.array((-2.0, -1.0))
    projected = np.diag((0.9, 0.8)).astype(complex)
    left, _ = band_operator_from_projected_vectors(energies, projected)
    phases = np.diag(np.exp(1j * np.array((0.37, -0.91))))
    right, diagnostics = band_operator_from_projected_vectors(
        energies, projected @ phases
    )
    np.testing.assert_allclose(left, right)
    assert diagnostics["pullback_unitarity_error"] < 1.0e-12


def test_full_hilbert_counterterm_preserves_complement_and_trace():
    full = np.array([-4.0, -3.0, 1.0, 2.0])
    corrected = full_hilbert_counterterm_spectrum(full, np.array([-1.0, 1.0]))
    np.testing.assert_allclose(corrected, np.array([-4.5, -2.5, 1.0, 2.0]))
