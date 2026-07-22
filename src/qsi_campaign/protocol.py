"""Linear-algebra primitives for the projected-band construction."""

from __future__ import annotations

import numpy as np


def _hermitian(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.complex128)
    return 0.5 * (matrix + matrix.conj().T)


def inverse_sqrt_hermitian(matrix: np.ndarray, cutoff: float = 1.0e-12) -> np.ndarray:
    """Return the positive inverse square root of a Hermitian matrix."""
    values, vectors = np.linalg.eigh(_hermitian(matrix))
    if values.min(initial=np.inf) <= cutoff:
        raise ValueError(f"matrix is singular below cutoff: lambda_min={values.min():.3e}")
    return (vectors / np.sqrt(values)) @ vectors.conj().T


def band_operator_from_eigenvectors(
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    model_indices: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    """Pull an isolated exact band back to a fixed model-space basis.

    The polar (Loewdin) map fixes the otherwise arbitrary gauge within the
    band.  It is the exact Kato/canonical pullback whenever the projected Gram
    matrix is nonsingular.
    """
    energies = np.asarray(eigenvalues, dtype=float)
    psi = np.asarray(eigenvectors, dtype=np.complex128)
    if psi.shape[1] != len(energies):
        raise ValueError("one eigenvector column is required per eigenvalue")

    full_gram = _hermitian(psi.conj().T @ psi)
    full_inverse = inverse_sqrt_hermitian(full_gram)
    orthonormal = psi @ full_inverse
    projected = orthonormal[np.asarray(model_indices, dtype=int), :]
    gram = _hermitian(projected.conj().T @ projected)
    gram_values = np.linalg.eigvalsh(gram)
    if gram_values.max(initial=-np.inf) > 1.0 + 1.0e-8:
        raise ValueError(
            "projected Gram matrix exceeds the identity; input vectors are not "
            "an orthonormal full-space band"
        )
    q = projected @ inverse_sqrt_hermitian(gram)
    operator = _hermitian(q @ np.diag(energies) @ q.conj().T)
    diagnostics = {
        "model_overlap_min": float(gram_values.min()),
        "model_overlap_mean": float(gram_values.mean()),
        "model_overlap_max": float(gram_values.max()),
        "pullback_unitarity_error": float(
            np.linalg.norm(q.conj().T @ q - np.eye(q.shape[1]))
        ),
    }
    return operator, diagnostics


def band_operator_from_projected_vectors(
    eigenvalues: np.ndarray,
    projected_eigenvectors: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    """Pull an orthonormal exact band back from its model-space projection."""
    energies = np.asarray(eigenvalues, dtype=float)
    projected = np.asarray(projected_eigenvectors, dtype=np.complex128)
    if projected.ndim != 2 or projected.shape[1] != len(energies):
        raise ValueError("projected vectors must have one column per eigenvalue")
    gram = _hermitian(projected.conj().T @ projected)
    gram_values = np.linalg.eigvalsh(gram)
    q = projected @ inverse_sqrt_hermitian(gram)
    operator = _hermitian(q @ np.diag(energies) @ q.conj().T)
    diagnostics = {
        "model_overlap_min": float(gram_values.min()),
        "model_overlap_mean": float(gram_values.mean()),
        "model_overlap_max": float(gram_values.max()),
        "pullback_unitarity_error": float(
            np.linalg.norm(q.conj().T @ q - np.eye(q.shape[1]))
        ),
    }
    return operator, diagnostics


def character_project(operators: np.ndarray) -> np.ndarray:
    """Average Hermitian band operators over a complete character grid."""
    matrices = np.asarray(operators, dtype=np.complex128)
    if matrices.ndim != 3 or matrices.shape[1] != matrices.shape[2]:
        raise ValueError("operators must have shape (n_character, n, n)")
    return _hermitian(matrices.mean(axis=0))


LOCAL_AXES = np.array(
    [[1.0, 1.0, 1.0], [1.0, -1.0, -1.0], [-1.0, 1.0, -1.0], [-1.0, -1.0, 1.0]]
) / np.sqrt(3.0)


def ice_magnetization(cluster) -> np.ndarray:
    """Total moment sum_i sigma_i zhat_i of each ice configuration.

    On the ice manifold this is exactly ``-16/sqrt(3)`` times the polarization
    coordinate of Eq. (5), so it is constant on a transport sector: a
    longitudinal field is a c-number on each block of the winding-free
    Hamiltonian and cannot mix sectors.
    """
    states = np.asarray(cluster.ice_states, dtype=np.uint64)
    n_sites = int(cluster.n_sites)
    bits = (states[:, None] >> np.arange(n_sites, dtype=np.uint64)) & np.uint64(1)
    sigma = 2 * bits.astype(int) - 1
    return sigma @ LOCAL_AXES[np.arange(n_sites) % 4]


def zeeman_band_term(cluster, field) -> np.ndarray:
    """Diagonal Zeeman operator on the ice reference basis, in units of g_z mu_B.

    ``field`` is the laboratory field vector; the coupling is
    ``-sum_i (B . zhat_i) S_i^z``, which is diagonal in the S^z basis and so
    carries no character source phase.
    """
    field = np.asarray(field, dtype=float)
    if field.shape != (3,):
        raise ValueError("field must have three components")
    return -0.5 * (ice_magnetization(cluster) @ field)


def polarization_sector_labels(cluster) -> np.ndarray:
    """Label each ice basis state by its polarization coordinate.

    The polarization ``p(C) = sum_i S_i^z r_i`` is the bookkeeping coordinate
    whose change along a microscopic sequence is the transported dipole.  A
    move therefore connects two ice configurations only if it transports
    ``q = 2 dp``, so the exact zero-character Hamiltonian is block diagonal in
    ``p``.  Labels index distinct ``p`` values in sorted order and follow the
    ordering of ``cluster.ice_states``.
    """
    states = np.asarray(cluster.ice_states)
    n_sites = int(cluster.n_sites)
    bits = np.array(
        [[(int(state) >> site) & 1 for site in range(n_sites)] for state in states]
    )
    polarization = (bits - 0.5) @ np.asarray(cluster.positions, dtype=float)
    keys = [tuple(np.round(row, 6)) for row in polarization]
    order = {key: index for index, key in enumerate(sorted(set(keys)))}
    return np.array([order[key] for key in keys], dtype=np.int64)


def sector_project(operator: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Discard matrix elements between distinct polarization sectors.

    A finite ``M^3`` grid only annihilates transport with ``q != 0 (mod M)``,
    so it leaves a residual coupling between sectors that a continuous source
    average removes exactly.  Imposing the block structure directly is exact at
    any ``M`` and restores the sector degeneracies the grid would otherwise
    split.
    """
    matrix = _hermitian(np.asarray(operator, dtype=np.complex128))
    labels = np.asarray(labels, dtype=np.int64)
    if matrix.shape[0] != len(labels):
        raise ValueError("operator dimension must match the number of labels")
    return _hermitian(np.where(labels[:, None] == labels[None, :], matrix, 0.0))


def sector_leakage(operator: np.ndarray, labels: np.ndarray) -> float:
    """Largest matrix element the sector projection removes."""
    matrix = _hermitian(np.asarray(operator, dtype=np.complex128))
    labels = np.asarray(labels, dtype=np.int64)
    off = np.abs(matrix[labels[:, None] != labels[None, :]])
    return float(off.max()) if off.size else 0.0


def centered_relative_error(left: np.ndarray, right: np.ndarray) -> float:
    """Frobenius error after removing physically irrelevant scalar shifts."""
    left = _hermitian(left)
    right = _hermitian(right)
    n = left.shape[0]
    left -= np.eye(n) * np.trace(left) / n
    right -= np.eye(n) * np.trace(right) / n
    return float(np.linalg.norm(left - right) / max(np.linalg.norm(right), 1.0e-15))


def full_hilbert_counterterm_spectrum(
    full_spectrum: np.ndarray,
    winding_free_band: np.ndarray,
    *,
    match_trace: bool = True,
) -> np.ndarray:
    """Return the spectrum of the full-Hilbert counterterm Hamiltonian.

    The band-supported correction changes the exact low-energy block and is
    zero on its microscopic complement.  ``match_trace`` fixes the scalar
    placement of the corrected block relative to that complement.
    """
    full = np.sort(np.asarray(full_spectrum, dtype=float))
    clean = np.sort(np.asarray(winding_free_band, dtype=float))
    if len(clean) >= len(full):
        raise ValueError("the corrected band must be smaller than the full spectrum")
    if match_trace:
        clean = clean + full[: len(clean)].mean() - clean.mean()
    return np.sort(np.concatenate((clean, full[len(clean) :])))


def replace_low_band(
    full_spectrum: np.ndarray,
    clean_band: np.ndarray,
    *,
    match_trace: bool = True,
) -> np.ndarray:
    """Compatibility alias for :func:`full_hilbert_counterterm_spectrum`."""
    return full_hilbert_counterterm_spectrum(
        full_spectrum, clean_band, match_trace=match_trace
    )
