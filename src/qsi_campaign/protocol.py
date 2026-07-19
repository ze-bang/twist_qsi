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


def centered_relative_error(left: np.ndarray, right: np.ndarray) -> float:
    """Frobenius error after removing physically irrelevant scalar shifts."""
    left = _hermitian(left)
    right = _hermitian(right)
    n = left.shape[0]
    left -= np.eye(n) * np.trace(left) / n
    right -= np.eye(n) * np.trace(right) / n
    return float(np.linalg.norm(left - right) / max(np.linalg.norm(right), 1.0e-15))


def replace_low_band(
    full_spectrum: np.ndarray,
    clean_band: np.ndarray,
    *,
    match_trace: bool = True,
) -> np.ndarray:
    """Replace the lowest microscopic band while retaining its complement.

    Character projection preserves the model-space trace.  `match_trace`
    enforces that identity when the clean band comes from a truncated series,
    which also fixes its energy relative to the untouched complement.
    """
    full = np.sort(np.asarray(full_spectrum, dtype=float))
    clean = np.sort(np.asarray(clean_band, dtype=float))
    if len(clean) >= len(full):
        raise ValueError("the replacement band must be smaller than the full spectrum")
    if match_trace:
        clean = clean + full[: len(clean)].mean() - clean.mean()
    return np.sort(np.concatenate((clean, full[len(clean) :])))
