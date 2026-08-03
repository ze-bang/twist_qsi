"""Brillouin-Wigner downfolding onto the zero-spinon block.

The polar-pullback protocol needs the ice-descended band to be spectrally
isolated.  The zero-spinon subspace itself is combinatorial and exists at any
coupling, so the ice-sector physics can instead be reached through the exact
Feshbach map

    F(z) = PHP + PHQ (z - QHQ)^{-1} QHP,

whose self-consistent eigenvalues E = eig F(E) are exact eigenvalues of H
with nonzero ice component.  This file is a fresh implementation written
against the verified active machinery; it deliberately imports nothing from
``legacy/``.

The winding-free version applies the exact polarization-block projection to
F(z; theta=0).  At the M=2 gauge corners H(theta) = V H(0) V^dagger with V
diagonal, so the corner average of F reduces to a mask of the theta=0 operator
by the same identity used for the band protocol; the block projection imposed
here is the stricter Delta-p = 0 selection that the continuous source average
enforces exactly.

Diagnostics per self-consistent root: convergence, distance to the nearest
pole of the self-energy (the spinon continuum), and the ice weight
Z = 1 / (1 - <dF/dz>).  Unconverged roots and Z -> 0 signal that the
ice-descended state has dissolved into the continuum -- a physical verdict,
not a numerical failure.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .protocol import polarization_sector_labels, sector_project


@dataclass(frozen=True)
class DownfoldBlocks:
    """Spectral data of the exact Feshbach map for one coupling."""

    php: np.ndarray          # (d, d) ice-block Hamiltonian
    coupling: np.ndarray     # (nq, d): U_Q^T (QHP), the P-Q coupling in the QHQ eigenbasis
    poles: np.ndarray        # (nq,) eigenvalues of QHQ
    sector_labels: np.ndarray

    def self_energy(self, z: float) -> np.ndarray:
        weights = 1.0 / (z - self.poles)
        return (self.coupling * weights[:, None]).T @ self.coupling

    def f_matrix(self, z: float, *, masked: bool) -> np.ndarray:
        f = self.php + self.self_energy(z)
        return sector_project(f, self.sector_labels).real if masked else f

    def self_energy_slope(self, z: float, vector: np.ndarray) -> float:
        """<v| dSigma/dz |v> = -sum_q |<q|QHP|v>|^2 / (z - eps_q)^2  (always <= 0)."""
        amplitude = self.coupling @ vector
        return -float(np.sum(np.abs(amplitude) ** 2 / (z - self.poles) ** 2))

    def pole_distance(self, z: float) -> float:
        return float(np.min(np.abs(z - self.poles)))


def build_blocks(cluster, states: np.ndarray, ice_indices: np.ndarray,
                 hamiltonian) -> DownfoldBlocks:
    """Assemble PHP, QHQ eigendata, and the rotated coupling block.

    ``hamiltonian`` is the sparse microscopic Hamiltonian on ``states`` (the
    fixed-Sz basis for XXZ).  The single dense diagonalization of QHQ is the
    only expensive step and is done exactly, with no iterative solver.
    """
    dense = np.asarray(hamiltonian.todense())
    imaginary = float(np.max(np.abs(dense.imag), initial=0.0))
    if imaginary > 1e-12:
        raise ValueError(f"downfolding expects a real Hamiltonian at theta=0: {imaginary:.3e}")
    dense = dense.real

    n = dense.shape[0]
    ice = np.asarray(ice_indices, dtype=np.int64)
    complement = np.setdiff1d(np.arange(n), ice)

    php = dense[np.ix_(ice, ice)].copy()
    qhp = dense[np.ix_(complement, ice)].copy()
    qhq = dense[np.ix_(complement, complement)]
    poles, rotation = np.linalg.eigh(qhq)

    return DownfoldBlocks(
        php=php,
        coupling=rotation.T @ qhp,
        poles=poles,
        sector_labels=polarization_sector_labels(cluster),
    )


@dataclass(frozen=True)
class RootResult:
    energies: np.ndarray      # (d,) self-consistent roots (nan where unconverged)
    converged: np.ndarray     # (d,) bool
    ice_weight: np.ndarray    # (d,) Z_n in (0, 1]; nan where unconverged
    pole_distance: np.ndarray  # (d,) |E_n - nearest QHQ pole|; nan where unconverged
    iterations: np.ndarray    # (d,) int


def solve_roots(blocks: DownfoldBlocks, *, masked: bool,
                tolerance: float = 1e-12,
                max_bisections: int = 200) -> RootResult:
    """Bisection solve of E = eig_n F(E) for every branch below the continuum.

    Between poles of the self-energy each sorted eigenvalue branch
    ``lambda_n(z)`` of F(z) is monotone decreasing: analytic branches have
    slope ``<v| dSigma/dz |v> <= 0``, sorting preserves monotonicity, and the
    polarization-block mask is a pinching, which preserves the sign of
    dSigma/dz.  Hence ``g_n(z) = lambda_n(z) - z`` is strictly decreasing and
    has exactly one zero below the lowest pole -- located here by guaranteed
    bisection, with no damping and no branch tracking.

    A branch with ``g_n > 0`` at the continuum edge has no root below the
    poles: its ice-descended state has migrated into the spinon continuum.
    It is reported as unconverged -- in the non-isolated regime that count is
    the dissolution diagnostic, not an error.
    """
    d = blocks.php.shape[0]
    edge = float(blocks.poles.min()) - 1e-9

    def branch_values(z: float) -> np.ndarray:
        return np.linalg.eigvalsh(blocks.f_matrix(z, masked=masked))

    gap_at_edge = branch_values(edge) - edge
    # The floor must lie below every branch root.  Since Sigma(z) -> 0 as
    # z -> -infinity, g_n(z) = lambda_n(z) - z grows without bound there, so
    # descending in fixed steps always terminates; a fixed offset is not
    # enough once the continuum sinks with strong transverse coupling.
    floor = min(float(np.linalg.eigvalsh(blocks.php).min()), edge) - 2.0
    for _ in range(10):
        gap_at_floor = branch_values(floor) - floor
        if np.all(gap_at_floor > 0):
            break
        floor -= 4.0
    else:
        raise RuntimeError("bisection floor is not below every branch root")

    energies = np.full(d, np.nan)
    converged = np.zeros(d, dtype=bool)
    weights = np.full(d, np.nan)
    distances = np.full(d, np.nan)
    iterations = np.zeros(d, dtype=np.int64)

    for branch in range(d):
        if gap_at_edge[branch] > 0.0:
            continue  # root lies inside the continuum: dissolved/resonant
        low, high = floor, edge
        for step in range(1, max_bisections + 1):
            middle = 0.5 * (low + high)
            if branch_values(middle)[branch] - middle > 0.0:
                low = middle
            else:
                high = middle
            if high - low < tolerance:
                break
        iterations[branch] = step
        z = 0.5 * (low + high)
        values, vectors = np.linalg.eigh(blocks.f_matrix(z, masked=masked))
        vector = vectors[:, branch]
        energies[branch] = z
        converged[branch] = True
        slope = blocks.self_energy_slope(z, vector)
        weights[branch] = 1.0 / (1.0 - slope)
        distances[branch] = blocks.pole_distance(z)

    return RootResult(
        energies=energies,
        converged=converged,
        ice_weight=weights,
        pole_distance=distances,
        iterations=iterations,
    )
