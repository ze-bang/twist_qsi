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
from scipy.linalg import eigh as scipy_eigh

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
    fixed-Sz basis for XXZ, or the full spin basis when ``Jpmpm`` breaks Sz).
    The single dense diagonalization of QHQ is the only expensive step and is
    done exactly, with no iterative solver.  The blocks are sliced from the
    sparse matrix so that the only large dense allocations are QHQ itself and
    the LAPACK workspace: on the full 65,536-state cubic-16 basis the dense
    intermediate of ``todense()`` alone would cost 34 GB before slicing.
    """
    matrix = hamiltonian.tocsr()
    if np.iscomplexobj(matrix.data):
        imaginary = float(np.max(np.abs(matrix.data.imag), initial=0.0))
        if imaginary > 1e-12:
            raise ValueError(
                f"downfolding expects a real Hamiltonian at theta=0: {imaginary:.3e}")
        matrix = matrix.real.tocsr()

    n = matrix.shape[0]
    ice = np.asarray(ice_indices, dtype=np.int64)
    complement = np.setdiff1d(np.arange(n), ice)

    php = matrix[ice][:, ice].toarray()
    qhp = matrix[complement][:, ice].toarray()
    qhq = matrix[complement][:, complement].toarray()
    # overwrite_a reuses the QHQ storage for the eigenvectors; peak memory is
    # QHQ plus the dsyevd workspace (~3 n^2 doubles total) instead of ~4 n^2.
    poles, rotation = scipy_eigh(qhq, overwrite_a=True, check_finite=False,
                                 driver="evd")
    del qhq

    return DownfoldBlocks(
        php=php,
        coupling=rotation.T @ qhp,
        poles=poles,
        sector_labels=polarization_sector_labels(cluster),
    )


def _dense_eigh(matrix: np.ndarray, gpu: bool):
    """Dense symmetric eigendecomposition, on the H100 when ``gpu``.

    A 16,384 Klein block is 2.1 GB in fp64 -- trivial for an 80 GB H100 --
    and cuSOLVER syevd runs it ~10-30x faster than the ~91 GFLOPS the AOCL
    dsyevd sustains on a full Genoa node (the measured backend ceiling
    there).  Transfers are ~2 GB each way, seconds, so only the
    eigendecomposition itself moves to the device.
    """
    if gpu:
        import cupy
        device_values, device_vectors = cupy.linalg.eigh(cupy.asarray(matrix))
        return cupy.asnumpy(device_values), cupy.asnumpy(device_vectors)
    return scipy_eigh(matrix, overwrite_a=True, check_finite=False,
                      driver="evd")


def build_blocks_klein(cluster, states: np.ndarray, ice_indices: np.ndarray,
                       hamiltonian, sector_bases, *,
                       gpu: bool = False) -> DownfoldBlocks:
    """``build_blocks`` through the Klein translation decomposition.

    On the full 65,536-state cubic-16 basis the direct QHQ eigendecomposition
    is a single 65,446-dimensional dense solve (~9 h at measured node
    throughput).  The four FCC translations of the cubic cell commute with
    ``H(theta=0)`` at any ``(Jpm, Jpmpm)`` and the ice manifold is closed under
    them, so QHQ block-diagonalizes over the Klein irreps: four ~16,384 solves
    (~16x fewer flops, ~10 GB instead of ~100 GB).

    Per irrep ``k`` with sparse basis ``B_k`` (columns orthonormal):
    the ice components are ``I_k = B_k[ice]^dagger`` (m_k x 90, rank d_k with
    sum_k d_k = 90), orthonormalized by SVD into ``O_k``; the sector QHQ is
    ``(1 - O_k O_k^dagger) H_k (1 - O_k O_k^dagger)`` whose spectrum splits
    exactly into d_k spurious ice-kernel modes (classified by their overlap
    with ``O_k`` -- NOT by eigenvalue, since genuine poles cross zero at
    strong coupling) and the true QHQ poles; the coupling rows follow as
    ``W^dagger H_k I_k`` because the kept eigenvectors are orthogonal to the
    ice subspace.  Poles and coupling concatenate over irreps into the same
    ``DownfoldBlocks`` contract as the direct path.
    """
    matrix = hamiltonian.tocsr()
    if np.iscomplexobj(matrix.data):
        imaginary = float(np.max(np.abs(matrix.data.imag), initial=0.0))
        if imaginary > 1e-12:
            raise ValueError(
                f"downfolding expects a real Hamiltonian at theta=0: {imaginary:.3e}")
        matrix = matrix.real.tocsr()

    ice = np.asarray(ice_indices, dtype=np.int64)
    php = matrix[ice][:, ice].toarray()

    poles_parts, coupling_parts, rank_total = [], [], 0
    for name, basis in sector_bases.items():
        block = (basis.conj().T @ matrix @ basis).toarray()
        imaginary = float(np.abs(block.imag).max())
        if imaginary > 1e-10:
            raise ValueError(f"Klein block {name} is not real: {imaginary:.3e}")
        block = np.ascontiguousarray(block.real)

        # ice components of this irrep: (m_k, 90); real because the Klein
        # characters are +-1
        ice_components = np.asarray(basis[ice].conj().T.todense()).real
        left, singular, _ = np.linalg.svd(ice_components, full_matrices=False)
        rank = int(np.sum(singular > 1e-10))
        rank_total += rank
        ortho = left[:, :rank]                              # (m_k, d_k)

        projected = block - ortho @ (ortho.T @ block)
        projected -= (projected @ ortho) @ ortho.T          # (1-P) H (1-P)
        projected = 0.5 * (projected + projected.T)
        values, vectors = _dense_eigh(projected, gpu)

        overlap = np.linalg.norm(ortho.T @ vectors, axis=0)
        spurious = overlap > 0.5
        if int(spurious.sum()) != rank or (
                np.any((overlap > 1e-6) & (overlap < 1.0 - 1e-6))):
            raise RuntimeError(
                f"Klein block {name}: ice-kernel classification failed "
                f"({int(spurious.sum())} spurious vs rank {rank}; "
                f"overlap range [{overlap.min():.2e}, {overlap.max():.2e}])")
        keep = ~spurious
        poles_parts.append(values[keep])
        coupling_parts.append(vectors[:, keep].T @ (block @ ice_components))

    if rank_total != len(ice):
        raise RuntimeError(
            f"ice ranks over Klein irreps sum to {rank_total}, expected {len(ice)}")

    return DownfoldBlocks(
        php=php,
        coupling=np.vstack(coupling_parts),
        poles=np.concatenate(poles_parts),
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
                max_bisections: int = 200,
                coupling_floor: float | None = None) -> RootResult:
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
    if coupling_floor is None:
        # Historical behaviour, kept as the default so existing consumers
        # are not changed underneath them.
        edge = float(blocks.poles.min()) - 1e-9
    else:
        # A pole with no coupling to P is not a pole of the self-energy at
        # all: its residue vanishes, Sigma is analytic there, and an
        # ice-derived level cannot decay into it.  Taking such a state as
        # the continuum edge rejects perfectly good roots against a
        # continuum the band can never reach -- measured on the full
        # 65,536-state basis, where the lowest pole is an odd-parity state
        # with exactly zero coupling, this loses all 90 roots at points
        # where the even-parity edge keeps 25.  Monotonicity of the
        # branches is untouched, since Sigma is analytic across the
        # decoupled poles being skipped.
        weight = np.linalg.norm(blocks.coupling, axis=1)
        coupled = blocks.poles[weight > coupling_floor]
        if coupled.size == 0:
            raise RuntimeError(
                "no pole couples to the model space above "
                f"{coupling_floor:g}: the fold is empty")
        edge = float(coupled.min()) - 1e-9

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
