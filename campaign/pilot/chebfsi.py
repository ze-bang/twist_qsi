"""Chebyshev-filtered subspace iteration for the ice-descended band.

The filter amplifies the spectrum below a cut relative to the rest, so a block
seeded with the ice gather converges onto the band without ever forming a
Krylov basis. Convergence is geometric in the filter degree -- unlike a
polynomial used directly as a step projector, where Gibbs ringing at the cut
gives only ~1/N.

Counts matvecs, because campaign cost is linear in that count and every
estimate of it so far has been mine rather than measured.
"""

from __future__ import annotations

import numpy as np


class MatvecCounter:
    def __init__(self, apply):
        self._apply = apply
        self.count = 0

    def __call__(self, block):
        """block is (dim, nvec); counts one matvec per column."""
        if block.ndim == 1:
            self.count += 1
            return self._apply(block)
        self.count += block.shape[1]
        return np.column_stack([self._apply(np.ascontiguousarray(block[:, j]))
                                for j in range(block.shape[1])])


def spectral_bounds(matvec, dim, steps=20, rng=None):
    """Cheap Lanczos estimate of [lambda_min, lambda_max]."""
    rng = rng or np.random.default_rng(0)
    v = rng.normal(size=dim) + 1j * rng.normal(size=dim)
    v /= np.linalg.norm(v)
    alphas, betas, v_prev, beta = [], [], np.zeros(dim, dtype=np.complex128), 0.0
    for _ in range(steps):
        w = matvec(v)
        alpha = float(np.real(np.vdot(v, w)))
        w = w - alpha * v - beta * v_prev
        alphas.append(alpha)
        beta = float(np.linalg.norm(w))
        if beta < 1e-12:
            break
        betas.append(beta)
        v_prev, v = v, w / beta
    t = np.diag(alphas) + np.diag(betas[:len(alphas) - 1], 1) + np.diag(betas[:len(alphas) - 1], -1)
    ev = np.linalg.eigvalsh(t)
    spread = ev[-1] - ev[0]
    return ev[0] - 0.05 * spread, ev[-1] + 0.05 * spread


def chebyshev_filter(matvec, block, degree, cut, upper):
    """Apply the degree-m Chebyshev polynomial on [cut, upper] to every column.

    Eigenvalues below ``cut`` map outside [-1,1], where T_m grows like
    cosh(m arccosh|x|) -- exponential amplification of exactly the wanted space.
    """
    centre = 0.5 * (upper + cut)
    half = 0.5 * (upper - cut)
    previous = block
    current = (matvec(block) - centre * block) / half
    for _ in range(degree - 1):
        nxt = 2.0 * (matvec(current) - centre * current) / half - previous
        previous, current = current, nxt
    return current


def chebfsi(matvec, seed, n_want, *, degree=25, max_outer=30, tol=1e-10,
            cut=None, verbose=False):
    """Return (values, vectors, diagnostics) for the ``n_want`` lowest states."""
    dim = seed.shape[0]
    lo, hi = spectral_bounds(matvec, dim)
    block, _ = np.linalg.qr(seed)
    if cut is None:
        # Start the cut just above the seeded subspace's own Ritz spread.
        small = block.conj().T @ matvec(block)
        cut = float(np.linalg.eigvalsh(0.5 * (small + small.conj().T)).max())
    history = []
    for outer in range(1, max_outer + 1):
        block = chebyshev_filter(matvec, block, degree, cut, hi)
        block, _ = np.linalg.qr(block)
        projected = block.conj().T @ matvec(block)
        projected = 0.5 * (projected + projected.conj().T)
        values, small_vectors = np.linalg.eigh(projected)
        block = block @ small_vectors
        residual = matvec(block[:, :n_want]) - block[:, :n_want] * values[:n_want]
        worst = float(np.linalg.norm(residual, axis=0).max())
        history.append({"outer": outer, "max_residual": worst, "matvecs": matvec.count})
        if verbose:
            print(f"    outer {outer:2d}: max residual {worst:.3e}  "
                  f"matvecs {matvec.count}", flush=True)
        if worst < tol:
            break
        cut = float(values[n_want - 1] + 0.5 * (values[min(n_want, len(values) - 1)] - values[n_want - 1]))
    return values[:n_want], block[:, :n_want], {
        "outer_iterations": outer, "matvecs": matvec.count,
        "max_residual": worst, "history": history,
        "bounds": (lo, hi),
    }
