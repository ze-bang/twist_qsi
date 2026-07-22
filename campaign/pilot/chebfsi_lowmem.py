"""Memory-bounded Chebyshev-filtered subspace iteration.

At FCC-32 one block is ~313 GB in fp32, against ~510 GB of host RAM, so every
step has to work in place with only small temporaries:

  * the Chebyshev recurrence holds three blocks (previous/current/next), which
    would be ~939 GB -- but it is independent per column, so it runs on column
    chunks and writes back into the block;
  * ``np.linalg.qr`` returns new arrays (~626 GB), so orthonormalisation is
    Cholesky-QR2: a k x k Gram, a Cholesky, and an in-place triangular solve,
    repeated twice for stability;
  * the Rayleigh-Ritz projection and the subsequent k x k rotation are both
    accumulated chunkwise.

Peak is one block plus O(chunk) columns.
"""

from __future__ import annotations

import numpy as np


class MatvecCounter:
    def __init__(self, apply):
        self._apply = apply
        self.count = 0

    def one(self, v):
        self.count += 1
        return self._apply(v)


def _gram(block, chunk):
    """X^H X accumulated over row chunks."""
    k = block.shape[1]
    out = np.zeros((k, k), dtype=np.complex128)
    for start in range(0, block.shape[0], chunk):
        piece = block[start:start + chunk]
        out += piece.conj().T.astype(np.complex128) @ piece.astype(np.complex128)
    return 0.5 * (out + out.conj().T)


def _apply_right(block, matrix, chunk):
    """block <- block @ matrix, in place over row chunks."""
    for start in range(0, block.shape[0], chunk):
        piece = block[start:start + chunk]
        block[start:start + chunk] = (piece.astype(np.complex128) @ matrix).astype(block.dtype)


def cholesky_qr2(block, chunk, jitter=1e-14):
    """Orthonormalise the columns in place. Two passes: one is unstable."""
    for _ in range(2):
        gram = _gram(block, chunk)
        scale = float(np.real(np.trace(gram))) / gram.shape[0]
        factor = np.linalg.cholesky(gram + jitter * scale * np.eye(gram.shape[0]))
        _apply_right(block, np.linalg.inv(factor.conj().T), chunk)
    return block


def filter_in_place(matvec, block, degree, cut, upper, columns):
    """Chebyshev filter, column-chunked so only 3 chunks are ever resident."""
    centre = 0.5 * (upper + cut)
    half = 0.5 * (upper - cut)
    dtype = block.dtype
    for start in range(0, block.shape[1], columns):
        stop = min(start + columns, block.shape[1])
        previous = block[:, start:stop].astype(np.complex128)
        current = np.empty_like(previous)
        for j in range(stop - start):
            current[:, j] = (matvec.one(previous[:, j]) - centre * previous[:, j]) / half
        for _ in range(degree - 1):
            nxt = np.empty_like(previous)
            for j in range(stop - start):
                nxt[:, j] = 2.0 * (matvec.one(current[:, j]) - centre * current[:, j]) / half \
                            - previous[:, j]
            previous, current = current, nxt
        block[:, start:stop] = current.astype(dtype)
    return block


def rayleigh_ritz(matvec, block, chunk, columns):
    """Return (values, rotation) for X^H H X without materialising H X."""
    k = block.shape[1]
    projected = np.zeros((k, k), dtype=np.complex128)
    for start in range(0, k, columns):
        stop = min(start + columns, k)
        applied = np.empty((block.shape[0], stop - start), dtype=np.complex128)
        for j in range(start, stop):
            applied[:, j - start] = matvec.one(block[:, j].astype(np.complex128))
        for row in range(0, block.shape[0], chunk):
            piece = block[row:row + chunk].astype(np.complex128)
            projected[:, start:stop] += piece.conj().T @ applied[row:row + chunk]
    projected = 0.5 * (projected + projected.conj().T)
    return np.linalg.eigh(projected)


def chebfsi_lowmem(matvec, block, n_want, *, degree=25, outer=3, cut, upper,
                   row_chunk=1 << 20, col_chunk=64):
    """Run ``outer`` filtered subspace iterations in place. Returns Ritz values.

    ``n_want`` is the number of states actually sought -- the sector's band
    share. The cut has to sit at the BAND EDGE, between the last wanted state
    and the first unwanted one. Putting it at the top of the block instead
    damps nothing, because the surplus columns are themselves inside the block,
    and the filter then has no separation to amplify: measured 4.5e-09 on
    h(theta) against 1.5e-10 with the cut placed correctly.
    """
    for _ in range(outer):
        filter_in_place(matvec, block, degree, cut, upper, col_chunk)
        cholesky_qr2(block, row_chunk)
        values, rotation = rayleigh_ritz(matvec, block, row_chunk, col_chunk)
        _apply_right(block, rotation, row_chunk)
        edge = min(n_want, len(values) - 1)
        cut = float(values[n_want - 1] + 0.5 * (values[edge] - values[n_want - 1]))
    return values, block
