#!/usr/bin/env python3
"""Fully winding-free ED at arbitrary coupling via two nonperturbative counterterms.

The spurious winding four-cycle flip has no bare matrix element in the XXZ
Hamiltonian: it is generated entirely by virtual excursions.  We therefore
add the microscopic four-site ring operator on the 36 winding cycles,

    H_c = H + c * sum_W ( O_W + O_W^dag ),

and tune the single scalar c (all cycles equivalent by cubic symmetry) so
that the EXACT effective winding amplitude vanishes: on the two-dimensional
ice block P2 = span{|a>, |b>} of one winding pair,

    A(c) = c + [ P2 H_c Q (z - Q H_c Q)^{-1} Q H_c P2 ]_{ab}  = 0 ,

evaluated at z = the ice-block diagonal energy (self-consistently updated).
This requires no band, no reference frame, and no spectral isolation:
one sparse linear solve per iteration, valid at any Jpm.  At weak coupling
c -> +4 Jpm^2/Jzz (cancelling the second-order amplitude -4 Jpm^2/Jzz).

For each coupling the script then diagonalizes H_c (dense translation
sectors, lowest levels) and records the spectrum, the band-only ring peak,
the ground-state hexagon flux (the flux-law restoration test), and the
polarization sector drift of the low band.

Usage: run_counterterm.py <jpm> [<jpm> ...]
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from scipy.linalg import eigh
from scipy.sparse import csr_matrix, identity
from scipy.sparse.linalg import cg

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "notes"))

import recompute_finite_size_artifact as geometry  # noqa: E402
from qsi_campaign.exact_band import (  # noqa: E402
    fixed_magnetization_basis,
    microscopic_character_hamiltonian,
    translation_sector_bases,
    cubic16_translation_permutations,
)

OUTPUT = ROOT / "campaign" / "outputs" / "counterterm3"
OUTPUT.mkdir(parents=True, exist_ok=True)
N_KEEP = 150


def loop_operator(states, index, sites):
    """Sparse O + O^dag for one alternating loop (4 or 6 sites)."""
    mask = np.uint64(0)
    pattern = np.uint64(0)
    for position, site in enumerate(sites):
        mask |= np.uint64(1) << np.uint64(site)
        if position % 2 == 0:
            pattern |= np.uint64(1) << np.uint64(site)
    other = mask ^ pattern
    rows, cols = [], []
    for row, state in enumerate(states):
        occupied = np.uint64(state) & mask
        if occupied == pattern or occupied == other:
            rows.append(row)
            cols.append(index[int(np.uint64(state) ^ mask)])
    return csr_matrix((np.ones(len(rows)), (rows, cols)),
                      shape=(len(states),) * 2)


def winding_pair(cluster, index, sites):
    """Two ice states connected by flipping the given winding cycle."""
    mask = 0
    for site in sites:
        mask |= 1 << int(site)
    ice = set(int(s) for s in cluster.ice_states)
    for candidate in ice:
        flipped = candidate ^ mask
        if flipped != candidate and flipped in ice:
            return index[candidate], index[flipped]
    raise RuntimeError("no flippable ice pair on the winding cycle")


def exact_winding_amplitude(h_c, a, b, z, complement):
    """[P H Q (z - QHQ)^{-1} Q H P]_{ab} + bare, P = the full ice manifold.

    The complement contains only excited (charged) states, so for z below
    the two-spinon spectrum the shifted block QHQ - z is positive definite
    and conjugate gradients converge quickly.
    """
    hqq = h_c[np.ix_(complement, complement)]
    col_b = np.asarray(h_c[:, [b]].todense()).ravel()[complement]
    row_a = np.asarray(h_c[[a], :].todense()).ravel()[complement]
    shifted = (hqq - z * identity(hqq.shape[0], format="csc")).tocsc()
    x, info = cg(shifted, col_b, rtol=1e-12, atol=0.0, maxiter=5000)
    if info != 0:
        raise RuntimeError(f"CG failed: info={info}")
    bare = h_c[a, b]
    return float(bare - row_a @ x)


def main() -> None:
    couplings = [float(v) for v in sys.argv[1:]]
    cluster = geometry.build_cluster("cubic", (1, 1, 1))
    states = fixed_magnetization_basis(cluster.n_sites, cluster.n_sites // 2)
    index = {int(s): i for i, s in enumerate(states)}
    dim = len(states)
    sectors = translation_sector_bases(
        states, cubic16_translation_permutations(cluster))

    # three winding classes: axial four-cycles (|w| = (0,0,1)), diagonal
    # four-cycles (|w| = (0,1,1)), wrapping hexagons (all (0,1,1))
    axial = [s4 for s4, w in cluster.loops4
             if sorted(map(abs, w)) == [0, 0, 1]]
    diagonal = [s4 for s4, w in cluster.loops4
                if sorted(map(abs, w)) == [0, 1, 1]]
    wrap_sites = [sites for sites, w in cluster.hexes
                  if tuple(w) != (0, 0, 0)]
    class_ops = [sum(loop_operator(states, index, s) for s in group)
                 for group in (axial, diagonal, wrap_sites)]
    ice_rows = np.asarray([index[int(s)] for s in cluster.ice_states])
    keep = np.ones(dim, dtype=bool)
    keep[ice_rows] = False
    complement = np.where(keep)[0]
    hexes = [sites for sites, w in cluster.hexes if tuple(w) == (0, 0, 0)]
    hex_ops = [loop_operator(states, index, sites) for sites in hexes]
    pairs = [winding_pair(cluster, index, group[0])
             for group in (axial, diagonal, wrap_sites)]

    for jpm in couplings:
        started = time.perf_counter()
        h0 = microscopic_character_hamiltonian(
            cluster, states, jpm, np.zeros(3)).tocsc().real

        # evaluation point: the true ground-state energy of H0 (the
        # counterterm shifts it negligibly), from one cheap eigensolve
        z = min(
            float(eigh((basis.conj().T @ (h0 @ basis)).toarray().real,
                       subset_by_index=(0, 0), eigvals_only=True)[0])
            for basis in sectors.values())

        coeffs = np.array([4.0 * jpm * jpm, 4.0 * jpm * jpm,
                           12.0 * abs(jpm) ** 3 * np.sign(-jpm)])
        history = []
        for iteration in range(40):
            h_c = (h0 + sum(cv * op for cv, op in
                            zip(coeffs, class_ops))).tocsc()
            amps = np.array([
                exact_winding_amplitude(h_c, *pair, z, complement)
                for pair in pairs])
            history.append(np.concatenate([coeffs, amps]))
            if np.max(np.abs(amps)) < 1e-10:
                break
            coeffs = coeffs - amps      # joint Newton, Jacobian ~ identity

        # exact diagonalization of the counterterm Hamiltonian
        h_c = (h0 + sum(cv * op for cv, op in
                        zip(coeffs, class_ops))).tocsc()
        vals, vecs = [], []
        for basis in sectors.values():
            block = (basis.conj().T @ (h_c @ basis)).toarray()
            block = 0.5 * (block + block.conj().T)
            v, w = eigh(block, subset_by_index=(0, N_KEEP - 1))
            vals.append(v)
            vecs.append(np.asarray(basis @ w))
        vals = np.concatenate(vals)
        vecs = np.concatenate(vecs, axis=1)
        order = np.argsort(vals)[:N_KEEP]
        levels = vals[order]
        psi = vecs[:, order]

        # flux averaged over the quasi-degenerate ground multiplet: the
        # counterterms decouple the polarization sectors, so a single eigh
        # vector inside the multiplet is basis-arbitrary.
        multiplet = psi[:, levels - levels[0] < 1e-6]
        if multiplet.shape[1] < 2:
            multiplet = psi[:, :6]
        flux = np.mean([
            float(np.real(np.einsum("in,in->", multiplet.conj(),
                                    op @ multiplet))) / multiplet.shape[1]
            for op in hex_ops])

        band = levels[:90] - levels[0]
        grid = np.geomspace(1e-7, 0.2, 2600)
        beta = 1.0 / grid[:, None]
        w = np.exp(-beta * band[None, :])
        zf = w.sum(axis=1)
        mean = (w * band[None, :]).sum(axis=1) / zf
        second = (w * band[None, :] ** 2).sum(axis=1) / zf
        heat = (second - mean**2) / grid**2
        peak = float(grid[int(np.argmax(heat))])

        tag = f"{jpm:+.3f}".replace(".", "p")
        np.savez_compressed(
            OUTPUT / f"counterterm_{tag}.npz",
            jpm=jpm, coeffs=coeffs, history=np.asarray(history),
            levels=levels, band_peak=peak, gs_flux=flux)
        g6 = 12.0 * abs(jpm) ** 3
        print(f"jpm={jpm:+.3f} c4a={coeffs[0]:+.6f} c4d={coeffs[1]:+.6f} "
              f"c6={coeffs[2]:+.6f} iters={len(history)} "
              f"peak/g6={peak/g6:.4f} flux6={flux:+.4f} "
              f"({time.perf_counter()-started:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
