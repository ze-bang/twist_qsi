#!/usr/bin/env python3
"""Frame-free longitudinal DSSF: the mask turned off against the mask turned on.

The polar-pullback route needed an isometry to carry spin operators from the
full Hilbert space into a 90-dimensional model space, and that isometry is the
part of the construction that fails first.  In the counterterm route no isometry
is needed at all, for a reason specific to S^z: the operator

    S^z_q = sum_i S^z_i exp(-i q . r_i)

is diagonal in the Ising configuration basis, so its projection onto the ice
manifold is exactly the diagonal matrix of its values on the 90 ice
configurations.  P S^z_q P requires no frame, no orthonormalisation and no Gram
inverse -- only the list of ice configurations.

That makes an exact apples-to-apples comparison possible.  Both spectra below
are eigenvalue problems for a 90 x 90 operator on the same basis, fed the same
diagonal sources, and differ *only* by whether the transport mask was applied:

    periodic       F(z_0)
    winding free   Delta[F(z_0)]

at the *same* anchor z_0 = E_0(H + C).  Using one anchor for both is the point:
the two operators then differ by exactly the matrix elements the mask deletes,
so the figure isolates the artifact rather than the downfolding.

A caveat that is easy to get wrong.  F(z) evaluated at a single energy is not
the exact band: only the branch whose self-consistent root sits at z is exact
there, which is the ground state at z = z_0.  Measured, the eigenvalues of the
unmasked F(z_0) differ from the exact lowest ninety by 5.5e-2 J_zz, so the
periodic panel must not be advertised as the exact periodic band -- it is the
unmasked downfolding at the anchor.  The exact-band claim belongs to the
self-consistent roots (3e-13), not to a fixed-energy evaluation.

A selection rule falls out.  The q = 0 sublattice magnetisations are *exactly*
constant within each transport sector on cubic-16 (verified: one value per
sector), because the transport label is a linear function of them.  So
P S^z_{q=0} P commutes with the mask and is a function of the conserved label:
it cannot connect the winding-free ground multiplet to anything, and the
winding-free longitudinal response at Gamma vanishes identically.  The unmasked
theory has weight there.  All of the Gamma-point longitudinal weight of ordinary
periodic ED on this cluster is therefore artifact -- which is a prediction of
the construction, not an input to it.

Usage: run_dssf_counterterm.py [<jpm>]   (default 0.046)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.linalg import eigh, eigvalsh
from scipy.sparse.linalg import eigsh

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "notes"))
sys.path.insert(0, str(ROOT / "campaign"))

import recompute_finite_size_artifact as geometry  # noqa: E402
from qsi_campaign.exact_band import (  # noqa: E402
    cubic16_translation_permutations,
    fixed_magnetization_basis,
    microscopic_character_hamiltonian,
    translation_sector_bases,
)
from qsi_campaign.protocol import polarization_sector_labels  # noqa: E402
from run_winding_residual import feshbach_matrix  # noqa: E402
from run_exact_counterterm import embed  # noqa: E402

OUTPUT = ROOT / "campaign" / "outputs"
N_ICE = 90
ANCHOR_TOL = 1.0e-13


def self_consistent_anchor(hamiltonian, ice_rows, complement, cross,
                           dimension, masked):
    """Secant solve of z = E_0(H + C(z)), or z = E_0(H) when masked is False."""
    if not masked:
        anchor = float(eigsh(hamiltonian, k=1, which="SA", tol=1e-14,
                             return_eigenvectors=False)[0])
        return anchor, feshbach_matrix(hamiltonian, ice_rows, complement,
                                       anchor), 0.0

    def ground(anchor):
        f_matrix = feshbach_matrix(hamiltonian, ice_rows, complement, anchor)
        corrected = (hamiltonian + embed(-np.where(cross, f_matrix, 0.0),
                                         ice_rows, dimension)).tocsc()
        value = float(eigsh(corrected, k=1, which="SA", tol=1e-14,
                            return_eigenvectors=False)[0])
        return value, f_matrix

    z_previous = float(eigsh(hamiltonian, k=1, which="SA", tol=1e-14,
                             return_eigenvectors=False)[0])
    value, f_matrix = ground(z_previous)
    g_previous = value - z_previous
    z_current = value
    for _ in range(24):
        value, f_matrix = ground(z_current)
        g_current = value - z_current
        if abs(g_current) < ANCHOR_TOL:
            break
        denominator = g_current - g_previous
        step = (-g_current * (z_current - z_previous) / denominator
                if abs(denominator) > 1e-16 else g_current)
        z_previous, g_previous = z_current, g_current
        z_current = float(z_current + step)
    return z_current, f_matrix, abs(g_current)


def diagonal_sources(cluster, momenta):
    """P S^z_q P on the ice manifold: exactly diagonal, no frame required."""
    ice = np.asarray(cluster.ice_states, dtype=np.uint64)
    bits = ((ice[:, None] >> np.arange(cluster.n_sites, dtype=np.uint64))
            & np.uint64(1))
    spins = bits.astype(float) - 0.5                      # 90 x n_sites
    sublattices = np.arange(cluster.n_sites) % 4
    sources = np.empty((len(momenta), 4, N_ICE), dtype=np.complex128)
    for index, momentum in enumerate(momenta):
        phase = np.exp(-2j * np.pi * (cluster.positions @ momentum))
        for sublattice in range(4):
            sources[index, sublattice] = spins @ (phase
                                                  * (sublattices == sublattice))
    return sources


def spectral_lines(operator, sources, n_sites, tolerance=1.0e-9):
    energies, vectors = eigh(operator)
    excitation = energies - energies[0]
    ground = np.flatnonzero(excitation <= tolerance)
    weights = np.zeros((sources.shape[0], len(energies)))
    for index in range(sources.shape[0]):
        for sublattice in range(4):
            # diagonal source: (V^dag D V) needs only the diagonal of D
            matrix = vectors.conj().T @ (sources[index, sublattice][:, None]
                                         * vectors)
            weights[index] += np.mean(
                np.abs(matrix[:, ground]) ** 2, axis=1) / n_sites
        weights[index, ground] = 0.0
    return excitation, weights, len(ground)


def broaden(excitation, weights, frequency, eta):
    distance = (frequency[:, None] - excitation[None, :]) / eta
    kernel = np.exp(-0.5 * distance**2) / (np.sqrt(2.0 * np.pi) * eta)
    return kernel @ weights.T


def main() -> None:
    jpm = float(sys.argv[1]) if len(sys.argv) > 1 else 0.046
    cluster = geometry.build_cluster("cubic", (1, 1, 1))
    permutations = cubic16_translation_permutations(cluster)
    states = fixed_magnetization_basis(cluster.n_sites, cluster.n_sites // 2)
    index = {int(s): i for i, s in enumerate(states)}
    ice_rows = np.asarray([index[int(s)] for s in cluster.ice_states])
    outside = np.ones(len(states), dtype=bool)
    outside[ice_rows] = False
    complement = np.where(outside)[0]
    labels = polarization_sector_labels(cluster)
    cross = labels[:, None] != labels[None, :]
    sectors = translation_sector_bases(states, permutations)

    hamiltonian = microscopic_character_hamiltonian(
        cluster, states, jpm, np.zeros(3)).tocsc().real

    # ---- one anchor, two operators: mask off and mask on
    z_counterterm, f_counterterm, residual = self_consistent_anchor(
        hamiltonian, ice_rows, complement, cross, len(states), masked=True)
    periodic_operator = 0.5 * (f_counterterm + f_counterterm.T)
    winding_free_operator = np.where(cross, 0.0, periodic_operator)
    winding_free_operator = 0.5 * (winding_free_operator
                                   + winding_free_operator.T)
    z_periodic = z_counterterm

    # how far the fixed-energy evaluation is from the exact band, reported so
    # that the periodic panel is not over-claimed
    exact = np.sort(np.concatenate([
        eigvalsh(np.ascontiguousarray(
            (basis.conj().T @ (hamiltonian @ basis)).toarray().real))
        for basis in sectors.values()]))[:N_ICE]
    unmasked_error = float(np.max(np.abs(
        np.sort(eigvalsh(periodic_operator)) - exact)))
    ground_error = float(abs(np.min(eigvalsh(winding_free_operator))
                             - z_counterterm))

    momenta = np.asarray(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0),
                          (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)))
    momentum_labels = np.asarray((r"$\Gamma$", r"$X_x$", r"$X_y$", r"$X_z$"))
    sources = diagonal_sources(cluster, momenta)

    periodic_excitation, periodic_weights, periodic_ground = spectral_lines(
        periodic_operator, sources, cluster.n_sites)
    free_excitation, free_weights, free_ground = spectral_lines(
        winding_free_operator, sources, cluster.n_sites)

    # the unmasked band is an order of magnitude wider than the masked one, so
    # the window has to cover both; that disparity is itself the result
    upper = 1.10 * max(float(periodic_excitation.max()),
                       float(free_excitation.max()))
    frequency = np.linspace(0.0, upper, 1400)
    eta = 6.0e-4
    periodic_spectrum = broaden(periodic_excitation, periodic_weights,
                                frequency, eta)
    free_spectrum = broaden(free_excitation, free_weights, frequency, eta)

    def summarise(excitation, weights):
        out = []
        for index in range(len(momenta)):
            active = weights[index] > 1.0e-12
            total = float(weights[index].sum())
            out.append({
                "integrated_weight": total,
                "silent": bool(total <= 1.0e-12),
                "first_active_frequency": float(
                    excitation[active].min() if np.any(active) else np.nan),
                "centroid": float(np.dot(excitation, weights[index]) / total
                                  if total > 0 else np.nan),
            })
        return out

    tag = f"{jpm:+.4f}".replace(".", "p")
    np.savez_compressed(
        OUTPUT / f"dssf_counterterm_{tag}.npz",
        jpm=jpm, frequency=frequency, eta=np.asarray(eta), momenta=momenta,
        momentum_labels=momentum_labels,
        periodic_excitation=periodic_excitation,
        periodic_weights=periodic_weights,
        winding_free_excitation=free_excitation,
        winding_free_weights=free_weights,
        periodic_spectrum=periodic_spectrum,
        winding_free_spectrum=free_spectrum,
        periodic_operator=periodic_operator,
        winding_free_operator=winding_free_operator)

    report = {
        "jpm": jpm,
        "anchor_periodic": z_periodic,
        "anchor_counterterm": z_counterterm,
        "anchor_residual": residual,
        "unmasked_at_anchor_vs_exact_max_error": unmasked_error,
        "masked_ground_vs_anchor": ground_error,
        "periodic_ground_degeneracy": periodic_ground,
        "winding_free_ground_degeneracy": free_ground,
        "periodic_band_width": float(np.ptp(periodic_excitation)),
        "winding_free_band_width": float(np.ptp(free_excitation)),
        "periodic": summarise(periodic_excitation, periodic_weights),
        "winding_free": summarise(free_excitation, free_weights),
        "sum_rule_periodic": float(periodic_weights.sum()),
        "sum_rule_winding_free": float(free_weights.sum()),
    }
    with open(OUTPUT / f"dssf_counterterm_{tag}.json", "w") as handle:
        json.dump(report, handle, indent=1, default=float)

    print(f"jpm={jpm:+.4f}")
    print(f"  single anchor z_0 = {z_counterterm:.9f} (residual "
          f"{residual:.1e}); masked ground vs anchor: {ground_error:.1e}")
    print(f"  unmasked F(z_0) vs the exact lowest 90: {unmasked_error:.3e} "
          f"-- a fixed-energy evaluation is NOT the exact band")
    print(f"  ground degeneracy: periodic {periodic_ground}, "
          f"winding free {free_ground}")
    print(f"  band width: periodic {report['periodic_band_width']:.6f}, "
          f"winding free {report['winding_free_band_width']:.6f}")
    for name in ("periodic", "winding_free"):
        first = ["silent" if entry["silent"]
                 else f"{entry['first_active_frequency']:.5f}"
                 for entry in report[name]]
        print(f"  {name:>13} first active frequency per momentum "
              f"(Gamma, Xx, Xy, Xz): {', '.join(first)}")
    print(f"  integrated weight at Gamma: periodic "
          f"{report['periodic'][0]['integrated_weight']:.3e}, winding free "
          f"{report['winding_free'][0]['integrated_weight']:.3e} "
          f"(the selection rule of the docstring)")


if __name__ == "__main__":
    main()
