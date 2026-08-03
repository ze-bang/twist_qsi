#!/usr/bin/env python3
"""Steps 1-3 of the resonance-extension program.

Per coupling (one dense QHQ eigendecomposition shared by all steps):

  step 1  golden-rule widths of the masked Feshbach roots,
          Gamma_n = 2*pi * sum_mu |<mu|QHP|v_n>|^2 * delta_eta(E_n - e_mu),
          with a Gaussian delta of width eta = ETA_FACTOR times the local
          pole spacing (reported for three eta values as a robustness band);

  step 2  second-sheet roots: complex fixed points z = eig_n F_eta(z) of the
          retarded smoothed map F_eta(z) = PHP + Sigma_eta(z),
          Sigma_eta(z) = B^T diag[1/(z - e_mu + i*eta)] B, tracked from the
          real roots by right-eigenvector overlap; Gamma2_n = -2 Im z_n;

  step 3  the winding-free band of poles: assemble the non-Hermitian
          effective operator from the complex roots and their right vectors,
          conjugate by the corner gauge phases (diagonal in the ice frame),
          average the eight corners, apply the polarization mask, and
          diagonalize: eigenvalues E - i*Gamma/2 of the winding-free
          resonant band.

Usage: run_gamow.py <jpm> [<jpm> ...]
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "notes"))

import recompute_finite_size_artifact as geometry  # noqa: E402
from qsi_campaign.exact_band import (  # noqa: E402
    basis_character_phases,
    fixed_magnetization_basis,
    microscopic_character_hamiltonian,
    uniform_character_grid,
)
from qsi_campaign.downfold import build_blocks, solve_roots  # noqa: E402
from qsi_campaign.protocol import sector_project  # noqa: E402

OUTPUT = ROOT / "campaign" / "outputs" / "gamow"
OUTPUT.mkdir(parents=True, exist_ok=True)
ETA_FACTORS = (2.0, 4.0, 8.0)


def local_spacing(poles, energy, k=40):
    nearest = np.sort(np.abs(poles - energy))[:k]
    window = np.sort(poles[np.argsort(np.abs(poles - energy))[:k]])
    return float(np.mean(np.diff(window)))


def golden_rule(blocks, vector, energy, eta):
    amplitude = blocks.coupling @ vector
    gaussian = np.exp(-0.5 * ((energy - blocks.poles) / eta) ** 2) \
        / (np.sqrt(2.0 * np.pi) * eta)
    return float(2.0 * np.pi * np.sum(np.abs(amplitude) ** 2 * gaussian))


def f_eta(blocks, z, eta):
    weights = 1.0 / (z - blocks.poles + 1j * eta)
    sigma = (blocks.coupling * weights[:, None]).T @ blocks.coupling
    return blocks.php + sigma


def masked_f_eta(blocks, z, eta):
    return sector_project(f_eta(blocks, z, eta), blocks.sector_labels)


def complex_roots(blocks, real_result, eta, masked=True, damping=0.5,
                  tol=1e-10, iterations=300):
    d = blocks.php.shape[0]
    roots = np.full(d, np.nan + 0j)
    vectors = np.zeros((d, d), dtype=complex)
    # branches without a bound root are continued from the continuum edge,
    # where their eigenvalue branch of the smoothed map sits: these are the
    # escaped states, i.e. the resonances.
    edge = float(blocks.poles.min())
    edge_matrix = masked_f_eta(blocks, complex(edge, -0.5 * eta), eta) \
        if masked else f_eta(blocks, complex(edge, -0.5 * eta), eta)
    edge_values = np.linalg.eigvals(edge_matrix)
    edge_values = edge_values[np.argsort(edge_values.real)]
    for n in range(d):
        if real_result.converged[n]:
            z = complex(real_result.energies[n], -0.5 * eta)
        else:
            z = complex(edge_values[n].real, min(edge_values[n].imag,
                                                 -0.5 * eta))
        tracker = None
        for _ in range(iterations):
            matrix = masked_f_eta(blocks, z, eta) if masked \
                else f_eta(blocks, z, eta)
            values, right = np.linalg.eig(matrix)
            if tracker is None:
                index = int(np.argmin(np.abs(values - z)))
            else:
                index = int(np.argmax(np.abs(tracker.conj() @ right)))
            tracker = right[:, index]
            update = (1.0 - damping) * z + damping * values[index]
            if abs(update - z) < tol:
                z = update
                break
            z = update
        roots[n] = z
        vectors[:, n] = tracker
    return roots, vectors


def main() -> None:
    couplings = [float(v) for v in sys.argv[1:]]
    cluster = geometry.build_cluster("cubic", (1, 1, 1))
    states = fixed_magnetization_basis(cluster.n_sites, cluster.n_sites // 2)
    index = {int(s): i for i, s in enumerate(states)}
    ice_indices = np.asarray([index[int(s)] for s in cluster.ice_states])
    ice_states = np.asarray(cluster.ice_states, dtype=np.uint64)
    corner_phases = [basis_character_phases(cluster, ice_states, np.asarray(t))
                     for t in uniform_character_grid(2)]

    for jpm in couplings:
        started = time.perf_counter()
        hamiltonian = microscopic_character_hamiltonian(
            cluster, states, jpm, np.zeros(3))
        blocks = build_blocks(cluster, states, ice_indices, hamiltonian)
        real_result = solve_roots(blocks, masked=True)
        n_conv = int(real_result.converged.sum())

        # step 1: golden-rule widths at three smoothing scales
        widths = {}
        for factor in ETA_FACTORS:
            gam = np.full(len(real_result.energies), np.nan)
            for n, (ok, energy) in enumerate(
                    zip(real_result.converged, real_result.energies)):
                if not ok:
                    continue
                eta = factor * local_spacing(blocks.poles, energy)
                gam[n] = golden_rule(
                    blocks, _root_vector(blocks, energy), energy, eta)
            widths[f"gamma1_eta{factor:g}"] = gam

        # step 2: second-sheet roots at the middle smoothing
        eta_mid = ETA_FACTORS[1] * local_spacing(
            blocks.poles, float(np.nanmax(real_result.energies)))
        z_roots, right = complex_roots(blocks, real_result, eta_mid)

        # step 3: winding-free band of poles
        finite = np.isfinite(z_roots)
        v = right[:, finite]
        z = z_roots[finite]
        try:
            v_inv = np.linalg.inv(v)
            h_eff = v @ np.diag(z) @ v_inv
            corners = [ph[:, None] * h_eff * ph.conj()[None, :]
                       for ph in corner_phases]
            averaged = np.mean(corners, axis=0)
            masked = sector_project(averaged, blocks.sector_labels)
            wf_poles = np.linalg.eigvals(masked)
            wf_poles = wf_poles[np.argsort(wf_poles.real)]
        except np.linalg.LinAlgError:
            wf_poles = np.full(int(finite.sum()), np.nan + 0j)

        tag = f"{jpm:+.3f}".replace(".", "p")
        np.savez_compressed(
            OUTPUT / f"gamow_{tag}.npz",
            jpm=jpm, energies=real_result.energies,
            converged=real_result.converged,
            ice_weight=real_result.ice_weight,
            eta_mid=eta_mid, z_roots=z_roots, wf_poles=wf_poles,
            **widths)
        gam2 = -2.0 * np.nanmean(z_roots.imag)
        print(f"jpm={jpm:+.3f} conv={n_conv}/90 "
              f"G1(mid)={np.nanmean(widths['gamma1_eta4'] ):.5f} "
              f"G2={gam2:.5f} wf_G="
              f"{-2.0*np.nanmean(wf_poles.imag):.5f} "
              f"({time.perf_counter()-started:.0f}s)", flush=True)


def _root_vector(blocks, energy):
    values, vectors = np.linalg.eigh(
        sector_project(blocks.php + blocks.self_energy(energy),
                       blocks.sector_labels))
    return vectors[:, int(np.argmin(np.abs(values - energy)))]


if __name__ == "__main__":
    main()
