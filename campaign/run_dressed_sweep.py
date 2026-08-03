#!/usr/bin/env python3
"""Winding-free construction with the dressed band as its own reference frame.

The original protocol maps the exact ice-descended band onto the bare ice
basis (P_ice) before averaging over the M=2 gauge corners.  Here the
reference is instead the exact theta = 0 band itself:

  * corner overlap   A_c = Psi0^dag U_c Psi0   (U_c the diagonal corner gauge
    unitary on the full basis),
  * corner transport V_c = polar(A_c),  h_c = V_c diag(E) V_c^dag,
  * winding-free     h_wf = mask( mean_c h_c ),

with the polarization mask built from the band-projected polarization
operator (its eigenvalues cluster into the six transport sectors; the
cluster spread is the frame-free diagnostic of sector sharpness).  No bare
ice projector enters anywhere except for the side-by-side comparison with
the original construction, computed from the same exact eigendata.

Usage: run_dressed_sweep.py <jpm> [<jpm> ...]; one npz per coupling in
campaign/outputs/dressed_sweep/.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from scipy.linalg import eigh, eigvalsh, svd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "notes"))

import recompute_finite_size_artifact as geometry  # noqa: E402
from qsi_campaign.exact_band import (  # noqa: E402
    basis_character_phases,
    fixed_magnetization_basis,
    microscopic_character_hamiltonian,
    translation_sector_bases,
    cubic16_translation_permutations,
    uniform_character_grid,
)
from qsi_campaign.protocol import (  # noqa: E402
    polarization_sector_labels,
    sector_project,
)

OUTPUT = ROOT / "campaign" / "outputs" / "dressed_sweep"
OUTPUT.mkdir(parents=True, exist_ok=True)
N_BAND = 90


def band_peak(levels, lo=1e-7, hi=0.2, n=2600):
    shifted = np.asarray(levels) - np.min(levels)
    grid = np.geomspace(lo, hi, n)
    beta = 1.0 / grid[:, None]
    w = np.exp(-beta * shifted[None, :])
    z = w.sum(axis=1)
    mean = (w * shifted[None, :]).sum(axis=1) / z
    second = (w * shifted[None, :] ** 2).sum(axis=1) / z
    heat = (second - mean * mean) / grid**2
    return float(grid[int(np.argmax(heat))])


def sector_assign(values, ideal):
    """Assign each eigenvalue to the nearest quantized transport value.

    The ideal values are geometric (the allowed polarization sectors of the
    cluster), so the assignment is frame free; the maximum drift from the
    ideal is the diagnostic of sector sharpness.
    """
    labels = np.argmin(np.abs(values[:, None] - ideal[None, :]), axis=1)
    drift = float(np.max(np.abs(values - ideal[labels])))
    separation = float(np.min(np.diff(np.sort(ideal))))
    return labels, drift, separation


def main() -> None:
    couplings = [float(v) for v in sys.argv[1:]]
    cluster = geometry.build_cluster("cubic", (1, 1, 1))
    states = fixed_magnetization_basis(cluster.n_sites, cluster.n_sites // 2)
    index = {int(s): i for i, s in enumerate(states)}
    ice_rows = np.asarray([index[int(s)] for s in cluster.ice_states])
    ice_labels = polarization_sector_labels(cluster)
    sectors = translation_sector_bases(
        states, cubic16_translation_permutations(cluster))

    bits = ((np.asarray(states, dtype=np.uint64)[:, None]
             >> np.arange(cluster.n_sites, dtype=np.uint64)) & np.uint64(1))
    polarization = (bits.astype(float) - 0.5) @ np.asarray(
        cluster.positions, dtype=float)
    corner_thetas = list(uniform_character_grid(2))
    weights = np.array([1.0, np.sqrt(2.0), np.sqrt(3.0)])
    ice_mixer = (polarization[ice_rows] @ weights)
    ideal_values = np.unique(np.round(ice_mixer, 9))
    ideal_counts = np.array(
        [int(np.sum(np.abs(ice_mixer - v) < 1e-8)) for v in ideal_values])

    for jpm in couplings:
        started = time.perf_counter()
        hamiltonian = microscopic_character_hamiltonian(
            cluster, states, jpm, np.zeros(3))
        all_values, band_vals, band_vecs = [], [], []
        for basis in sectors.values():
            block = (basis.conj().T @ (hamiltonian @ basis)).toarray()
            block = 0.5 * (block + block.conj().T)
            all_values.append(eigvalsh(block))
            vals, vecs = eigh(block, subset_by_index=(0, N_BAND + 19))
            band_vals.append(vals)
            band_vecs.append(np.asarray(basis @ vecs))
        full_spectrum = np.sort(np.concatenate(all_values))
        band_vals = np.concatenate(band_vals)
        band_vecs = np.concatenate(band_vecs, axis=1)
        order = np.argsort(band_vals)[:N_BAND]
        energies = band_vals[order]
        psi = band_vecs[:, order]

        # --- dressed frame -------------------------------------------------
        corners, sigma_min = [], 1.0
        for theta in corner_thetas:
            phases = basis_character_phases(cluster, states, np.asarray(theta))
            overlap = psi.conj().T @ (phases[:, None] * psi)
            u, s, vh = svd(overlap)
            sigma_min = min(sigma_min, float(s.min()))
            rotation = u @ vh
            corners.append(rotation @ np.diag(energies) @ rotation.conj().T)
        averaged = np.mean(corners, axis=0)
        projected_p = np.stack([
            psi.conj().T @ (polarization[:, α][:, None] * psi)
            for α in range(3)])
        mixer = (weights[0] * projected_p[0] + weights[1] * projected_p[1]
                 + weights[2] * projected_p[2])
        mixer = 0.5 * (mixer + mixer.conj().T)
        mix_vals, mix_vecs = np.linalg.eigh(mixer)
        labels, spread, separation = sector_assign(mix_vals, ideal_values)
        counts = np.array([int(np.sum(labels == i))
                           for i in range(len(ideal_values))])
        sectors_intact = bool(np.array_equal(counts, ideal_counts))
        rotated = mix_vecs.conj().T @ averaged @ mix_vecs
        masked = sector_project(rotated, labels)
        dressed_band = np.linalg.eigvalsh(masked)

        # --- original ice frame from the same eigendata ---------------------
        overlap_ice = psi[ice_rows, :]
        gram = overlap_ice @ overlap_ice.conj().T
        gram_vals, gram_vecs = np.linalg.eigh(gram)
        ice_ok = bool(gram_vals.min() > 1e-10)
        ice_band = None
        if ice_ok:
            inv_sqrt = gram_vecs @ np.diag(gram_vals**-0.5) @ gram_vecs.conj().T
            frame = overlap_ice.conj().T @ inv_sqrt
            h_ice = frame.conj().T * energies @ frame
            ice_states = np.asarray(cluster.ice_states, dtype=np.uint64)
            ice_corners = []
            for theta in corner_thetas:
                ph = basis_character_phases(cluster, ice_states, np.asarray(theta))
                ice_corners.append(ph[:, None] * h_ice * ph.conj()[None, :])
            ice_band = np.linalg.eigvalsh(
                sector_project(np.mean(ice_corners, axis=0), ice_labels))

        record = {
            "jpm": jpm,
            "energies": energies,
            "full_spectrum": full_spectrum,
            "gap_90": float(band_vals[np.argsort(band_vals)[N_BAND]]
                            - energies[-1]),
            "dressed_band": dressed_band,
            "dressed_peak": band_peak(dressed_band),
            "corner_sigma_min": sigma_min,
            "sector_spread": spread,
            "sector_separation": separation,
            "sectors_intact": sectors_intact,
            "ice_ok": ice_ok,
            "ice_gram_min": float(gram_vals.min()),
        }
        if ice_band is not None:
            record["ice_band"] = ice_band
            record["ice_peak"] = band_peak(ice_band)
        tag = f"{jpm:+.3f}".replace(".", "p")
        np.savez_compressed(OUTPUT / f"dressed_{tag}.npz", **record)
        g6 = 12.0 * abs(jpm) ** 3
        print(f"jpm={jpm:+.3f} dressed_peak/g6={record['dressed_peak']/g6:.4f} "
              + (f"ice_peak/g6={record['ice_peak']/g6:.4f} " if ice_band is not None
                 else "ice_frame=FAILED ")
              + f"sigma_min={sigma_min:.3f} drift={spread:.3f} sep={separation:.3f} "
              f"intact={sectors_intact} "
              f"({time.perf_counter()-started:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
