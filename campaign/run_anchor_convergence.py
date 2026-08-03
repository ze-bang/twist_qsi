#!/usr/bin/env python3
"""Is the counterterm's spurious low-temperature anomaly physical, or is it an
under-converged anchor?

The counterterm C = -(1 - Delta)[F(z_0)] satisfies F_{H+C}(z_0) = Delta[F(z_0)]
exactly, and the right-hand side is block diagonal in the transport label, so it
carries the degeneracies that symmetry-related transport blocks force.  The
anchor is fixed by the Brillouin-Wigner condition z_0 = E_0(H + C).  AT that
fixed point the ground multiplet of H + C is an exact eigenvalue multiplet of
the masked map -- degeneracy included -- because that is the one energy at which
the counterterm is exact.

So the prediction is sharp: the splitting of the ground multiplet, and the
spurious low-temperature anomaly it produces, must vanish with the anchor
residual |E_0(H+C) - z_0| rather than saturating at some intrinsic value.  This
script measures that by tightening the anchor tolerance over many decades.

A plain fixed-point iteration converges linearly with rate ~ (1 - Z) and is too
slow to reach machine precision at large |Jpm|; a secant step on
g(z) = E_0(H + C(z)) - z is used instead.

Usage: run_anchor_convergence.py <jpm> [<jpm> ...]
"""

from __future__ import annotations

import json
import sys
import time
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

OUTPUT = ROOT / "campaign" / "outputs" / "anchor_convergence"
OUTPUT.mkdir(parents=True, exist_ok=True)
N_ICE = 90
N_SITES = 16
GRID = np.geomspace(1e-12, 4.0, 8000)


def heat_capacity(levels, grid=GRID):
    energies = np.sort(np.asarray(levels, dtype=float))
    energies = energies - energies[0]
    beta = 1.0 / grid[:, None]
    weight = np.exp(-beta * energies[None, :])
    partition = weight.sum(axis=1)
    mean = (weight * energies[None, :]).sum(axis=1) / partition
    second = (weight * energies[None, :] ** 2).sum(axis=1) / partition
    return (second - mean * mean) / (N_SITES * grid**2)


def main() -> None:
    couplings = [float(v) for v in sys.argv[1:] if not v.startswith("--")]
    cluster = geometry.build_cluster("cubic", (1, 1, 1))
    permutations = cubic16_translation_permutations(cluster)
    states = fixed_magnetization_basis(cluster.n_sites, cluster.n_sites // 2)
    index = {int(s): i for i, s in enumerate(states)}
    ice_rows = np.asarray([index[int(s)] for s in cluster.ice_states])
    outside = np.ones(len(states), dtype=bool)
    outside[ice_rows] = False
    complement = np.where(outside)[0]
    sectors = translation_sector_bases(states, permutations)
    labels = polarization_sector_labels(cluster)
    cross = labels[:, None] != labels[None, :]

    for jpm in couplings:
        started = time.perf_counter()
        hamiltonian = microscopic_character_hamiltonian(
            cluster, states, jpm, np.zeros(3)).tocsc().real

        def counterterm_spectrum(anchor, full=False):
            f_matrix = feshbach_matrix(hamiltonian, ice_rows, complement,
                                       anchor)
            corrected = (hamiltonian + embed(-np.where(cross, f_matrix, 0.0),
                                             ice_rows, len(states))).tocsc()
            if not full:
                ground = float(eigsh(corrected, k=1, which="SA", tol=1e-14,
                                     return_eigenvectors=False)[0])
                return ground, f_matrix, corrected
            levels = np.sort(np.concatenate([
                eigvalsh(np.ascontiguousarray(
                    (basis.conj().T @ (corrected @ basis)).toarray().real))
                for basis in sectors.values()]))
            return levels, f_matrix, corrected

        # secant iteration on g(z) = E_0(H + C(z)) - z, recording the whole
        # trajectory so that the anomaly can be watched as the anchor converges
        z_previous = float(eigsh(hamiltonian, k=1, which="SA", tol=1e-14,
                                 return_eigenvectors=False)[0])
        ground_previous, _, _ = counterterm_spectrum(z_previous)
        g_previous = ground_previous - z_previous
        z_current = ground_previous
        trajectory = []
        for step in range(24):
            ground, f_matrix, corrected = counterterm_spectrum(z_current)
            g_current = ground - z_current
            band = np.sort(np.concatenate([
                eigvalsh(np.ascontiguousarray(
                    (basis.conj().T @ (corrected @ basis)).toarray().real))
                for basis in sectors.values()]))[:N_ICE]
            masked = np.where(cross, 0.0, f_matrix)
            protected = []
            for sector in np.unique(labels):
                rows = np.flatnonzero(labels == sector)
                protected.extend(np.sort(eigvalsh(masked[np.ix_(rows, rows)])))
            protected = np.sort(np.asarray(protected))
            multiplicity = int(np.sum(protected - protected[0] < 1e-10))
            splitting = float(band[multiplicity - 1] - band[0])
            width = float(band[-1] - band[0])
            trajectory.append({
                "step": step, "anchor": z_current,
                "residual": abs(g_current),
                "ground_multiplicity": multiplicity,
                "splitting": splitting,
                "splitting_over_width": splitting / width,
                "band_width": width,
                "schottky_T": splitting / 2.4,
            })
            print(f"  jpm={jpm:+.3f} step {step:2d} residual={abs(g_current):.3e} "
                  f"split={splitting:.3e} ({splitting/width:.2e} of width) "
                  f"g={multiplicity}", flush=True)
            if abs(g_current) < 1e-13:
                break
            denominator = g_current - g_previous
            if abs(denominator) < 1e-16:
                z_next = ground
            else:
                z_next = z_current - g_current * (z_current - z_previous) \
                    / denominator
            z_previous, g_previous = z_current, g_current
            z_current = float(z_next)

        record = {"jpm": jpm, "trajectory": trajectory,
                  "anchor": trajectory[-1]["anchor"],
                  "runtime": time.perf_counter() - started}
        tag = f"{jpm:+.3f}".replace(".", "p")
        with open(OUTPUT / f"anchor_{tag}.json", "w") as handle:
            json.dump(record, handle, indent=1, default=float)

        first, last = trajectory[0], trajectory[-1]
        print(f"jpm={jpm:+.3f} residual {first['residual']:.2e} -> "
              f"{last['residual']:.2e} in {len(trajectory)} steps; "
              f"ground-multiplet splitting "
              f"{first['splitting']:.3e} -> {last['splitting']:.3e} "
              f"(g={last['ground_multiplicity']}); "
              f"spurious Schottky would sit at T={last['schottky_T']:.2e} "
              f"({last['runtime'] if 'runtime' in last else record['runtime']:.0f}s)",
              flush=True)


if __name__ == "__main__":
    main()
