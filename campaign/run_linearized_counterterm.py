#!/usr/bin/env python3
"""A counterterm linear in the energy: the O(1-Z) band error, to second order.

The fixed counterterm C = -(1-Delta)[F(z_0)] makes F_{H+C}(z) = Delta[F(z)] at
z = z_0 and nowhere else, so band levels away from the anchor inherit an error
of order (1-Z).  Measured, that is 0.33(1-Z) of the band width -- 16% at
Jpm/Jzz = -0.20.

Nothing forces the counterterm to be a constant.  Because F_{H+C}(z) = F(z) + C
holds identically for any C = PCP, taking

    C(z) = C_0 + C_1 (z - z_0),
    C_0  = -(1 - Delta)[F(z_0)],
    C_1  = -(1 - Delta)[F'(z_0)],      F'(z) = -(QHP)^dag (z - QHQ)^{-2} (QHP)

reproduces Delta[F(z)] through first order in (z - z_0) instead of zeroth, so
the band error should fall from O(1-Z) to O((1-Z)^2).

The price is that H + C(z) is a linear matrix pencil rather than a Hamiltonian.
Its eigenvalue condition

    det[ z - H - C_0 - C_1 (z - z_0) ] = 0
      = det[ z (1 - C_1) - (H + C_0 - z_0 C_1) ] = 0

is a generalized Hermitian eigenvalue problem

    (H + C_0 - z_0 C_1) psi = z (1 - C_1) psi ,

which has a real spectrum provided the metric 1 - C_1 is positive definite.
That is checked here rather than assumed: C_1 is a masked-out part of F', and
masking destroys the sign definiteness that F' <= 0 would otherwise give, so
positivity is a property of this problem and not a theorem.

Both F(z_0) and F'(z_0) come from the same shifted solves: one extra
application of (QHQ - z)^{-1} per ice column.

Usage: run_linearized_counterterm.py <jpm> [<jpm> ...]
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.linalg import eigh, eigvalsh
from scipy.sparse import identity
from scipy.sparse.linalg import cg, eigsh

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
from qsi_campaign.downfold import build_blocks, solve_roots  # noqa: E402
from run_exact_counterterm import embed  # noqa: E402

OUTPUT = ROOT / "campaign" / "outputs" / "linearized_counterterm"
OUTPUT.mkdir(parents=True, exist_ok=True)
N_ICE = 90
ANCHOR_TOL = 1.0e-13


def feshbach_and_derivative(hamiltonian, ice_rows, complement, z):
    """F(z) and dF/dz, from two shifted solves per ice column.

    With u solving (QHQ - z) u = (QHP) e_c one has
    F[:, c] = (PHP)[:, c] - (QHP)^dag u, and with w solving (QHQ - z) w = u,
    F'[:, c] = -(QHP)^dag w.
    """
    block = hamiltonian[complement][:, complement].tocsc()
    shifted = (block - z * identity(block.shape[0], format="csc")).tocsc()
    coupling = hamiltonian[complement][:, ice_rows].tocsc()
    ice_block = hamiltonian[ice_rows][:, ice_rows].toarray()
    f_matrix = np.zeros_like(ice_block)
    derivative = np.zeros_like(ice_block)
    for column in range(len(ice_rows)):
        rhs = np.asarray(coupling[:, [column]].todense()).ravel()
        first, info = cg(shifted, rhs, rtol=1e-11, atol=0.0, maxiter=20000)
        if info != 0:
            raise RuntimeError(f"CG failed on column {column}: {info}")
        second, info = cg(shifted, first, rtol=1e-11, atol=0.0, maxiter=20000)
        if info != 0:
            raise RuntimeError(f"CG (second solve) failed on {column}: {info}")
        f_matrix[:, column] = ice_block[:, column] - coupling.T @ first
        derivative[:, column] = -(coupling.T @ second)
    return (0.5 * (f_matrix + f_matrix.T),
            0.5 * (derivative + derivative.T))


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
    dimension = len(states)

    for jpm in couplings:
        started = time.perf_counter()
        hamiltonian = microscopic_character_hamiltonian(
            cluster, states, jpm, np.zeros(3)).tocsc().real

        # ---- the anchor, from the constant-counterterm problem (secant)
        def ground(anchor):
            f_matrix, _ = feshbach_and_derivative(hamiltonian, ice_rows,
                                                  complement, anchor)
            corrected = (hamiltonian + embed(-np.where(cross, f_matrix, 0.0),
                                             ice_rows, dimension)).tocsc()
            return float(eigsh(corrected, k=1, which="SA", tol=1e-14,
                               return_eigenvectors=False)[0])

        z_previous = float(eigsh(hamiltonian, k=1, which="SA", tol=1e-14,
                                 return_eigenvectors=False)[0])
        value = ground(z_previous)
        g_previous = value - z_previous
        z_current = value
        for _ in range(24):
            value = ground(z_current)
            g_current = value - z_current
            if abs(g_current) < ANCHOR_TOL:
                break
            denominator = g_current - g_previous
            step = (-g_current * (z_current - z_previous) / denominator
                    if abs(denominator) > 1e-16 else g_current)
            z_previous, g_previous = z_current, g_current
            z_current = float(z_current + step)
        anchor = z_current

        f_matrix, derivative = feshbach_and_derivative(
            hamiltonian, ice_rows, complement, anchor)
        c_zero = -np.where(cross, f_matrix, 0.0)
        c_one = -np.where(cross, derivative, 0.0)

        # ---- the pencil, and the check that its metric is positive definite
        metric_block = np.eye(N_ICE) - c_one
        metric_eigenvalues = eigvalsh(metric_block)
        metric_ok = bool(metric_eigenvalues.min() > 0.0)

        if not metric_ok:
            # Positivity was measured but not acted on in the first version, so
            # a non-positive-definite pencil crashed inside eigh instead of
            # being reported.  It is a real limit of the scheme, not an error:
            # once 1 - C_1 loses positivity the linear expansion of F is being
            # extrapolated past its useful range and the pencil is no longer a
            # Hermitian eigenvalue problem.
            record = {"jpm": jpm, "anchor": anchor,
                      "metric_min_eigenvalue": float(metric_eigenvalues.min()),
                      "metric_positive_definite": False,
                      "runtime": time.perf_counter() - started}
            tag = f"{jpm:+.3f}".replace(".", "p")
            with open(OUTPUT / f"linear_{tag}.json", "w") as handle:
                json.dump(record, handle, indent=1, default=float)
            print(f"jpm={jpm:+.3f} metric min eig="
                  f"{metric_eigenvalues.min():+.4f} -> NOT POSITIVE DEFINITE; "
                  f"the linearised pencil is ill posed here, skipped "
                  f"({record['runtime']:.0f}s)", flush=True)
            continue

        left = (hamiltonian
                + embed(c_zero - anchor * c_one, ice_rows, dimension)).tocsc()
        right = (identity(dimension, format="csc")
                 + embed(-c_one, ice_rows, dimension)).tocsc()

        constant = (hamiltonian + embed(c_zero, ice_rows, dimension)).tocsc()

        linear_levels, constant_levels = [], []
        for basis in sectors.values():
            a = np.ascontiguousarray(
                (basis.conj().T @ (left @ basis)).toarray().real)
            s = np.ascontiguousarray(
                (basis.conj().T @ (right @ basis)).toarray().real)
            a = 0.5 * (a + a.T)
            s = 0.5 * (s + s.T)
            linear_levels.append(eigh(a, s, eigvals_only=True))
            k = np.ascontiguousarray(
                (basis.conj().T @ (constant @ basis)).toarray().real)
            constant_levels.append(eigvalsh(0.5 * (k + k.T)))
        linear = np.sort(np.concatenate(linear_levels))[:N_ICE]
        constant_band = np.sort(np.concatenate(constant_levels))[:N_ICE]

        # ---- the exact masked reference
        blocks = build_blocks(cluster, states, ice_rows,
                              microscopic_character_hamiltonian(
                                  cluster, states, jpm, np.zeros(3)))
        masked = solve_roots(blocks, masked=True)
        roots = np.sort(masked.energies[masked.converged])

        record = {"jpm": jpm, "anchor": anchor,
                  "metric_min_eigenvalue": float(metric_eigenvalues.min()),
                  "metric_positive_definite": metric_ok,
                  "n_reference_roots": int(len(roots))}
        if len(roots) == N_ICE:
            width = float(constant_band[-1] - constant_band[0])
            record["band_width"] = width
            record["constant_band_error"] = float(
                np.max(np.abs(constant_band - roots)) / width)
            record["linear_band_error"] = float(
                np.max(np.abs(linear - roots)) / width)
            record["constant_ground_error"] = float(
                abs(constant_band[0] - roots[0]))
            record["linear_ground_error"] = float(abs(linear[0] - roots[0]))
            record["improvement"] = (record["constant_band_error"]
                                     / record["linear_band_error"]
                                     if record["linear_band_error"] > 0
                                     else float("inf"))

        record["runtime"] = time.perf_counter() - started
        tag = f"{jpm:+.3f}".replace(".", "p")
        np.savez_compressed(OUTPUT / f"linear_{tag}.npz", jpm=jpm,
                            linear_band=linear, constant_band=constant_band,
                            reference=roots, c_zero=c_zero, c_one=c_one)
        with open(OUTPUT / f"linear_{tag}.json", "w") as handle:
            json.dump(record, handle, indent=1, default=float)

        show = lambda k: ("---" if record.get(k) is None
                          else f"{record[k] * 100:.2f}%")
        print(f"jpm={jpm:+.3f} metric min eig={metric_eigenvalues.min():.4f} "
              f"({'ok' if metric_ok else 'NOT POSITIVE DEFINITE'}) | "
              f"band error constant {show('constant_band_error')} -> "
              f"linear {show('linear_band_error')} "
              f"(x{record.get('improvement', float('nan')):.2f}) | "
              f"ground error {record.get('constant_ground_error', float('nan')):.1e}"
              f" -> {record.get('linear_ground_error', float('nan')):.1e} "
              f"({record['runtime']:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
