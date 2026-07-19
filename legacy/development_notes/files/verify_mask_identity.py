#!/usr/bin/env python3
"""Verify the two structure theorems behind the M=2 character protocol.

Theorem A (isospectrality): at theta in {0,pi}^3 the dipole2-deformed
microscopic Hamiltonian is unitarily equivalent to the undeformed one via the
diagonal polarization unitary V(theta) = exp(2i theta . X), X = sum_i r_i S_i^z,
because the leftover boundary-twist factor exp(-2i theta . n_ij L) is trivial
when L is an integer matrix.  Consequence: all eight corner spectra coincide,
which is WHY observable averaging provably cannot remove the artifact.

Theorem B (mask identity): the eight-corner QED low-band operator average
equals the theta=0 band operator with every ice-basis entry (alpha, beta)
killed unless 2*(x_alpha - x_beta) is componentwise even, where
x_alpha = sum_i r_i S_i^z(alpha) is the ice-state polarization.  The entire
M=2 character average is a single diagonalization plus a Z_2^3
polarization-parity mask.

Negative control: the dipole4 M=3 grid contains genuine boundary twists
(2 L^T theta not in 2 pi Z^3), so its spectra split and it is NOT a mask.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import recompute_finite_size_artifact as R  # noqa: E402


def ice_polarizations(cl) -> np.ndarray:
    ice = np.asarray(cl.ice_states, dtype=np.uint64)
    n = cl.n_sites
    sz = ((ice[:, None] >> np.arange(n, dtype=np.uint64)) & np.uint64(1)).astype(float) - 0.5
    return sz @ cl.positions


def parity_mask(x: np.ndarray, tol: float = 1e-9) -> np.ndarray:
    diff = 2.0 * (x[:, None, :] - x[None, :, :])
    frac = np.mod(diff, 2.0)
    return np.all((np.abs(frac) < tol) | (np.abs(frac - 2.0) < tol), axis=2)


def main() -> int:
    cl = R.build_cluster("cubic", (1, 1, 1))
    keep = parity_mask(ice_polarizations(cl))
    ok = True

    for tag in ("jm0p03", "jm0p05"):
        f = HERE / f"twist_resolved_qed_dipole2_M2_{tag}.npz"
        d = np.load(f, allow_pickle=True)
        spread = float(np.max(np.ptp(d["E_low_by_twist"], axis=0)))
        resid = float(np.abs(np.where(keep, d["H_qed_phi0"], 0.0) - d["H_qed_twist_avg"]).max())
        print(f"{f.name}: corner-spectra spread {spread:.3e}, mask residual {resid:.3e}")
        ok &= spread < 1e-10 and resid < 1e-10

    f = HERE / "twist_resolved_qed_dipole4_M3_jm0p05.npz"
    d = np.load(f, allow_pickle=True)
    spread = float(np.max(np.ptp(d["E_low_by_twist"], axis=0)))
    print(f"{f.name}: corner-spectra spread {spread:.3e} (must be LARGE: genuine twists)")
    ok &= spread > 1e-6

    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
