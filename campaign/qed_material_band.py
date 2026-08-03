#!/usr/bin/env python3
"""QED-backed winding-free XYZ band for the Ce2Hf2O7 material points, M=2.

Independent cross-check of ``material_scan``: the microscopic band is
diagonalized by the trusted local QED package rather than by this repository's
own sparse solver.  The winding-free replacement, the counterterm complement,
and the thermodynamics are otherwise identical to the paper's prescription.

The XYZ Hamiltonian mirrors ``microscopic_character_hamiltonian`` exactly:

    H(theta) = Jzz sum Sz Sz
             - Jpm  sum ( e^{+2i theta.d_ij} S+_i S-_j + h.c. )
             + Jpmpm sum ( e^{-2i theta.c_ij} S+_i S+_j + h.c. ),

with d_ij = R_{j|i} - r_i and c_ij = r_i + R_{j|i}.  For Jpmpm=0 the solve
runs in the fixed-Sz sector; otherwise it uses the full Hilbert space.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from itertools import product
from pathlib import Path

import h5py
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "notes"))

# A stale editable copy of ``qed`` lives in site-packages and its
# ``point_group_routing`` submodule can be cached at interpreter startup by the
# qed_nlce editable finder, before this file runs.  Force the local QED source
# to win: put it first on the path and drop any pre-imported qed modules.
_LOCAL_QED = str(ROOT.parent / "QED" / "python")
sys.path.insert(0, _LOCAL_QED)
for _m in [m for m in list(sys.modules) if m == "qed" or m.startswith("qed.")]:
    del sys.modules[_m]

import qed  # noqa: E402
if not qed.__file__.startswith(_LOCAL_QED):
    raise ImportError(f"resolved stale qed at {qed.__file__}, expected under {_LOCAL_QED}")
import recompute_finite_size_artifact as R  # noqa: E402
from qsi_campaign.material_scan import _sector_blocks, KB_MEV_PER_K, R_GAS  # noqa: E402
from qsi_campaign.exact_band import (  # noqa: E402
    full_spin_basis, microscopic_character_hamiltonian,
    translation_sector_bases, cubic16_translation_permutations,
    basis_character_phases, uniform_character_grid,
)
from qsi_campaign.protocol import (  # noqa: E402
    sector_project, polarization_sector_labels, full_hilbert_counterterm_spectrum,
    character_project,
)
from qsi_campaign.thermodynamics import thermal_observables, peak_in_window  # noqa: E402


def fixed_sz_basis(n_sites: int, n_up: int) -> np.ndarray:
    if n_up == 0:
        return np.array([0], dtype=np.uint64)
    basis = []
    state = (1 << n_up) - 1
    limit = 1 << n_sites
    while state < limit:
        basis.append(state)
        c = state & -state
        r = state + c
        state = (((r ^ state) >> 2) // c) | r
    return np.asarray(basis, dtype=np.uint64)


def build_qed_xyz(cl, jpm, jpmpm, theta):
    """Twisted XYZ Hamiltonian as a QED Operator (full Hilbert space)."""
    op = qed.Operator(num_sites=cl.n_sites, spin=0.5)
    theta = np.asarray(theta, dtype=float)
    for (i, j), n in zip(cl.bonds, cl.bond_wrap):
        i, j = int(i), int(j)
        op.add_two_body(qed.OP_SZ, i, qed.OP_SZ, j, complex(1.0, 0.0))
        lifted_j = cl.positions[j] - np.asarray(n) @ cl.Lvecs
        d_ij = lifted_j - cl.positions[i]
        pm = complex(-jpm * np.exp(2j * float(theta @ d_ij)))
        op.add_two_body(qed.OP_SPLUS, i, qed.OP_SMINUS, j, pm)
        op.add_two_body(qed.OP_SMINUS, i, qed.OP_SPLUS, j, np.conjugate(pm))
        if jpmpm != 0.0:
            c_ij = cl.positions[i] + lifted_j
            pp = complex(jpmpm * np.exp(-2j * float(theta @ c_ij)))
            op.add_two_body(qed.OP_SPLUS, i, qed.OP_SPLUS, j, pp)
            op.add_two_body(qed.OP_SMINUS, i, qed.OP_SMINUS, j, np.conjugate(pp))
    return op


def _h5_complex(ds):
    arr = ds[()]
    if arr.dtype.fields and "real" in arr.dtype.fields:
        return np.asarray(arr["real"] + 1j * arr["imag"], dtype=np.complex128)
    return np.asarray(arr, dtype=np.complex128)


def qed_low_band(cl, jpm, jpmpm, theta, n_band, solver, tol, max_iter, device):
    op = build_qed_xyz(cl, jpm, jpmpm, theta)
    kwargs = dict(
        num_eigenvalues=n_band + 1, compute_eigenvectors=True, solver=solver,
        device=device, tolerance=tol, max_iterations=max_iter, verbose=False,
        spin_flip="off", time_reversal="off", symmetry=None, point_group=False,
    )
    if jpmpm == 0.0:
        kwargs.update(sz=cl.n_sites // 2, auto_sz=False)
    else:
        kwargs.update(sz=None, auto_sz=False)
    with tempfile.TemporaryDirectory() as tmp:
        t0 = time.time()
        res = qed.solve(op, output_dir=tmp, **kwargs)
        evals = np.asarray(res.eigenvalues, dtype=float)
        with h5py.File(res.eigenvectors_path, "r") as h5:
            vecs = [_h5_complex(h5[f"eigendata/eigenvector_{k}"]) for k in range(len(evals))]
        elapsed = time.time() - t0
    psi = np.column_stack(vecs)
    order = np.argsort(evals)
    return evals[order], psi[:, order], elapsed


def band_operator(evals, psi, ice_indices, n_band):
    """Polar (Loewdin) pullback of the lowest n_band states to the ice basis."""
    evals = evals[:n_band]
    psi = psi[:, :n_band]
    d = np.diag(evals)
    gram = psi.conj().T @ psi
    sv, U = np.linalg.eigh(0.5 * (gram + gram.conj().T))
    inv = U @ np.diag(1.0 / np.sqrt(sv)) @ U.conj().T
    m = 0.5 * (gram @ d + d @ gram)
    h_orth = 0.5 * ((inv @ m @ inv) + (inv @ m @ inv).conj().T)
    x = psi[ice_indices, :] @ inv
    g = x.conj().T @ x
    s2, V = np.linalg.eigh(0.5 * (g + g.conj().T))
    q = x @ (V @ np.diag(1.0 / np.sqrt(s2)) @ V.conj().T)
    h = q @ h_orth @ q.conj().T
    return 0.5 * (h + h.conj().T), float(np.min(s2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jpm", type=float, required=True)
    ap.add_argument("--jpmpm", type=float, default=0.0)
    ap.add_argument("--n-grid", type=int, default=2)
    ap.add_argument("--n-band", type=int, default=90)
    ap.add_argument("--solver", default="krylov_schur")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--tol", type=float, default=1e-10)
    ap.add_argument("--max-iter", type=int, default=2000)
    ap.add_argument("--ja-mev", type=float, default=0.050)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    cl = R.build_cluster("cubic", (1, 1, 1))
    if args.jpmpm == 0.0:
        basis = fixed_sz_basis(cl.n_sites, cl.n_sites // 2)
        index = {int(s): k for k, s in enumerate(basis)}
    else:
        index = {s: s for s in range(1 << cl.n_sites)}  # QED full basis: state == index
    ice_indices = np.asarray([index[int(s)] for s in cl.ice_states], dtype=np.int64)

    print(f"QED winding-free XYZ: Jpm={args.jpm:+.4f} Jpmpm={args.jpmpm:+.4f} "
          f"M={args.n_grid}, basis {'full' if args.jpmpm else 'Sz=0'}", flush=True)

    # The M-grid corner Hamiltonians obey the exact gauge identity
    # H(theta) = V(theta) H(0) V(theta)^dag with V diagonal (paper Thm A), so
    # they share H(0)'s spectrum and each corner band operator is the
    # phase-dressed theta=0 band.  One QED solve therefore suffices; the M^3
    # winding-free average is a gauge dressing of the single theta=0 band.
    evals, psi, dt = qed_low_band(cl, args.jpm, args.jpmpm, np.zeros(3),
                                  args.n_band, args.solver, args.tol, args.max_iter, args.device)
    h_periodic, overlap0 = band_operator(evals, psi, ice_indices, args.n_band)
    print(f"  theta=0 QED solve {dt:.1f}s E0={evals[0]:.8f} overlap_min={overlap0:.4f}", flush=True)

    ice_states = np.asarray(cl.ice_states, dtype=np.uint64)
    corners = []
    for theta in uniform_character_grid(args.n_grid):
        ph = basis_character_phases(cl, ice_states, np.asarray(theta))
        corners.append(ph[:, None] * h_periodic * ph.conj()[None, :])
    labels = polarization_sector_labels(cl)
    h_wf = sector_project(character_project(np.asarray(corners)), labels)
    overlaps = [overlap0]

    # Counterterm complement from the verified symmetry-block full spectrum.
    states = full_spin_basis(cl.n_sites)
    sectors = translation_sector_bases(states, cubic16_translation_permutations(cl))
    H0 = microscopic_character_hamiltonian(cl, states, args.jpm, np.zeros(3), jpmpm=args.jpmpm)
    full = np.sort(np.concatenate([np.linalg.eigvalsh(b) for b in _sector_blocks(sectors, states, H0)]))

    T = np.geomspace(1e-4, 2.0, 4000)
    peaks = {}
    for name, band in (("winding_free", h_wf), ("periodic", h_periodic)):
        spec = full_hilbert_counterterm_spectrum(full, np.linalg.eigvalsh(band))
        C = thermal_observables(spec, T, n_sites=16)["heat_capacity_per_site"]
        Tpk, _ = peak_in_window(T, C, 5e-5, 3e-2)
        peaks[name] = Tpk

    g6 = 12.0 * abs(args.jpm) ** 3
    out = {
        "backend": "QED", "jpm": args.jpm, "jpmpm": args.jpmpm, "M": args.n_grid,
        "ice_overlap_min": float(min(overlaps)),
        "g6_Jzz": g6,
        "winding_free_peak_Jzz": peaks["winding_free"],
        "periodic_peak_Jzz": peaks["periodic"],
        "winding_free_peak_over_g6": peaks["winding_free"] / g6,
        "winding_free_peak_mK": peaks["winding_free"] * args.ja_mev / KB_MEV_PER_K * 1e3,
        "periodic_peak_mK": peaks["periodic"] * args.ja_mev / KB_MEV_PER_K * 1e3,
    }
    print(json.dumps(out, indent=2))
    outpath = args.out or (ROOT / "campaign" / "outputs" /
        f"qed_windfree_jpm{args.jpm:+.4f}_pp{args.jpmpm:+.4f}_M{args.n_grid}.json".replace("+", "p"))
    outpath.parent.mkdir(parents=True, exist_ok=True)
    outpath.write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {outpath}")


if __name__ == "__main__":
    main()
