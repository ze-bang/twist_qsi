#!/usr/bin/env python3
"""QED-backed twist-resolved full-Hamiltonian band extraction.

This script diagonalizes the microscopic 16-site XXZ Hamiltonian in the full
fixed-Sz sector, projects the lowest ice-like band onto the ice basis, and
averages the resulting band operator over a smooth twist grid.

It is deliberately independent of git history.  Geometry, ice states, and the
perturbative reference are rebuilt by ``recompute_finite_size_artifact.py``.
The full-Hamiltonian eigenvectors are computed by the local QED package.
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
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
QED_PY = ROOT / "QED" / "python"
if QED_PY.exists():
    sys.path.insert(0, str(QED_PY))

import qed  # noqa: E402
import recompute_finite_size_artifact as R  # noqa: E402

FIGS = HERE / "figs"
FIGS.mkdir(exist_ok=True)


def fixed_sz_basis(n_sites: int, n_up: int) -> np.ndarray:
    """Reproduce QED's ``generateFixedSzBasis`` ordering."""
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


def build_qed_operator(cl: R.Cluster, jpm: float, phi: np.ndarray, twist_kind: str):
    """Build the twisted microscopic Hamiltonian.

    ``physical`` uses a uniform vector potential A = phi L^{-1}.
    ``dipole2`` and ``dipole4`` use auxiliary source fields conjugate to 2 or
    4 times the transported dipole.  They are not ordinary flux insertions;
    they are microscopic deformations whose perturbative paths acquire
    exp(-i theta . q), with q = 2 delta or 4 delta.
    """
    op = qed.Operator(num_sites=cl.n_sites, spin=0.5)
    phi = np.asarray(phi, dtype=float)
    A = phi @ np.linalg.inv(cl.Lvecs)
    for (i, j), n in zip(cl.bonds, cl.bond_wrap):
        i = int(i)
        j = int(j)
        op.add_two_body(qed.OP_SZ, i, qed.OP_SZ, j, complex(1.0, 0.0))
        d_ij = cl.positions[j] - cl.positions[i] - np.asarray(n) @ cl.Lvecs
        if twist_kind == "physical":
            # S_i^+ S_j^- raises i and lowers j.  The phase is the usual
            # Peierls phase for moving spin from j to i in a uniform field.
            phase = np.exp(-1j * float(A @ d_ij))
        elif twist_kind == "dipole4":
            # For S_i^+ S_j^-, the transported dipole increment is
            # delta = r_i - r_j + n L = -d_ij.  We want the path character
            # exp(-i theta . delta4), with delta4 = 4 delta, hence
            # exp(+i theta . 4 d_ij) on this microscopic term.
            d4 = np.rint(4.0 * d_ij).astype(int)
            phase = np.exp(1j * float(phi @ d4))
        elif twist_kind == "dipole2":
            # Effective ice-manifold rows through order Jpm^3 have
            # 2*delta in Z^3, so this is the minimal row-level character.
            # Individual microscopic hops need not have integer 2*d_ij.
            phase = np.exp(1j * float(phi @ (2.0 * d_ij)))
        else:
            raise ValueError(twist_kind)
        coeff = complex(-jpm * phase)
        op.add_two_body(qed.OP_SPLUS, i, qed.OP_SMINUS, j, coeff)
        op.add_two_body(qed.OP_SMINUS, i, qed.OP_SPLUS, j, np.conjugate(coeff))
    return op


def h5_vector_to_complex(ds) -> np.ndarray:
    arr = ds[()]
    if arr.dtype.fields and "real" in arr.dtype.fields and "imag" in arr.dtype.fields:
        return np.asarray(arr["real"] + 1j * arr["imag"], dtype=np.complex128)
    return np.asarray(arr, dtype=np.complex128)


def solve_low_vectors_qed(cl: R.Cluster, jpm: float, phi: np.ndarray, args):
    op = build_qed_operator(cl, jpm, phi, args.twist_kind)
    with tempfile.TemporaryDirectory() as tmp:
        t0 = time.time()
        res = qed.solve(
            op,
            sz=cl.n_sites // 2,
            auto_sz=False,
            symmetry=None,
            spin_flip="off",
            time_reversal="off",
            num_eigenvalues=args.n_band,
            compute_eigenvectors=True,
            output_dir=tmp,
            solver=args.solver,
            device="cpu",
            tolerance=args.tolerance,
            max_iterations=args.max_iterations,
        )
        elapsed = time.time() - t0
        evals = np.asarray(res.eigenvalues, dtype=float)
        path = Path(res.eigenvectors_path)
        with h5py.File(path, "r") as h5:
            vecs = [
                h5_vector_to_complex(h5[f"eigendata/eigenvector_{k}"])
                for k in range(len(evals))
            ]
    psi = np.column_stack(vecs)
    order = np.argsort(evals)
    return evals[order], psi[:, order], elapsed


def invsqrt_hermitian(a: np.ndarray, cutoff: float = 1e-12):
    vals, vecs = np.linalg.eigh(0.5 * (a + a.conj().T))
    if np.min(vals) <= cutoff:
        raise RuntimeError(f"singular Gram matrix: min eigenvalue {np.min(vals)}")
    return vals, vecs @ np.diag(1.0 / np.sqrt(vals)) @ vecs.conj().T


def band_operator_from_full_eigenvectors(evals, psi, ice_indices):
    """Orthonormalize the full low band, then reconstruct H in the ice basis."""
    d = np.diag(evals)
    full_gram = psi.conj().T @ psi
    full_s_eval, full_invsqrt = invsqrt_hermitian(full_gram)

    # If the saved Ritz vectors are not perfectly orthonormal, use the
    # generalized low-subspace matrix implied by H |psi_i> ~= E_i |psi_i>.
    m = 0.5 * (full_gram @ d + d @ full_gram)
    h_orth = full_invsqrt @ m @ full_invsqrt
    h_orth = 0.5 * (h_orth + h_orth.conj().T)

    x = psi[ice_indices, :] @ full_invsqrt
    gram = x.conj().T @ x
    s_eval, u = np.linalg.eigh(0.5 * (gram + gram.conj().T))
    if np.min(s_eval) <= 1e-12:
        raise RuntimeError(f"projected band is singular: min eigenvalue {np.min(s_eval)}")
    inv_half = u @ np.diag(1.0 / np.sqrt(s_eval)) @ u.conj().T
    q = x @ inv_half
    h_band = q @ h_orth @ q.conj().T
    h_band = 0.5 * (h_band + h_band.conj().T)
    return h_band, {
        "full_vector_gram_min": float(np.min(full_s_eval)),
        "full_vector_gram_max": float(np.max(full_s_eval)),
        "ice_overlap_min": float(np.min(s_eval)),
        "ice_overlap_mean": float(np.mean(s_eval)),
        "ice_overlap_max": float(np.max(s_eval)),
    }


def twist_grid(mode: str, n_grid: int):
    if mode == "one":
        return [(0.0, 0.0, 0.0)]
    if n_grid < 2:
        raise ValueError("--n-grid must be at least 2")
    points = [2.0 * np.pi * m / n_grid for m in range(n_grid)]
    return list(product(points, repeat=3))


def heat_from_matrix(h, temps):
    evals = np.linalg.eigvalsh(h)
    return evals, R.specific_heat(evals, temps)


def centered_frobenius(a, b):
    da = a - np.eye(a.shape[0]) * (np.trace(a) / a.shape[0])
    db = b - np.eye(b.shape[0]) * (np.trace(b) / b.shape[0])
    return float(np.linalg.norm(da - db) / max(np.linalg.norm(db), 1e-15))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jpm", type=float, default=-0.05)
    ap.add_argument("--n-band", type=int, default=90)
    ap.add_argument("--grid", choices=["one", "uniform"], default="uniform")
    ap.add_argument("--n-grid", type=int, default=2,
                    help="uniform grid has phi=2*pi*m/n_grid; n_grid=2 gives {0,pi}")
    ap.add_argument("--twist-kind", choices=["physical", "dipole2", "dipole4"], default="physical",
                    help="physical flux or exact transported-dipole character field")
    ap.add_argument("--solver", choices=["block_lanczos", "krylov_schur", "lanczos", "full"],
                    default="full")
    ap.add_argument("--tolerance", type=float, default=1e-10)
    ap.add_argument("--max-iterations", type=int, default=800)
    ap.add_argument("--out", type=Path, default=HERE / "twist_resolved_qed_full_band_jm0p05.npz")
    args = ap.parse_args()

    cl = R.build_cluster("cubic", (1, 1, 1))
    if args.n_band > cl.n_ice:
        raise ValueError(f"--n-band={args.n_band} exceeds ice dimension {cl.n_ice}")
    sz_basis = fixed_sz_basis(cl.n_sites, cl.n_sites // 2)
    sz_index = {int(s): k for k, s in enumerate(sz_basis)}
    ice_indices = np.asarray([sz_index[int(s)] for s in cl.ice_states], dtype=np.int64)
    twists = twist_grid(args.grid, args.n_grid)

    print(
        f"QED full-Hamiltonian twist run: N={cl.n_sites}, "
        f"Sz dim={len(sz_basis)}, ice={cl.n_ice}, n_band={args.n_band}",
        flush=True,
    )
    print(f"Jpm={args.jpm:+.6f}, grid={args.grid}, n_twists={len(twists)}", flush=True)
    print(f"twist_kind={args.twist_kind}", flush=True)

    h_bands = []
    low_evals = []
    diagnostics = []
    for k, phi in enumerate(twists, start=1):
        phi_arr = np.asarray(phi, dtype=float)
        print(
            f"[{k}/{len(twists)}] phi/pi="
            f"{tuple(round(float(x / np.pi), 6) for x in phi_arr)}",
            flush=True,
        )
        evals, psi, elapsed = solve_low_vectors_qed(cl, args.jpm, phi_arr, args)
        h_band, diag = band_operator_from_full_eigenvectors(evals, psi, ice_indices)
        diag.update({
            "phi": [float(x) for x in phi_arr],
            "elapsed_s": float(elapsed),
            "E0": float(evals[0]),
            "Etop": float(evals[-1]),
        })
        h_bands.append(h_band)
        low_evals.append(evals)
        diagnostics.append(diag)
        print(
            f"    QED {elapsed:.1f}s, E0={evals[0]:.10f}, Etop={evals[-1]:.10f}, "
            f"ice overlap min={diag['ice_overlap_min']:.6f}",
            flush=True,
        )

    h_phi0 = h_bands[0]
    h_avg = sum(h_bands) / len(h_bands)
    temps = np.geomspace(1e-4, 0.12, 900)
    e_phi0, c_phi0 = heat_from_matrix(h_phi0, temps)
    e_avg, c_avg = heat_from_matrix(h_avg, temps)

    print("building perturbative reference", flush=True)
    pt = R.sw_order23(cl, verbose=False)
    h_pt_all = R.assemble(cl, pt, args.jpm, "all")
    h_pt_clean = R.assemble(cl, pt, args.jpm, "delta0")
    e_pt_all, c_pt_all = heat_from_matrix(h_pt_all, temps)
    e_pt_clean, c_pt_clean = heat_from_matrix(h_pt_clean, temps)

    summary = {
        "method": "QED full microscopic Hamiltonian, fixed Sz",
        "jpm": args.jpm,
        "n_sites": cl.n_sites,
        "sz_dim": int(len(sz_basis)),
        "ice_dim": int(cl.n_ice),
        "n_band": int(args.n_band),
        "grid": args.grid,
        "n_grid": int(args.n_grid),
        "n_twists": int(len(twists)),
        "twist_kind": args.twist_kind,
        "solver": args.solver,
        "tolerance": args.tolerance,
        "max_iterations": args.max_iterations,
        "diagnostics": diagnostics,
        "Tpk_qed_phi0": R.refined_peak(temps, c_phi0),
        "Tpk_qed_twist_operator_avg": R.refined_peak(temps, c_avg),
        "Tpk_pt_all": R.refined_peak(temps, c_pt_all),
        "Tpk_pt_delta0": R.refined_peak(temps, c_pt_clean),
        "g4": 4.0 * args.jpm * args.jpm,
        "ghex": 12.0 * abs(args.jpm) ** 3,
        "fro_qed_avg_vs_pt_all_centered": centered_frobenius(h_avg, h_pt_all),
        "fro_qed_avg_vs_pt_delta0_centered": centered_frobenius(h_avg, h_pt_clean),
        "fro_qed_phi0_vs_pt_all_centered": centered_frobenius(h_phi0, h_pt_all),
        "fro_qed_phi0_vs_pt_delta0_centered": centered_frobenius(h_phi0, h_pt_clean),
    }
    print(json.dumps(summary, indent=2))

    np.savez_compressed(
        args.out,
        T=temps,
        H_qed_phi0=h_phi0,
        H_qed_twist_avg=h_avg,
        E_qed_phi0=e_phi0,
        C_qed_phi0=c_phi0,
        E_qed_twist_avg=e_avg,
        C_qed_twist_avg=c_avg,
        H_pt_all=h_pt_all,
        H_pt_delta0=h_pt_clean,
        E_pt_all=e_pt_all,
        C_pt_all=c_pt_all,
        E_pt_delta0=e_pt_clean,
        C_pt_delta0=c_pt_clean,
        E_low_by_twist=np.asarray(low_evals),
        summary=json.dumps(summary),
    )
    args.out.with_suffix(".json").write_text(json.dumps(summary, indent=2))

    fig, ax = plt.subplots(figsize=(7.0, 4.3))
    ax.plot(temps, c_phi0 / cl.n_sites, color="#e67e22", lw=2.0,
            label="QED full band, $\\phi=0$")
    ax.plot(temps, c_avg / cl.n_sites, color="#27ae60", lw=2.0,
            label="QED full band, twist-operator avg.")
    ax.plot(temps, c_pt_all / cl.n_sites, color="#e67e22", ls=":", lw=1.4,
            label="SW ice matrix, all rows")
    ax.plot(temps, c_pt_clean / cl.n_sites, color="#27ae60", ls=":", lw=1.4,
            label="SW ice matrix, $\\delta=0$")
    ax.axvline(summary["g4"], color="black", ls="-.", lw=0.9, label="$g_4$")
    ax.axvline(summary["ghex"], color="#8e44ad", ls="--", lw=0.9, label="$g_{\\rm hex}$")
    ax.set_xscale("log")
    ax.set_xlabel("$T/J_{zz}$")
    ax.set_ylabel("$C(T)/N$")
    ax.set_title(f"QED full-Hamiltonian twist extraction, $J_\\pm={args.jpm:+.2f}$")
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    for ext in ("pdf", "png"):
        fig.savefig(FIGS / f"fig_twist_resolved_qed_full_band.{ext}",
                    bbox_inches="tight", dpi=220)
    plt.close(fig)
    print(f"wrote {args.out}")
    print(f"wrote {args.out.with_suffix('.json')}")
    print(f"wrote {FIGS / 'fig_twist_resolved_qed_full_band.pdf'}")


if __name__ == "__main__":
    main()
