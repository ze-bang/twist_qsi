"""
Pure-Python computation of the T=0 SzSz dynamical structure factor on
the 16-site (1,1,1) cubic pyrochlore cluster.

Strategy (memory-light, scales to ~60k-dim full Hilbert spaces too):

  1.  Read InterAll.dat / Trans.dat / positions.dat from the existing
      twist run directories under output/demo/phi_*/ham/ -- these were
      written by run_demo.py with the same U(1) twist phases that
      generated the FTLM thermodynamics curves.
  2.  Build the Hamiltonian as a complex sparse matrix on the
      Sz_tot=0 subspace (dim = 12,870 for 16 sites at half filling).
      This used to be a hot Python loop; we now build with a
      vectorised generator for the dominant two-body part.
  3.  Find the ground state |0> with sparse Lanczos (eigsh, k=1).
  4.  For each cluster-allowed momentum Q, build the diagonal of
      S^z_Q = sum_i e^{-i Q . r_i} S^z_i (Sz preserves Sz_tot, so the
      operator is diagonal in our occupation-number basis), construct
      |b_0> = S^z_Q |0>, and run the continued-fraction Lanczos
      starting from |b_0>/||b_0|| against the same sparse H.
  5.  Convert the Lanczos tridiagonal to the spectral function via
      the standard continued-fraction expression
            G(z) = 1 / (z - a_0 - b_1^2 / (z - a_1 - ...))
      and S^{zz}(Q, omega) = -mu_0 / pi * Im G(omega + E_0 + i eta),
      with mu_0 = ||b_0||^2.

This bypasses the C++ dssf engine entirely (which silently produced
~1e-22 weight on this cluster) and verifies the bare/twist-averaged
SzSz DSSF directly.
"""
from __future__ import annotations

import argparse
import json
import time
from itertools import product
from pathlib import Path

import h5py
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Hamiltonian I/O
# ---------------------------------------------------------------------------
def parse_interall(path: Path) -> np.ndarray:
    rows: list = []
    with open(path) as f:
        for line in f:
            ln = line.strip()
            if not ln or ln.startswith("="):
                continue
            parts = ln.split()
            if len(parts) < 6:
                continue
            try:
                vals = [float(p) for p in parts[:6]]
                rows.append(vals)
            except ValueError:
                continue
    return np.asarray(rows)


def parse_trans(path: Path) -> np.ndarray:
    rows: list = []
    with open(path) as f:
        for line in f:
            ln = line.strip()
            if not ln or ln.startswith("="):
                continue
            parts = ln.split()
            if len(parts) < 4:
                continue
            try:
                vals = [float(p) for p in parts[:4]]
                rows.append(vals)
            except ValueError:
                continue
    return np.asarray(rows)


def parse_positions(path: Path):
    pos = {}
    with open(path) as f:
        for line in f:
            ln = line.strip()
            if not ln or ln.startswith("#"):
                continue
            parts = ln.split()
            site = int(parts[0])
            x, y, z = float(parts[3]), float(parts[4]), float(parts[5])
            pos[site] = (x, y, z)
    return pos


# ---------------------------------------------------------------------------
# Sz=0 basis
# ---------------------------------------------------------------------------
def fixed_sz_basis(n_sites: int, n_up: int):
    states: list = []
    for k in range(1 << n_sites):
        if bin(k).count("1") == n_up:
            states.append(k)
    states = np.asarray(states, dtype=np.int64)
    idx = {int(s): i for i, s in enumerate(states)}
    return states, idx


# ---------------------------------------------------------------------------
# H assembly
# ---------------------------------------------------------------------------
def _bit_signs(states: np.ndarray, site: int) -> np.ndarray:
    """Return +1/2 or -1/2 per state for site `site`."""
    return np.where(((states >> site) & 1) == 1, 0.5, -0.5)


def build_hamiltonian(interall, trans, n_sites, states, idx):
    """Build the Sz=0 sparse Hamiltonian."""
    dim = len(states)
    rows: list = []
    cols: list = []
    vals: list = []

    if trans.size:
        for r in trans:
            op, s_i, t_re, t_im = int(r[0]), int(r[1]), r[2], r[3]
            coef = t_re + 1j * t_im
            bit_i = 1 << s_i
            if op == 2:
                diag = coef * _bit_signs(states, s_i)
                rows.extend(range(dim))
                cols.extend(range(dim))
                vals.extend(diag.tolist())
            else:
                up_i = ((states >> s_i) & 1) == 1
                if op == 0:  # S+
                    src_mask = ~up_i
                    sign = 1.0
                else:  # S-
                    src_mask = up_i
                    sign = 1.0
                src_states = states[src_mask]
                if op == 0:
                    new = src_states | bit_i
                else:
                    new = src_states & ~bit_i
                src_idx = np.flatnonzero(src_mask)
                for ns, si in zip(new, src_idx):
                    j = idx.get(int(ns))
                    if j is not None:
                        rows.append(j); cols.append(si)
                        vals.append(coef * sign)

    for r in interall:
        op_i, s_i, op_j, s_j, J_re, J_im = (
            int(r[0]), int(r[1]), int(r[2]), int(r[3]), r[4], r[5]
        )
        coef0 = J_re + 1j * J_im
        bit_i = 1 << s_i
        bit_j = 1 << s_j

        # Decompose action on each state vectorised by op_i then op_j.
        #   keep an array (sign, new_state) per state, trim to nonzero,
        #   then look up new->idx.
        # Stage 1: apply op_i.
        if op_i == 2:
            sgn1 = _bit_signs(states, s_i)
            new1 = states.copy()
            keep1 = np.ones(len(states), dtype=bool)
        elif op_i == 0:  # S+: needs site i empty
            keep1 = ((states >> s_i) & 1) == 0
            sgn1 = np.where(keep1, 1.0, 0.0)
            new1 = np.where(keep1, states | bit_i, states)
        else:  # S-: needs site i occupied
            keep1 = ((states >> s_i) & 1) == 1
            sgn1 = np.where(keep1, 1.0, 0.0)
            new1 = np.where(keep1, states & ~bit_i, states)

        if not keep1.any():
            continue

        # Stage 2: apply op_j to new1.
        if op_j == 2:
            up_j = ((new1 >> s_j) & 1) == 1
            sgn2 = np.where(up_j, 0.5, -0.5)
            new2 = new1
            keep2 = keep1
        elif op_j == 0:
            mask = ((new1 >> s_j) & 1) == 0
            keep2 = keep1 & mask
            sgn2 = np.where(keep2, 1.0, 0.0)
            new2 = np.where(mask, new1 | bit_j, new1)
        else:
            mask = ((new1 >> s_j) & 1) == 1
            keep2 = keep1 & mask
            sgn2 = np.where(keep2, 1.0, 0.0)
            new2 = np.where(mask, new1 & ~bit_j, new1)

        if not keep2.any():
            continue

        # Diagonal contributions (op_i=op_j=2) and off-diagonal: lookup
        keep_idx = np.flatnonzero(keep2)
        new_states = new2[keep_idx]
        signs = sgn1[keep_idx] * sgn2[keep_idx]
        # Map new_states to indices via idx dict (vectorised via np.searchsorted)
        # since `states` is sorted, we can use searchsorted:
        positions_in_basis = np.searchsorted(states, new_states)
        # validate hits
        valid = (positions_in_basis < dim) & (states[
            np.minimum(positions_in_basis, dim - 1)] == new_states)
        keep_idx = keep_idx[valid]
        positions_in_basis = positions_in_basis[valid]
        signs = signs[valid]

        rows.extend(positions_in_basis.tolist())
        cols.extend(keep_idx.tolist())
        vals.extend((coef0 * signs).tolist())

    H = sp.coo_matrix((vals, (rows, cols)), shape=(dim, dim), dtype=complex)
    return H.tocsr()


# ---------------------------------------------------------------------------
# Sz_q operator (diagonal in occupation basis)
# ---------------------------------------------------------------------------
def sz_q_diagonal(states, n_sites, q_vec, positions):
    phases = np.array(
        [np.exp(-1j * (q_vec[0] * positions[i][0]
                       + q_vec[1] * positions[i][1]
                       + q_vec[2] * positions[i][2]))
         for i in range(n_sites)],
        dtype=complex,
    )
    diag = np.zeros(len(states), dtype=complex)
    for i in range(n_sites):
        diag += phases[i] * _bit_signs(states, i)
    return diag


# ---------------------------------------------------------------------------
# Continued-fraction Lanczos for the spectral function
# ---------------------------------------------------------------------------
def cf_lanczos(H, b0, M):
    """Run M+1 Lanczos steps starting from b0 (not necessarily normalised).
    Returns (mu0, alpha[0..M], beta[1..M]) defining the continued
    fraction <b0|(z - H)^{-1}|b0> = mu0 / (z - a0 - b1^2/(z - a1 - ...)).
    """
    mu0 = float(np.vdot(b0, b0).real)
    if mu0 < 1e-30:
        return 0.0, np.zeros(0), np.zeros(0)

    f_prev = np.zeros_like(b0)
    f_curr = b0 / np.sqrt(mu0)
    alpha = np.zeros(M + 1)
    beta = np.zeros(M + 1)  # beta[0] unused

    Hf = H @ f_curr
    a0 = complex(np.vdot(f_curr, Hf))
    alpha[0] = a0.real
    r = Hf - alpha[0] * f_curr
    for n in range(1, M + 1):
        b = float(np.linalg.norm(r))
        beta[n] = b
        if b < 1e-12:
            return mu0, alpha[:n], beta[1:n]
        f_prev, f_curr = f_curr, r / b
        Hf = H @ f_curr
        a = complex(np.vdot(f_curr, Hf))
        alpha[n] = a.real
        r = Hf - alpha[n] * f_curr - b * f_prev
    return mu0, alpha, beta[1:]


def cf_evaluate(z, alpha, beta):
    """Evaluate the continued fraction
        1 / (z - a_0 - b_1^2 / (z - a_1 - b_2^2 / (...)))
    at scalar (or array) z.
    """
    z = np.asarray(z, dtype=complex)
    M = len(alpha) - 1
    cf = z - alpha[M]
    for n in range(M - 1, -1, -1):
        cf = z - alpha[n] - (beta[n] ** 2) / cf
    return 1.0 / cf


# ---------------------------------------------------------------------------
# Top-level driver
# ---------------------------------------------------------------------------
def twist_label(phi):
    return "phi_" + "_".join(
        ("pi" if abs(p - np.pi) < 1e-8 else f"{p/np.pi:.3f}pi") for p in phi
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo-root", default=str(ROOT / "output" / "demo"))
    ap.add_argument("--out", default=str(ROOT / "output" / "dssf_py"))
    ap.add_argument("--n-sites", type=int, default=16)
    ap.add_argument("--n-up", type=int, default=8)
    ap.add_argument("--omega-min", type=float, default=-0.05)
    ap.add_argument("--omega-max", type=float, default=3.0)
    ap.add_argument("--n-omega", type=int, default=1200)
    ap.add_argument("--eta", type=float, default=0.012)
    ap.add_argument("--lanczos-steps", type=int, default=400)
    args = ap.parse_args()

    demo_root = Path(args.demo_root)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    states, idx = fixed_sz_basis(args.n_sites, args.n_up)
    print(f"Sz=0 basis dim = {len(states)}")

    omega = np.linspace(args.omega_min, args.omega_max, args.n_omega)

    Q_list = {
        "Gamma": np.array([0.0, 0.0, 0.0]),
        "X1":    np.array([2 * np.pi, 0.0, 0.0]),
        "X2":    np.array([0.0, 2 * np.pi, 0.0]),
        "X3":    np.array([0.0, 0.0, 2 * np.pi]),
        "L":     np.array([np.pi, np.pi, np.pi]),
    }

    twists = list(product([0.0, np.pi], repeat=3))

    summary = {
        "n_sites": args.n_sites,
        "dim_Sz0": int(len(states)),
        "omega_min": args.omega_min,
        "omega_max": args.omega_max,
        "n_omega": args.n_omega,
        "eta": args.eta,
        "lanczos_steps": args.lanczos_steps,
        "Q_points": {k: v.tolist() for k, v in Q_list.items()},
        "twists": [],
    }

    for k, phi in enumerate(twists):
        tag = twist_label(phi)
        ham_dir = demo_root / tag / "ham"
        out_dir = out_root / tag
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n[{k+1}/{len(twists)}] twist = {tuple(phi)}  ({tag})")
        t0 = time.time()
        interall = parse_interall(ham_dir / "InterAll.dat")
        trans = parse_trans(ham_dir / "Trans.dat")
        positions = parse_positions(ham_dir / "positions.dat")
        H = build_hamiltonian(interall, trans, args.n_sites, states, idx)
        H = 0.5 * (H + H.getH())
        print(f"  built H in {time.time()-t0:.1f}s, nnz={H.nnz}")

        t1 = time.time()
        # Sparse GS via Lanczos.
        e0, v0 = spla.eigsh(H, k=1, which="SA", maxiter=2000, tol=1e-10)
        E0 = float(e0[0])
        psi0 = v0[:, 0]
        print(f"  GS via eigsh in {time.time()-t1:.1f}s, E0={E0:.6f}")

        S_qw = {}
        sum_sw = {}
        for label, qv in Q_list.items():
            sz_diag = sz_q_diagonal(states, args.n_sites, qv, positions)
            b0 = sz_diag * psi0
            mu0 = float(np.vdot(b0, b0).real)
            sum_sw[label] = mu0
            if mu0 < 1e-14:
                S_qw[label] = np.zeros_like(omega)
                continue
            t2 = time.time()
            mu0_, alpha, beta = cf_lanczos(H, b0, args.lanczos_steps)
            z = (omega + 1j * args.eta) + E0
            G = cf_evaluate(z, alpha, beta)
            spec = -(mu0_ / np.pi) * G.imag
            S_qw[label] = spec
            print(f"  {label}: |b0|^2={mu0:.3g}, "
                  f"Lanczos {len(alpha)} steps in {time.time()-t2:.1f}s, "
                  f"max S(Q,om)={spec.max():.3f} at "
                  f"omega={omega[np.argmax(spec)]:.4f}")

        np.savez_compressed(
            out_dir / "dssf.npz",
            omega=omega,
            E0=E0,
            **{f"S_{k}": v for k, v in S_qw.items()},
            **{f"sum_{k}": np.array([v]) for k, v in sum_sw.items()},
        )

        rec = {"phi": list(map(float, phi)), "tag": tag,
               "npz": str(out_dir / "dssf.npz"),
               "E0": E0,
               "static_sum_rule": {k: float(v) for k, v in sum_sw.items()},
               "elapsed_s": float(time.time() - t0)}
        summary["twists"].append(rec)
        with open(out_root / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
