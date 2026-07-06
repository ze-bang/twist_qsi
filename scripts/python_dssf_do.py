"""
T=0 polarized-neutron dynamical structure factor on the 16-site
(1,1,1) cubic pyrochlore cluster, in the *dipolar-octupolar (DO) QSI*
convention.

DO basis convention
-------------------
At each site i on sublattice mu in {0,1,2,3} the magnetic moment is

    M_i = g_z * S~^z_i * z_mu + g_y * S~^y_i * y_mu

with S~^z, S~^y the dipolar pseudospin components and S~^x the
octupolar one (which does NOT contribute to neutron scattering).
Local-frame axes follow the Curnoe / Onoda 2011 convention:

    z_0 = (+1,+1,+1)/sqrt 3,    x_0 = (-2,+1,+1)/sqrt 6,    y_0 = z_0 x x_0
    z_1 = (+1,-1,-1)/sqrt 3,    x_1 = (-2,-1,-1)/sqrt 6,    y_1 = z_1 x x_1
    z_2 = (-1,+1,-1)/sqrt 3,    x_2 = (+2,+1,-1)/sqrt 6,    y_2 = z_2 x x_2
    z_3 = (-1,-1,+1)/sqrt 3,    x_3 = (+2,-1,+1)/sqrt 6,    y_3 = z_3 x x_3

For the demo we use g_z = g_y = 1.

Polarized-neutron NSF/SF decomposition
--------------------------------------
For each q, define

    n_hat = projection of [1,-1,0]/sqrt 2 perpendicular to q_hat, normalised
            (the standard vertical polarisation analyser axis)
    n2_hat = q_hat x n_hat                           (in-plane perpendicular)

The NSF and SF cross sections are

    S_NSF(q,omega) = sum_n |<n| M_{n_hat}(q) |0>|^2  delta(omega - (E_n - E_0))
    S_SF (q,omega) = sum_n |<n| M_{n2_hat}(q) |0>|^2 delta(omega - (E_n - E_0))

with M_v(q) = sum_i e^{-i q.r_i} (v . M_i).  Both involve S~^y, so the
operator does NOT preserve S^z_tot:  M_v(q)|0> has weight in
S^z_tot = -1, 0, +1.  We build H in those three blocks and run
continued-fraction Lanczos in each block separately, then sum.

Output
------
Writes  output/dssf_do/<twist_tag>/dssf_do.npz  with arrays:
    omega, E0, S_<channel>_<qlabel>  (channel in {NSF, SF}),
    sum_<channel>_<qlabel>           (static k=0 moment of the spectrum),
    n_<qlabel>, n2_<qlabel>          (the two polarisation axes used).
"""
from __future__ import annotations

import argparse
import json
import time
from itertools import product
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from python_dssf import (
    parse_interall, parse_trans, parse_positions,
    fixed_sz_basis, build_hamiltonian,
    cf_lanczos, cf_evaluate,
)

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# DO local frames
# ---------------------------------------------------------------------------
def do_local_frame(mu: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (z, x, y) unit vectors of the local DO frame for sublattice mu."""
    if mu == 0:
        z = np.array([+1.0, +1.0, +1.0]) / np.sqrt(3.0)
        x = np.array([-2.0, +1.0, +1.0]) / np.sqrt(6.0)
    elif mu == 1:
        z = np.array([+1.0, -1.0, -1.0]) / np.sqrt(3.0)
        x = np.array([-2.0, -1.0, -1.0]) / np.sqrt(6.0)
    elif mu == 2:
        z = np.array([-1.0, +1.0, -1.0]) / np.sqrt(3.0)
        x = np.array([+2.0, +1.0, -1.0]) / np.sqrt(6.0)
    elif mu == 3:
        z = np.array([-1.0, -1.0, +1.0]) / np.sqrt(3.0)
        x = np.array([+2.0, -1.0, +1.0]) / np.sqrt(6.0)
    else:
        raise ValueError(f"bad sublattice {mu}")
    y = np.cross(z, x)
    y /= np.linalg.norm(y)
    return z, x, y


def parse_sublat(path: Path) -> dict[int, int]:
    """Read sublattice index per site from positions.dat
    (column 2 = sublattice_index)."""
    sub: dict[int, int] = {}
    with open(path) as f:
        for line in f:
            ln = line.strip()
            if not ln or ln.startswith("#"):
                continue
            toks = ln.split()
            sub[int(toks[0])] = int(toks[2])
    return sub


# ---------------------------------------------------------------------------
# Polarisation choice: project [1,-1,0]/sqrt 2 perpendicular to q, normalise
# ---------------------------------------------------------------------------
N_VERT = np.array([1.0, -1.0, 0.0]) / np.sqrt(2.0)


def polarisation_axes(q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (n_hat, n2_hat) for a given q.

    n_hat = projection of N_VERT = [1,-1,0]/sqrt 2 perpendicular to q,
            normalised. If q = 0 we return (N_VERT, [1,1,0]/sqrt 2).
            If q || N_VERT we fall back to [1,1,-2]/sqrt 6 (perp to N_VERT).
    n2_hat = q_hat x n_hat
    """
    qnorm = float(np.linalg.norm(q))
    if qnorm < 1e-10:
        n = N_VERT.copy()
        n2 = np.array([1.0, 1.0, 0.0]) / np.sqrt(2.0)
        return n, n2
    qhat = q / qnorm

    n = N_VERT - (N_VERT @ qhat) * qhat
    if np.linalg.norm(n) < 1e-8:
        fallback = np.array([1.0, 1.0, -2.0]) / np.sqrt(6.0)
        n = fallback - (fallback @ qhat) * qhat
    n /= np.linalg.norm(n)
    n2 = np.cross(qhat, n)
    n2 /= np.linalg.norm(n2)
    return n, n2


# ---------------------------------------------------------------------------
# Block H on a given S^z_tot sector
# ---------------------------------------------------------------------------
def build_block(interall, trans, n_sites: int, n_up: int):
    states, idx = fixed_sz_basis(n_sites, n_up)
    H = build_hamiltonian(interall, trans, n_sites, states, idx)
    H = 0.5 * (H + H.getH())
    return states, idx, H


# ---------------------------------------------------------------------------
# Build M_v(q) |psi0> for psi0 in S^z_tot = 0 sector.
# v is a real 3-vector (polarisation axis).
# Returns dict of {sector_key: vector}, sector_key in {-1, 0, +1}.
# ---------------------------------------------------------------------------
def Mv_q_on_psi0(psi0_states_0: np.ndarray, idx_0: dict, psi0: np.ndarray,
                 states_p: np.ndarray, idx_p: dict,
                 states_m: np.ndarray, idx_m: dict,
                 q_vec: np.ndarray, v: np.ndarray,
                 positions: dict, sublat: dict, n_sites: int):
    """Apply M_v(q) = sum_i e^{-i q . r_i} [(v.z_mu) S~^z_i + (v.y_mu) S~^y_i]
    to psi0 living in the S^z_tot = 0 sector.

    Returns three vectors b0_0, b0_p, b0_m living in S^z_tot = 0, +1, -1
    sectors respectively.
    """
    dim_0 = len(psi0_states_0)
    dim_p = len(states_p)
    dim_m = len(states_m)

    b0_0 = np.zeros(dim_0, dtype=complex)
    b0_p = np.zeros(dim_p, dtype=complex)
    b0_m = np.zeros(dim_m, dtype=complex)

    for i in range(n_sites):
        ri = np.array(positions[i])
        phase = np.exp(-1j * (q_vec @ ri))
        z_mu, _x_mu, y_mu = do_local_frame(sublat[i])
        c_z = float(v @ z_mu)
        c_y = float(v @ y_mu)

        if abs(c_z) > 1e-14:
            up_i = ((psi0_states_0 >> i) & 1) == 1
            sgn_z = np.where(up_i, 0.5, -0.5)
            b0_0 += (phase * c_z) * sgn_z * psi0

        if abs(c_y) > 1e-14:
            up_i = ((psi0_states_0 >> i) & 1) == 1
            empty_i = ~up_i
            bit_i = np.int64(1) << i

            sp_states = psi0_states_0[empty_i] | bit_i
            sp_amps = psi0[empty_i]
            sp_pos = np.searchsorted(states_p, sp_states)
            valid = (sp_pos < dim_p) & (states_p[
                np.minimum(sp_pos, dim_p - 1)] == sp_states)
            np.add.at(
                b0_p,
                sp_pos[valid],
                (phase * c_y / (2.0j)) * sp_amps[valid],
            )

            sm_states = psi0_states_0[up_i] & ~bit_i
            sm_amps = psi0[up_i]
            sm_pos = np.searchsorted(states_m, sm_states)
            valid = (sm_pos < dim_m) & (states_m[
                np.minimum(sm_pos, dim_m - 1)] == sm_states)
            np.add.at(
                b0_m,
                sm_pos[valid],
                -(phase * c_y / (2.0j)) * sm_amps[valid],
            )

    return b0_0, b0_p, b0_m


# ---------------------------------------------------------------------------
# Twist label
# ---------------------------------------------------------------------------
def twist_label(phi):
    return "phi_" + "_".join(
        ("pi" if abs(p - np.pi) < 1e-8 else f"{p/np.pi:.3f}pi") for p in phi
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo-root", default=str(ROOT / "output" / "demo"))
    ap.add_argument("--out", default=str(ROOT / "output" / "dssf_do"))
    ap.add_argument("--n-sites", type=int, default=16)
    ap.add_argument("--omega-min", type=float, default=-0.05)
    ap.add_argument("--omega-max", type=float, default=3.0)
    ap.add_argument("--n-omega", type=int, default=1200)
    ap.add_argument("--eta", type=float, default=0.012)
    ap.add_argument("--lanczos-steps", type=int, default=400)
    ap.add_argument("--twist-index", type=int, default=-1,
                    help="If >=0, run only that single twist corner index "
                    "(0..7) and write a per-corner result; combine later.")
    ap.add_argument("--aggregate-only", action="store_true",
                    help="Skip computation, just merge per-corner rec.json "
                    "into output summary.json.")
    args = ap.parse_args()

    demo_root = Path(args.demo_root)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    omega = np.linspace(args.omega_min, args.omega_max, args.n_omega)

    Q_list = {
        "Gamma": np.array([0.0, 0.0, 0.0]),
        "X1":    np.array([2 * np.pi, 0.0, 0.0]),
        "X2":    np.array([0.0, 2 * np.pi, 0.0]),
        "X3":    np.array([0.0, 0.0, 2 * np.pi]),
        "L":     np.array([np.pi, np.pi, np.pi]),
    }

    pol_axes = {k: polarisation_axes(v) for k, v in Q_list.items()}

    twists_all = list(product([0.0, np.pi], repeat=3))
    if args.twist_index >= 0:
        twists = [twists_all[args.twist_index]]
        twist_indices = [args.twist_index]
    else:
        twists = twists_all
        twist_indices = list(range(len(twists_all)))

    if args.aggregate_only:
        recs = []
        for phi in twists_all:
            tag = twist_label(phi)
            r_path = out_root / tag / "rec.json"
            if not r_path.exists():
                print(f"  skip missing rec.json: {r_path}")
                continue
            recs.append(json.loads(r_path.read_text()))
        merged = {
            "n_sites": args.n_sites,
            "omega_min": args.omega_min,
            "omega_max": args.omega_max,
            "n_omega": args.n_omega,
            "eta": args.eta,
            "lanczos_steps": args.lanczos_steps,
            "Q_points": {k: v.tolist() for k, v in Q_list.items()},
            "pol_axes": {k: {"n": pol_axes[k][0].tolist(),
                             "n2": pol_axes[k][1].tolist()} for k in Q_list},
            "twists": recs,
        }
        (out_root / "summary.json").write_text(json.dumps(merged, indent=2))
        print(f"merged {len(recs)}/{len(twists_all)} corners into "
              f"{out_root/'summary.json'}")
        return

    summary = {
        "n_sites": args.n_sites,
        "omega_min": args.omega_min,
        "omega_max": args.omega_max,
        "n_omega": args.n_omega,
        "eta": args.eta,
        "lanczos_steps": args.lanczos_steps,
        "Q_points": {k: v.tolist() for k, v in Q_list.items()},
        "pol_axes": {k: {"n": pol_axes[k][0].tolist(),
                         "n2": pol_axes[k][1].tolist()} for k in Q_list},
        "twists": [],
    }

    for k_local, (twist_idx, phi) in enumerate(zip(twist_indices, twists)):
        tag = twist_label(phi)
        ham_dir = demo_root / tag / "ham"
        out_dir = out_root / tag
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n[corner {twist_idx+1}/8] twist = {tuple(phi)}  ({tag})",
              flush=True)
        t0 = time.time()
        interall = parse_interall(ham_dir / "InterAll.dat")
        trans = parse_trans(ham_dir / "Trans.dat")
        positions = parse_positions(ham_dir / "positions.dat")
        sublat = parse_sublat(ham_dir / "positions.dat")

        states_0, idx_0, H_0 = build_block(interall, trans, args.n_sites, 8)
        states_p, idx_p, H_p = build_block(interall, trans, args.n_sites, 9)
        states_m, idx_m, H_m = build_block(interall, trans, args.n_sites, 7)
        print(f"  built H blocks in {time.time()-t0:.1f}s,"
              f" dim(Sz=0)={len(states_0)}, dim(Sz=+/-1)={len(states_p)}")

        t1 = time.time()
        e0, v0 = spla.eigsh(H_0, k=1, which="SA", maxiter=2000, tol=1e-10)
        E0 = float(e0[0])
        psi0 = v0[:, 0]
        print(f"  GS in {time.time()-t1:.1f}s, E0={E0:.6f}")

        S_qw: dict[str, np.ndarray] = {}
        sum_w: dict[str, float] = {}

        for q_label, q_vec in Q_list.items():
            n_hat, n2_hat = pol_axes[q_label]
            for chan, v_pol in (("NSF", n_hat), ("SF", n2_hat)):
                key = f"{chan}_{q_label}"
                t2 = time.time()
                b0_0, b0_p, b0_m = Mv_q_on_psi0(
                    states_0, idx_0, psi0,
                    states_p, idx_p,
                    states_m, idx_m,
                    q_vec, v_pol,
                    positions, sublat, args.n_sites,
                )

                spec_total = np.zeros_like(omega)
                mu_total = 0.0
                for tag_sec, (b0_sec, H_sec, dim_sec) in (
                    ("0",  (b0_0, H_0, len(states_0))),
                    ("+1", (b0_p, H_p, len(states_p))),
                    ("-1", (b0_m, H_m, len(states_m))),
                ):
                    mu0 = float(np.vdot(b0_sec, b0_sec).real)
                    if mu0 < 1e-14 or dim_sec == 0:
                        continue
                    mu0_, alpha, beta = cf_lanczos(
                        H_sec, b0_sec, args.lanczos_steps,
                    )
                    z = (omega + 1j * args.eta) + E0
                    G = cf_evaluate(z, alpha, beta)
                    spec = -(mu0_ / np.pi) * G.imag
                    spec_total += spec
                    mu_total += mu0

                S_qw[key] = spec_total
                sum_w[key] = mu_total
                print(f"  {key:<12s} sum={mu_total:6.4f}, "
                      f"max={spec_total.max():6.3f} at omega="
                      f"{omega[np.argmax(spec_total)]:.4f}, "
                      f"in {time.time()-t2:.1f}s", flush=True)

        np.savez_compressed(
            out_dir / "dssf_do.npz",
            omega=omega,
            E0=E0,
            **{f"S_{k}": v for k, v in S_qw.items()},
            **{f"sum_{k}": np.array([v]) for k, v in sum_w.items()},
            **{f"n_{q}":  pol_axes[q][0] for q in Q_list},
            **{f"n2_{q}": pol_axes[q][1] for q in Q_list},
        )

        rec = {"phi": list(map(float, phi)), "tag": tag,
               "npz": str(out_dir / "dssf_do.npz"),
               "E0": E0,
               "static_sum_rule": {k: float(v) for k, v in sum_w.items()},
               "elapsed_s": float(time.time() - t0)}
        summary["twists"].append(rec)
        # write per-corner record (safe for parallel runs)
        (out_dir / "rec.json").write_text(json.dumps(rec, indent=2))
        if args.twist_index < 0:
            with open(out_root / "summary.json", "w") as f:
                json.dump(summary, f, indent=2)
        print(f"[corner {twist_idx+1}/8] done in {rec['elapsed_s']:.1f}s",
              flush=True)


if __name__ == "__main__":
    main()
