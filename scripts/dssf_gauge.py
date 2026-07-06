"""Emergent-photon dynamical structure factor S^{zz}(q,omega) of quantum
spin ice from the transport-clean effective Hamiltonian.

The gauge (photon) sector of the DSSF lives entirely in the ice manifold:
S^z is diagonal in the ice-configuration basis (it is the emergent electric
field), so
    S^{zz}(q,omega) = sum_{n} |<n| S^z_q |0>|^2  delta(omega - (E_n - E_0)),
    S^z_q = sum_i e^{i q.r_i} S^z_i,
with |0>, |n> the eigenstates of the effective ring Hamiltonian H_eff in the
ice manifold. This is the low-omega part of the neutron cross-section that
the spurious four-site ring exchange contaminates; the zero-transport
projector ('delta0') removes that contamination (see SIMULATION_PLAN.md).

The spinon continuum (S^{+-} channel, omega ~ Jzz) is NOT computed here --
it is uncontaminated and is obtained from full-Hilbert-space FTLM.

Both flux signs and every coupling are served by one J±=1 row-table build
(H_k ~ J±^k exactly). On the 2×2×2 FCC cluster (ice = 2970) the full
spectrum is dense-diagonalizable, so S^{zz}(q,omega) is numerically exact.

Usage
-----
  python dssf_gauge.py --rows rows_fcc222_o4.npz --jpm -0.05 0.046 \
        --modes all delta0 --T 0.0 --out data/dssf_gauge.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import ice_pt_lib as ipl


def cluster_momenta(cl):
    """Reciprocal-lattice points of the cluster (the momenta the finite
    torus can resolve): q = 2*pi * (Lvecs^{-T}) m, m in the fundamental
    domain of integer triples. Returns (labels, qvecs)."""
    Linv_T = np.linalg.inv(cl.Lvecs).T
    L = np.rint(np.abs(cl.Lvecs)).astype(int)  # multiplicities per axis (FCC: shape)
    ms = []
    rng = [range(0, max(2, s)) for s in cl.shape]
    for m in np.ndindex(*[len(r) for r in rng]):
        ms.append(np.array([list(rng[a])[m[a]] for a in range(3)]))
    qs = [2 * np.pi * (Linv_T @ m) for m in ms]
    labels = ["(" + ",".join(str(int(x)) for x in m) + ")" for m in ms]
    return labels, np.array(qs), np.array(ms)


def szq_diag(cl, q):
    """Diagonal of S^z_q in the ice-config basis: (S^z_q)_cc =
    sum_i e^{i q.r_i} (n_i(c) - 1/2). Complex vector of length n_ice."""
    ice = cl.ice_states
    N = cl.n_sites
    bits = ((ice[:, None] >> np.arange(N, dtype=np.uint64)[None, :])
            & np.uint64(1)).astype(float) - 0.5
    phase = np.exp(1j * (cl.positions @ q))          # (N,)
    return bits @ phase                               # (n_ice,)


def dssf(cl, H, q, omega, eta, T=0.0):
    """S^{zz}(q,omega) with Lorentzian broadening eta. T=0 uses the ground
    state; T>0 sums over a Boltzmann-weighted initial ensemble."""
    E, V = np.linalg.eigh(H)
    E = E - E[0]
    sq = szq_diag(cl, q)                               # diagonal operator
    # matrix elements M[m,n] = <m| S^z_q |n> = sum_c conj(V[c,m]) sq[c] V[c,n]
    SV = sq[:, None] * V                               # (n_ice, n_ice)
    M = V.conj().T @ SV                                # (m,n)
    if T <= 0:
        w0 = np.zeros(len(E)); w0[0] = 1.0
    else:
        b = np.exp(-E / T); w0 = b / b.sum()
    S = np.zeros_like(omega)
    for n in np.nonzero(w0 > 1e-12)[0]:
        dw = E - E[n]                                  # excitation energies
        amp = np.abs(M[:, n]) ** 2 * w0[n]             # (n_ice,)
        x = omega[:, None] - dw[None, :]               # (nw, n_ice)
        lor = (eta / np.pi) / (x ** 2 + eta ** 2)
        S += (lor * amp[None, :]).sum(axis=1)
    return S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True)
    ap.add_argument("--jpm", type=float, nargs="+", required=True)
    ap.add_argument("--modes", nargs="+", default=["all", "delta0"])
    ap.add_argument("--order", type=int, default=None)
    ap.add_argument("--T", type=float, default=0.0)
    ap.add_argument("--wmax-over-ghex", type=float, default=40.0)
    ap.add_argument("--nw", type=int, default=600)
    ap.add_argument("--eta-over-ghex", type=float, default=0.3)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    pt, meta = ipl.load_rows(args.rows)
    cl = ipl.build_cluster(meta["basis"], meta["shape"])
    order = args.order or meta["order"]
    labels, qs, ms = cluster_momenta(cl)

    out = {"meta": meta, "order": order, "T": args.T,
           "q_labels": labels, "q_vecs": qs.tolist(), "curves": []}
    for J in args.jpm:
        ghex = 12 * abs(J) ** 3
        omega = np.linspace(0, args.wmax_over_ghex * ghex, args.nw)
        eta = args.eta_over_ghex * ghex
        for mode in args.modes:
            H = np.real(ipl.assemble(cl, pt, J, order, mode=mode))
            Sqw = np.array([dssf(cl, H, q, omega, eta, T=args.T) for q in qs])
            # q-integrated spectral function and its low-omega centroid
            Sw = Sqw.sum(axis=0)
            m = omega > 1e-9
            centroid = float(np.sum(omega[m] * Sw[m]) / np.sum(Sw[m]))
            out["curves"].append(dict(
                Jpm=J, mode=mode, ghex=ghex, g4=4 * J ** 2,
                omega=omega.tolist(), S_of_omega=Sw.tolist(),
                Sqw=Sqw.tolist(), centroid=centroid,
                centroid_over_ghex=centroid / ghex,
                centroid_over_g4=centroid / (4 * J ** 2)))
            print(f"Jpm={J:+.3f} {mode:>7}: S^zz centroid={centroid:.5f} "
                  f"= {centroid/(4*J**2):.2f} g4 = {centroid/ghex:.1f} ghex",
                  flush=True)
    Path(args.out).write_text(json.dumps(out))
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
