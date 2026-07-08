#!/usr/bin/env python3
"""Fast twist-resolved verification in a full-Hamiltonian Krylov subspace.

The subspace is generated from the ice manifold by repeated applications of
the microscopic nearest-neighbor exchange V.  Depth 2 contains the virtual
states needed for the order-Jpm^2 four-loop and order-Jpm^3 hexagon processes:

    K_2 = span{ |ice>, V|ice>, V^2|ice> }.

Within K_2 we diagonalize the projected *full microscopic Hamiltonian*
H0 + V(phi), reconstruct the low ice-like band, and average that operator
over smooth twists phi_a in {0, pi}.
"""
from __future__ import annotations

import argparse
import json
import time
from itertools import product
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import eigh
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import eigsh

import recompute_finite_size_artifact as R

HERE = Path(__file__).resolve().parent
FIGS = HERE / "figs"
FIGS.mkdir(exist_ok=True)


def exchange_neighbors(cl, st):
    one = np.uint64(1)
    st = np.uint64(st)
    out = []
    for (i, j), _n in zip(cl.bonds, cl.bond_wrap):
        bi = one << np.uint64(i)
        bj = one << np.uint64(j)
        if ((st & bi) != 0) != ((st & bj) != 0):
            out.append(int(st ^ (bi | bj)))
    return out


def krylov_states(cl, depth):
    all_states = set(int(s) for s in cl.ice_states)
    front = set(all_states)
    levels = [len(all_states)]
    for _ in range(depth):
        new = set()
        for st in front:
            new.update(exchange_neighbors(cl, st))
        front = new - all_states
        all_states.update(new)
        levels.append(len(all_states))
    return np.array(sorted(all_states), dtype=np.uint64), levels


def hamiltonian_in_subspace(cl, states, jpm, phi):
    index = {int(s): i for i, s in enumerate(states)}
    phi = np.asarray(phi, dtype=float)
    A = phi @ np.linalg.inv(cl.Lvecs)
    rows = []
    cols = []
    vals = []
    diag = R.ising_energy(cl, states)
    rows.extend(range(len(states)))
    cols.extend(range(len(states)))
    vals.extend(diag.astype(complex))
    one = np.uint64(1)
    dropped = 0
    for col, st in enumerate(states):
        for (i, j), n in zip(cl.bonds, cl.bond_wrap):
            bi = one << np.uint64(i)
            bj = one << np.uint64(j)
            up_i = (st & bi) != 0
            up_j = (st & bj) != 0
            if up_i == up_j:
                continue
            new = int(st ^ (bi | bj))
            row = index.get(new)
            if row is None:
                dropped += 1
                continue
            d_ij = cl.positions[j] - cl.positions[i] - np.asarray(n) @ cl.Lvecs
            phase = np.exp(-1j * float(A @ d_ij))
            amp = -jpm * (phase if ((not up_i) and up_j) else np.conjugate(phase))
            rows.append(row)
            cols.append(col)
            vals.append(amp)
    H = coo_matrix((np.asarray(vals), (rows, cols)), shape=(len(states), len(states))).tocsr()
    H = 0.5 * (H + H.conj().T)
    return H, dropped


def band_operator(cl, states, ice_indices, jpm, phi, n_band, solver):
    t0 = time.time()
    H, dropped = hamiltonian_in_subspace(cl, states, jpm, phi)
    print(f"    built H in {time.time() - t0:.2f}s, dim={H.shape[0]}, nnz={H.nnz}", flush=True)
    t1 = time.time()
    if solver == "dense":
        Hd = H.toarray()
        E, Psi = eigh(
            Hd,
            subset_by_index=(0, n_band - 1),
            driver="evr",
            overwrite_a=True,
            check_finite=False,
        )
    else:
        E, Psi = eigsh(
            H,
            k=n_band,
            sigma=-1.0,
            which="LM",
            tol=1e-10,
            ncv=max(80, 2 * n_band + 20),
        )
    print(f"    eigensolve in {time.time() - t1:.2f}s", flush=True)
    order = np.argsort(E)
    E = E[order]
    Psi = Psi[:, order]
    X = Psi[ice_indices, :]
    S = X.conj().T @ X
    s_eval, U = np.linalg.eigh(0.5 * (S + S.conj().T))
    if np.min(s_eval) < 1e-12:
        raise RuntimeError(f"singular ice projection: {np.min(s_eval)}")
    Q = X @ (U @ np.diag(1.0 / np.sqrt(s_eval)) @ U.conj().T)
    Hb = Q @ np.diag(E) @ Q.conj().T
    Hb = 0.5 * (Hb + Hb.conj().T)
    return Hb, E, {
        "phi": [float(x) for x in phi],
        "dropped_edges": int(dropped),
        "E0": float(E[0]),
        "Etop": float(E[-1]),
        "ice_overlap_min": float(np.min(s_eval)),
        "ice_overlap_mean": float(np.mean(s_eval)),
    }


def heat(H, T):
    E = np.linalg.eigvalsh(H)
    return E, R.specific_heat(E, T)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jpm", type=float, default=-0.05)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--grid", choices=["one", "two"], default="two")
    ap.add_argument("--n-band", type=int, default=None)
    ap.add_argument("--solver", choices=["sparse", "dense"], default="sparse")
    ap.add_argument("--out", type=Path, default=HERE / "twist_resolved_krylov_jm0p05.npz")
    args = ap.parse_args()

    cl = R.build_cluster("cubic", (1, 1, 1))
    states, levels = krylov_states(cl, args.depth)
    index = {int(s): i for i, s in enumerate(states)}
    ice_indices = np.array([index[int(s)] for s in cl.ice_states], dtype=int)
    twists = [(0.0, 0.0, 0.0)] if args.grid == "one" else list(product([0.0, np.pi], repeat=3))
    n_band = cl.n_ice if args.n_band is None else args.n_band
    print(f"Krylov depth {args.depth}: dim={len(states)}, levels={levels}, ice={cl.n_ice}, n_band={n_band}")

    Hbands = []
    diags = []
    lows = []
    for k, phi in enumerate(twists, 1):
        print(f"[{k}/{len(twists)}] phi/pi={tuple(float(x/np.pi) for x in phi)}", flush=True)
        Hb, E, diag = band_operator(cl, states, ice_indices, args.jpm, np.array(phi), n_band, args.solver)
        Hbands.append(Hb)
        lows.append(E)
        diags.append(diag)
        print(
            f"    E0={diag['E0']:.8f}, top={diag['Etop']:.8f}, "
            f"ice overlap min={diag['ice_overlap_min']:.6f}, dropped={diag['dropped_edges']}",
            flush=True,
        )

    T = np.geomspace(1e-4, 0.12, 900)
    H_phi0 = Hbands[0]
    H_avg = sum(Hbands) / len(Hbands)
    E_phi0, C_phi0 = heat(H_phi0, T)
    E_avg, C_avg = heat(H_avg, T)
    pt = R.sw_order23(cl, verbose=False)
    E_pt_all, C_pt_all = heat(R.assemble(cl, pt, args.jpm, "all"), T)
    E_pt_clean, C_pt_clean = heat(R.assemble(cl, pt, args.jpm, "delta0"), T)
    summary = {
        "jpm": args.jpm,
        "depth": args.depth,
        "krylov_dim": int(len(states)),
        "n_band": int(n_band),
        "levels": levels,
        "grid": args.grid,
        "n_twists": len(twists),
        "diagnostics": diags,
        "g4": 4 * args.jpm * args.jpm,
        "ghex": 12 * abs(args.jpm) ** 3,
        "Tpk_krylov_phi0": R.refined_peak(T, C_phi0),
        "Tpk_krylov_twist_operator_avg": R.refined_peak(T, C_avg),
        "Tpk_pt_all": R.refined_peak(T, C_pt_all),
        "Tpk_pt_delta0": R.refined_peak(T, C_pt_clean),
    }
    print(json.dumps(summary, indent=2))
    np.savez_compressed(
        args.out,
        T=T,
        C_phi0=C_phi0,
        C_avg=C_avg,
        C_pt_all=C_pt_all,
        C_pt_clean=C_pt_clean,
        E_phi0=E_phi0,
        E_avg=E_avg,
        E_pt_all=E_pt_all,
        E_pt_clean=E_pt_clean,
        H_phi0=H_phi0,
        H_avg=H_avg,
        E_low_by_twist=np.array(lows),
        summary=json.dumps(summary),
    )
    args.out.with_suffix(".json").write_text(json.dumps(summary, indent=2))

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.plot(T, C_phi0 / cl.n_sites, color="#e67e22", lw=2.0, label="Krylov band, $\\phi=0$")
    ax.plot(T, C_avg / cl.n_sites, color="#27ae60", lw=2.0, label="Krylov band, twist-operator avg")
    ax.plot(T, C_pt_all / cl.n_sites, color="#e67e22", lw=1.2, ls=":", label="SW all")
    ax.plot(T, C_pt_clean / cl.n_sites, color="#27ae60", lw=1.2, ls=":", label="SW $\\delta=0$")
    ax.axvline(summary["g4"], color="black", ls="-.", lw=0.9, label="$g_4$")
    ax.axvline(summary["ghex"], color="#8e44ad", ls="--", lw=0.9, label="$g_{\\rm hex}$")
    ax.set_xscale("log")
    ax.set_xlabel("$T/J_{zz}$")
    ax.set_ylabel("$C(T)/N$")
    ax.set_title(f"Twist-resolved full-Hamiltonian Krylov test, depth {args.depth}, $J_\\pm={args.jpm:+.2f}$")
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    for ext in ("pdf", "png"):
        fig.savefig(FIGS / f"fig_twist_resolved_krylov.{ext}", bbox_inches="tight", dpi=220)
    plt.close(fig)
    print(f"wrote {args.out}")
    print(f"wrote {args.out.with_suffix('.json')}")
    print(f"wrote {FIGS / 'fig_twist_resolved_krylov.pdf'}")


if __name__ == "__main__":
    main()
