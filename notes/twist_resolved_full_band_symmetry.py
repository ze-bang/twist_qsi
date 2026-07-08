#!/usr/bin/env python3
"""Symmetry-resolved twist-band verification for the full 16-site Hamiltonian.

This diagonalizes the full microscopic XXZ Hamiltonian in the Sz=0 sector,
using the four FCC translations of the 1x1x1 cubic pyrochlore cell to block
diagonalize into four irreps.  It then reconstructs the lowest ice-like band
as an operator in the fixed ice basis and averages that operator over smooth
twist corners.
"""
from __future__ import annotations

import argparse
import json
import time
from itertools import combinations, product
from pathlib import Path

import numpy as np
from scipy.linalg import eigh
from scipy.sparse import coo_matrix, csc_matrix

import recompute_finite_size_artifact as R

HERE = Path(__file__).resolve().parent


def sz0_basis(n_sites):
    states = []
    for occ in combinations(range(n_sites), n_sites // 2):
        s = 0
        for i in occ:
            s |= 1 << i
        states.append(s)
    return np.array(states, dtype=np.uint64)


def site_translation_perms(cl):
    translations = [
        np.array([0.0, 0.0, 0.0]),
        np.array([0.0, 0.5, 0.5]),
        np.array([0.5, 0.0, 0.5]),
        np.array([0.5, 0.5, 0.0]),
    ]
    keys = {}
    for i, r in enumerate(cl.positions):
        frac = np.mod(r, 1.0)
        keys[tuple(np.rint(4 * frac).astype(int) % 4)] = i
    perms = []
    for t in translations:
        p = []
        for r in cl.positions:
            frac = np.mod(r + t, 1.0)
            key = tuple(np.rint(4 * frac).astype(int) % 4)
            p.append(keys[key])
        perms.append(np.array(p, dtype=np.int64))
    return perms


def translate_state(st, perm):
    out = 0
    x = int(st)
    i = 0
    while x:
        if x & 1:
            out |= 1 << int(perm[i])
        x >>= 1
        i += 1
    return out


def build_sector_basis(states, perms, chars):
    """Return sparse V matrices (full basis x sector basis) for each irrep."""
    index = {int(s): i for i, s in enumerate(states)}
    transformed = np.empty((len(perms), len(states)), dtype=np.int64)
    for gi, p in enumerate(perms):
        for si, st in enumerate(states):
            transformed[gi, si] = index[translate_state(st, p)]

    sectors = {}
    for ir, char in chars.items():
        visited = np.zeros(len(states), dtype=bool)
        rows = []
        cols = []
        vals = []
        col = 0
        for si in range(len(states)):
            if visited[si]:
                continue
            orb = sorted(set(int(transformed[gi, si]) for gi in range(len(perms))))
            for oi in orb:
                visited[oi] = True
            coeff_by_state = {}
            for gi, chi in enumerate(char):
                ti = int(transformed[gi, si])
                coeff_by_state[ti] = coeff_by_state.get(ti, 0.0) + chi
            norm = np.sqrt(sum(abs(v) ** 2 for v in coeff_by_state.values()))
            if norm < 1e-12:
                continue
            for row, val in coeff_by_state.items():
                rows.append(row)
                cols.append(col)
                vals.append(val / norm)
            col += 1
        V = coo_matrix((vals, (rows, cols)), shape=(len(states), col), dtype=complex).tocsc()
        sectors[ir] = V
        print(f"sector {ir}: dim={col}", flush=True)
    return sectors


def full_hamiltonian_sparse(cl, states, jpm, phi):
    index = {int(s): i for i, s in enumerate(states)}
    dim = len(states)
    phi = np.asarray(phi, dtype=float)
    A = phi @ np.linalg.inv(cl.Lvecs)
    rows = []
    cols = []
    vals = []
    diag = R.ising_energy(cl, states)
    rows.extend(range(dim))
    cols.extend(range(dim))
    vals.extend(diag.astype(complex))
    one = np.uint64(1)
    for col, st in enumerate(states):
        for (i, j), n in zip(cl.bonds, cl.bond_wrap):
            bi = one << np.uint64(i)
            bj = one << np.uint64(j)
            up_i = (st & bi) != 0
            up_j = (st & bj) != 0
            if up_i == up_j:
                continue
            d_ij = cl.positions[j] - cl.positions[i] - np.asarray(n) @ cl.Lvecs
            phase = np.exp(-1j * float(A @ d_ij))
            new = int(st ^ (bi | bj))
            row = index[new]
            amp = -jpm * (phase if ((not up_i) and up_j) else np.conjugate(phase))
            rows.append(row)
            cols.append(col)
            vals.append(amp)
    H = coo_matrix((np.asarray(vals), (rows, cols)), shape=(dim, dim)).tocsc()
    return 0.5 * (H + H.conj().T)


def band_from_twist(cl, states, sectors, ice_indices, jpm, phi, n_band):
    H = full_hamiltonian_sparse(cl, states, jpm, phi)
    candidates = []
    t0 = time.time()
    for name, V in sectors.items():
        Hsec = (V.conj().T @ (H @ V)).toarray()
        Hsec = 0.5 * (Hsec + Hsec.conj().T)
        # Keep a margin above the requested band in case one sector dominates.
        nkeep = min(Hsec.shape[0], n_band + 20)
        evals, evecs = eigh(
            Hsec,
            subset_by_index=(0, nkeep - 1),
            driver="evr",
            overwrite_a=True,
            check_finite=False,
        )
        Vice = V[ice_indices, :].toarray()
        Xsec = Vice @ evecs[:, :nkeep]
        for a in range(nkeep):
            candidates.append((float(evals[a]), name, Xsec[:, a]))
        print(f"    sector {name}: dim={V.shape[1]}, E0={evals[0]:.8f}", flush=True)
    candidates.sort(key=lambda x: x[0])
    selected = candidates[:n_band]
    E = np.array([x[0] for x in selected], dtype=float)
    X = np.column_stack([x[2] for x in selected])
    S = X.conj().T @ X
    s_eval, U = np.linalg.eigh(0.5 * (S + S.conj().T))
    if np.min(s_eval) <= 1e-12:
        raise RuntimeError(f"singular ice projection: min={np.min(s_eval)}")
    Sinvhalf = U @ np.diag(1.0 / np.sqrt(s_eval)) @ U.conj().T
    Q = X @ Sinvhalf
    Hband = Q @ np.diag(E) @ Q.conj().T
    Hband = 0.5 * (Hband + Hband.conj().T)
    diag = {
        "phi": [float(x) for x in phi],
        "elapsed_s": time.time() - t0,
        "E_min": float(E[0]),
        "E_max_band": float(E[-1]),
        "ice_overlap_min": float(np.min(s_eval)),
        "ice_overlap_mean": float(np.mean(s_eval)),
        "sector_counts": {name: sum(1 for _, n, _ in selected if n == name) for name in sectors},
    }
    return Hband, E, diag


def heat(H, T):
    E = np.linalg.eigvalsh(H)
    return E, R.specific_heat(E, T)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jpm", type=float, default=-0.05)
    ap.add_argument("--grid", choices=["one", "two"], default="two")
    ap.add_argument("--out", type=Path, default=HERE / "twist_resolved_full_band_sym_jm0p05.npz")
    args = ap.parse_args()

    cl = R.build_cluster("cubic", (1, 1, 1))
    states = sz0_basis(cl.n_sites)
    state_index = {int(s): i for i, s in enumerate(states)}
    ice_indices = np.array([state_index[int(s)] for s in cl.ice_states], dtype=np.int64)
    perms = site_translation_perms(cl)
    chars = {
        "++": np.array([1, 1, 1, 1], dtype=complex),
        "+-": np.array([1, 1, -1, -1], dtype=complex),
        "-+": np.array([1, -1, 1, -1], dtype=complex),
        "--": np.array([1, -1, -1, 1], dtype=complex),
    }
    sectors = build_sector_basis(states, perms, chars)

    twists = [(0.0, 0.0, 0.0)] if args.grid == "one" else list(product([0.0, 2 * np.pi], repeat=3))
    T = np.geomspace(1e-4, 0.12, 900)
    Hbands = []
    diagnostics = []
    lows = []
    print(f"cluster N={cl.n_sites}, Sz0={len(states)}, ice={cl.n_ice}, Jpm={args.jpm:+.4f}", flush=True)
    for k, phi in enumerate(twists, 1):
        print(f"[{k}/{len(twists)}] phi/pi={tuple(float(x/np.pi) for x in phi)}", flush=True)
        Hb, E, diag = band_from_twist(cl, states, sectors, ice_indices, args.jpm, np.array(phi), cl.n_ice)
        Hbands.append(Hb)
        lows.append(E)
        diagnostics.append(diag)
        print(
            f"    band E0={diag['E_min']:.8f}, top={diag['E_max_band']:.8f}, "
            f"ice overlap min={diag['ice_overlap_min']:.6f}, sectors={diag['sector_counts']}",
            flush=True,
        )

    H_phi0 = Hbands[0]
    H_avg = sum(Hbands) / len(Hbands)
    E_phi0, C_phi0 = heat(H_phi0, T)
    E_avg, C_avg = heat(H_avg, T)

    pt = R.sw_order23(cl, verbose=False)
    E_pt_all, C_pt_all = heat(R.assemble(cl, pt, args.jpm, "all"), T)
    E_pt_clean, C_pt_clean = heat(R.assemble(cl, pt, args.jpm, "delta0"), T)
    summary = {
        "jpm": args.jpm,
        "grid": args.grid,
        "n_twists": len(twists),
        "n_sites": cl.n_sites,
        "sz0_dim": int(len(states)),
        "ice_dim": int(cl.n_ice),
        "diagnostics": diagnostics,
        "g4": 4 * args.jpm * args.jpm,
        "ghex": 12 * abs(args.jpm) ** 3,
        "Tpk_full_band_phi0": R.refined_peak(T, C_phi0),
        "Tpk_full_band_twist_avg_operator": R.refined_peak(T, C_avg),
        "Tpk_pt_all": R.refined_peak(T, C_pt_all),
        "Tpk_pt_delta0": R.refined_peak(T, C_pt_clean),
    }
    print(json.dumps(summary, indent=2))
    np.savez_compressed(
        args.out,
        T=T,
        H_phi0=H_phi0,
        H_avg=H_avg,
        C_phi0=C_phi0,
        C_avg=C_avg,
        E_phi0=E_phi0,
        E_avg=E_avg,
        C_pt_all=C_pt_all,
        C_pt_clean=C_pt_clean,
        E_pt_all=E_pt_all,
        E_pt_clean=E_pt_clean,
        E_low_by_twist=np.array(lows),
        summary=json.dumps(summary),
    )
    args.out.with_suffix(".json").write_text(json.dumps(summary, indent=2))
    print(f"wrote {args.out}")
    print(f"wrote {args.out.with_suffix('.json')}")


if __name__ == "__main__":
    main()
