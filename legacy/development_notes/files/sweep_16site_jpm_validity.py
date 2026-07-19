#!/usr/bin/env python3
"""16-site dense block-ED validity sweep for the clean ice-band protocol.

For each Jpm and each transported-dipole character point theta_mu in {0, pi},
this script diagonalizes the cubic-16 microscopic Hamiltonian in four
translation sectors, selects the lowest 90 states, pulls that low band back to
the common ice basis with the Loewdin map, and classifies whether the clean
ice-band replacement is controlled.
"""
from __future__ import annotations

import argparse
import json
import time
from itertools import combinations, product
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import eigh
from scipy.sparse import coo_matrix

import recompute_finite_size_artifact as R

HERE = Path(__file__).resolve().parent
FIGS = HERE / "figs"
FIGS.mkdir(exist_ok=True)


def sz0_basis(n_sites: int) -> np.ndarray:
    states = []
    for occ in combinations(range(n_sites), n_sites // 2):
        state = 0
        for i in occ:
            state |= 1 << i
        states.append(state)
    return np.asarray(states, dtype=np.uint64)


def site_translation_perms(cl: R.Cluster):
    translations = [
        np.array([0.0, 0.0, 0.0]),
        np.array([0.0, 0.5, 0.5]),
        np.array([0.5, 0.0, 0.5]),
        np.array([0.5, 0.5, 0.0]),
    ]
    keys = {}
    for i, r in enumerate(cl.positions):
        keys[tuple(np.rint(4 * np.mod(r, 1.0)).astype(int) % 4)] = i
    perms = []
    for t in translations:
        p = []
        for r in cl.positions:
            key = tuple(np.rint(4 * np.mod(r + t, 1.0)).astype(int) % 4)
            p.append(keys[key])
        perms.append(np.asarray(p, dtype=np.int64))
    return perms


def translate_state(state: int, perm: np.ndarray) -> int:
    out = 0
    x = int(state)
    i = 0
    while x:
        if x & 1:
            out |= 1 << int(perm[i])
        x >>= 1
        i += 1
    return out


def build_sector_bases(states: np.ndarray, perms):
    index = {int(s): i for i, s in enumerate(states)}
    transformed = np.empty((len(perms), len(states)), dtype=np.int64)
    for gi, perm in enumerate(perms):
        for si, state in enumerate(states):
            transformed[gi, si] = index[translate_state(int(state), perm)]

    chars = {
        "++": np.array([1, 1, 1, 1], dtype=complex),
        "+-": np.array([1, 1, -1, -1], dtype=complex),
        "-+": np.array([1, -1, 1, -1], dtype=complex),
        "--": np.array([1, -1, -1, 1], dtype=complex),
    }
    sectors = {}
    for name, char in chars.items():
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
        sectors[name] = V
    return sectors


def dipole2_hamiltonian(cl: R.Cluster, states: np.ndarray, jpm: float, theta: np.ndarray):
    index = {int(s): i for i, s in enumerate(states)}
    rows = []
    cols = []
    vals = []

    diag = R.ising_energy(cl, states)
    rows.extend(range(len(states)))
    cols.extend(range(len(states)))
    vals.extend(diag.astype(np.complex128))

    one = np.uint64(1)
    theta = np.asarray(theta, dtype=float)
    for col, state in enumerate(states):
        for (i, j), nwrap in zip(cl.bonds, cl.bond_wrap):
            bi = one << np.uint64(i)
            bj = one << np.uint64(j)
            up_i = (state & bi) != 0
            up_j = (state & bj) != 0
            if up_i == up_j:
                continue

            d_ij = cl.positions[j] - cl.positions[i] - np.asarray(nwrap) @ cl.Lvecs
            phase = np.exp(1j * float(theta @ (2.0 * d_ij)))
            new_state = int(state ^ (bi | bj))
            row = index[new_state]
            amp = -jpm * (phase if ((not up_i) and up_j) else np.conjugate(phase))
            rows.append(row)
            cols.append(col)
            vals.append(amp)

    H = coo_matrix(
        (np.asarray(vals, dtype=np.complex128), (rows, cols)),
        shape=(len(states), len(states)),
    ).tocsc()
    return 0.5 * (H + H.conj().T)


def band_from_theta(cl, states, sectors, ice_indices, jpm: float, theta: np.ndarray, n_band: int):
    H = dipole2_hamiltonian(cl, states, jpm, theta)
    candidates = []
    sector_counts = {}
    for name, V in sectors.items():
        Hsec = (V.conj().T @ (H @ V)).toarray()
        Hsec = 0.5 * (Hsec + Hsec.conj().T)
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
    candidates.sort(key=lambda item: item[0])
    selected = candidates[:n_band]
    E = np.asarray([x[0] for x in selected], dtype=float)
    X = np.column_stack([x[2] for x in selected])
    S = X.conj().T @ X
    s_eval, U = np.linalg.eigh(0.5 * (S + S.conj().T))
    diag = {
        "theta_over_pi": [float(x / np.pi) for x in theta],
        "E0": float(E[0]),
        "Etop_ice": float(E[-1]),
        "ice_overlap_min": float(np.min(s_eval)),
        "ice_overlap_mean": float(np.mean(s_eval)),
        "ice_overlap_max": float(np.max(s_eval)),
    }
    for name in sectors:
        sector_counts[name] = sum(1 for _, sector, _ in selected if sector == name)
    diag["sector_counts"] = sector_counts
    if np.min(s_eval) <= 1e-12:
        return None, E, diag
    Sinvhalf = U @ np.diag(1.0 / np.sqrt(s_eval)) @ U.conj().T
    Q = X @ Sinvhalf
    Hband = Q @ np.diag(E) @ Q.conj().T
    Hband = 0.5 * (Hband + Hband.conj().T)
    return Hband, E, diag


def classify(eta_min: float, peak_ratio: float | None) -> str:
    if eta_min <= 1e-8:
        return "failed ice-band pullback"
    if eta_min >= 0.85 and peak_ratio is not None and 0.45 <= peak_ratio <= 1.8:
        return "controlled ice band"
    if eta_min >= 0.5:
        return "diagnostic gray zone"
    return "failed ice-band pullback"


def twist_grid(mode: str):
    if mode == "one":
        return [(0.0, 0.0, 0.0)]
    if mode == "corners":
        return list(product([0.0, np.pi], repeat=3))
    raise ValueError(mode)


def run_one(cl, states, sectors, ice_indices, jpm: float, grid: str):
    twists = twist_grid(grid)
    t0 = time.time()
    hbands = []
    diagnostics = []
    low_evals = []
    for it, theta in enumerate(twists, 1):
        Hband, E, diag = band_from_theta(
            cl, states, sectors, ice_indices, jpm, np.asarray(theta, dtype=float), cl.n_ice
        )
        diagnostics.append(diag)
        low_evals.append(E)
        if Hband is not None:
            hbands.append(Hband)
        print(
            f"    twist {it}/8 theta/pi={diag['theta_over_pi']} "
            f"eta_min={diag['ice_overlap_min']:.6f} sectors={diag['sector_counts']}",
            flush=True,
        )

    eta_min = min(d["ice_overlap_min"] for d in diagnostics)
    eta_mean_min = min(d["ice_overlap_mean"] for d in diagnostics)
    g4 = 4.0 * jpm * jpm
    ghex = 12.0 * abs(jpm) ** 3

    bare_peak = None
    clean_peak = None
    peak_ratio = None
    if len(hbands) == len(twists) and len(twists) == 8:
        Hbare = hbands[0]
        Hclean = sum(hbands) / len(hbands)
        Ebare = np.linalg.eigvalsh(Hbare)
        Eclean = np.linalg.eigvalsh(Hclean)
        tmax = max(0.2, 2.0 * max(np.ptp(Ebare), np.ptp(Eclean)), 4.0 * g4)
        T = np.geomspace(1e-5, tmax, 1000)
        Cbare = R.specific_heat(Ebare, T)
        Cclean = R.specific_heat(Eclean, T)
        bare_peak = R.refined_peak(T, Cbare)
        clean_peak = R.refined_peak(T, Cclean)
        peak_ratio = clean_peak / ghex if ghex > 0 else None

    return {
        "Jpm": float(jpm),
        "eta_min": float(eta_min),
        "eta_mean_min": float(eta_mean_min),
        "ice_gap_min": None,
        "g4": float(g4),
        "ghex": float(ghex),
        "Tpeak_bare": None if bare_peak is None else float(bare_peak),
        "Tpeak_clean": None if clean_peak is None else float(clean_peak),
        "Tpeak_clean_over_ghex": None if peak_ratio is None else float(peak_ratio),
        "status": classify(eta_min, peak_ratio),
        "elapsed_s": float(time.time() - t0),
        "diagnostics": diagnostics,
    }


def make_plot(rows, out_prefix: Path):
    jabs = np.asarray([abs(r["Jpm"]) for r in rows], dtype=float)
    eta = np.asarray([r["eta_min"] for r in rows], dtype=float)
    ratio = np.asarray([
        np.nan if r["Tpeak_clean_over_ghex"] is None else r["Tpeak_clean_over_ghex"]
        for r in rows
    ])
    order = np.argsort(jabs)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
    axes[0].plot(jabs[order], eta[order], "o-", color="#2c7fb8")
    axes[0].axhline(0.85, color="black", lw=0.8, ls=":")
    axes[0].axhline(0.5, color="black", lw=0.8, ls="--")
    axes[0].set_xlabel(r"$|J_\pm|/J_{zz}$")
    axes[0].set_ylabel(r"$\min_\theta\lambda_{\min}S(\theta)$")
    axes[1].plot(jabs[order], ratio[order], "o-", color="#238b45")
    axes[1].axhline(1.0, color="black", lw=0.8, ls=":")
    axes[1].set_xlabel(r"$|J_\pm|/J_{zz}$")
    axes[1].set_ylabel(r"$T_{\rm peak}^{\rm clean}/g_{\rm hex}$")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out_prefix.with_suffix(f".{ext}"), bbox_inches="tight", dpi=220)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--jpm",
        nargs="+",
        type=float,
        default=[-0.02, -0.03, -0.05, -0.08, -0.10, -0.15, -0.18, -0.25, -0.30],
    )
    ap.add_argument("--grid", choices=["corners", "one"], default="corners")
    ap.add_argument("--out", type=Path, default=HERE / "jpm_validity_sweep_16site.json")
    args = ap.parse_args()

    cl = R.build_cluster("cubic", (1, 1, 1))
    states = sz0_basis(cl.n_sites)
    state_index = {int(s): i for i, s in enumerate(states)}
    ice_indices = np.asarray([state_index[int(s)] for s in cl.ice_states], dtype=np.int64)
    sectors = build_sector_bases(states, site_translation_perms(cl))
    print(
        f"cubic-16 block ED validity sweep: Sz0={len(states)}, ice={cl.n_ice}, "
        f"sector_dims={ {k: v.shape[1] for k, v in sectors.items()} }",
        flush=True,
    )

    rows = []
    for jpm in args.jpm:
        print(f"\n=== Jpm/Jzz={jpm:+.4f} ===", flush=True)
        if abs(jpm) < 1e-15:
            rows.append(
                {
                    "Jpm": 0.0,
                    "eta_min": 1.0,
                    "eta_mean_min": 1.0,
                    "ice_gap_min": None,
                    "g4": 0.0,
                    "ghex": 0.0,
                    "Tpeak_bare": None,
                    "Tpeak_clean": None,
                    "Tpeak_clean_over_ghex": None,
                    "status": "classical point",
                    "elapsed_s": 0.0,
                    "diagnostics": [],
                }
            )
            continue
        row = run_one(cl, states, sectors, ice_indices, jpm, args.grid)
        rows.append(row)
        print(
            f"  status={row['status']} eta_min={row['eta_min']:.6f} "
            f"Tclean/ghex={row['Tpeak_clean_over_ghex']}",
            flush=True,
        )

    out = {
        "method": "dense translation-block ED, cubic-16, fixed Sz=0, dipole2 M=2, lowest 90 states",
        "n_sites": cl.n_sites,
        "sz0_dim": int(len(states)),
        "ice_dim": int(cl.n_ice),
        "n_twists": len(twist_grid(args.grid)),
        "grid": args.grid,
        "rows": rows,
    }
    args.out.write_text(json.dumps(out, indent=2))
    make_plot(rows, FIGS / "fig_jpm_validity_sweep_16site")
    print(f"\nwrote {args.out}")
    print(f"wrote {FIGS / 'fig_jpm_validity_sweep_16site.pdf'}")


if __name__ == "__main__":
    main()
