"""
Analyze the bare vs twist-averaged 16-site pyrochlore demo.

Reads the FTLM and Lanczos HDF5 results from the 8 twist corners,
computes:
  - C(T), S(T), F(T), <E>(T) at each corner
  - Twist-averaged quantities (corner average)
  - Low-energy spectrum E_n(phi) and spurious 4-cycle gap
  - Effective ring-exchange model perturbative coefficients (diagnostic)

Generates publication-quality figures into figs/ for the LaTeX writeup.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from itertools import product

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import cluster_geometry_audit as cg  # noqa: E402


def robust_manifold_gap(e, floor=1e-11, jump_ratio=1e4, jump_abs=1e-6, max_check=20):
    """Return (degeneracy, gap): the number of quasi-degenerate ground
    states (split only by machine precision) and the true energy gap to
    the first excited manifold.

    Does NOT assume a fixed ground-manifold size (e.g. "8-fold"): it scans
    consecutive-eigenvalue spacings and reports the first one that jumps by
    >= jump_ratio over every spacing seen so far. A fixed-index cutoff
    (e.g. always reading E_8-E_0) silently returns ~0 whenever the true
    ground degeneracy exceeds the assumed size, which happens at several
    twist corners on this cluster.
    """
    n = min(max_check, len(e) - 1)
    gaps = np.diff(e[: n + 1])
    maxprior = floor
    for k in range(1, len(gaps) + 1):
        g = gaps[k - 1]
        if g > jump_abs and g > jump_ratio * maxprior:
            return k, float(e[k])
        maxprior = max(maxprior, g)
    return n + 1, float("nan")


def exact_ice_manifold_count(n_sites: int, dim) -> int:
    """Exact combinatorial count of 2-in-2-out ('ice rule') configurations
    on this finite pyrochlore cluster, by brute-force enumeration over all
    2**n_sites spin configurations (tractable for n_sites <~ 20). This is
    the correct finite-size benchmark for the entropy plateau -- distinct
    from (and, on small clusters, numerically different from) the
    infinite-volume Pauling estimate 0.5*ln(3/2) per site.
    """
    vertices, edges, _tets, _bond_wrap, adj = cg.build_graph(*dim)
    tets = cg._enumerate_4_cliques(adj, n_sites)
    count = 0
    for x in range(1 << n_sites):
        bits = [(x >> i) & 1 for i in range(n_sites)]
        if all(sum(bits[i] for i in t) == 2 for t in tets):
            count += 1
    return count


def read_ftlm(h5path: Path):
    with h5py.File(h5path, "r") as f:
        T = f["ftlm/averaged/temperatures"][...]
        E = f["ftlm/averaged/energy"][...]
        E_err = f["ftlm/averaged/energy_error"][...]
        C = f["ftlm/averaged/specific_heat"][...]
        C_err = f["ftlm/averaged/specific_heat_error"][...]
        S = f["ftlm/averaged/entropy"][...]
        F = f["ftlm/averaged/free_energy"][...]
    return dict(T=T, E=E, E_err=E_err, C=C, C_err=C_err, S=S, F=F)


def read_eigvals(h5path: Path):
    with h5py.File(h5path, "r") as f:
        eigs = f["/eigendata/eigenvalues"][...]
    return np.sort(eigs)


def label_phi(phi):
    parts = []
    for p in phi:
        if abs(p) < 1e-8:
            parts.append("0")
        elif abs(p - np.pi) < 1e-8:
            parts.append(r"\pi")
        else:
            parts.append(f"{p:.2f}")
    return r"(" + ",".join(parts) + r")"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(ROOT / "output" / "demo"))
    ap.add_argument("--figs", default=str(ROOT / "paper" / "figs"))
    ap.add_argument("--suffix", default="",
                    help="suffix appended to figure / json filenames (e.g. _jpm005)")
    args = ap.parse_args()

    root = Path(args.root)
    figs = Path(args.figs)
    figs.mkdir(parents=True, exist_ok=True)
    suf = args.suffix

    summary = json.loads((root / "summary.json").read_text())
    Jpm = summary["Jpm"]
    Jzz = summary["Jzz"]

    rows = summary["twists"]

    ftlm_data = {}
    eigs_data = {}
    for r in rows:
        phi = tuple(r["phi"])
        ftlm_data[phi] = read_ftlm(Path(r["ftlm_h5"]))
        eigs_data[phi] = read_eigvals(Path(r["spec_h5"]))

    bare_phi = (0.0, 0.0, 0.0)
    pi = float(np.pi)
    cube_corners = list(product([0.0, pi], repeat=3))

    bare = ftlm_data[bare_phi]
    T = bare["T"]
    n_T = len(T)

    C_stack = np.stack([ftlm_data[p]["C"] for p in cube_corners], axis=0)
    E_stack = np.stack([ftlm_data[p]["E"] for p in cube_corners], axis=0)
    S_stack = np.stack([ftlm_data[p]["S"] for p in cube_corners], axis=0)
    Cerr_stack = np.stack([ftlm_data[p]["C_err"] for p in cube_corners], axis=0)
    Eerr_stack = np.stack([ftlm_data[p]["E_err"] for p in cube_corners], axis=0)

    C_avg = C_stack.mean(axis=0)
    E_avg = E_stack.mean(axis=0)
    S_avg = S_stack.mean(axis=0)
    C_avg_err = np.sqrt((Cerr_stack ** 2).mean(axis=0)) / np.sqrt(len(cube_corners))
    E_avg_err = np.sqrt((Eerr_stack ** 2).mean(axis=0)) / np.sqrt(len(cube_corners))
    C_spread = C_stack.std(axis=0)

    np.savez(figs / f"thermo_curves{suf}.npz",
             T=T,
             C_corners=C_stack, E_corners=E_stack, S_corners=S_stack,
             C_corners_err=Cerr_stack, E_corners_err=Eerr_stack,
             C_avg=C_avg, C_avg_err=C_avg_err, C_spread=C_spread,
             E_avg=E_avg, E_avg_err=E_avg_err, S_avg=S_avg,
             corners=np.array(cube_corners),
             Jpm=Jpm, Jzz=Jzz)

    g_hex = 12.0 * abs(Jpm) ** 3 / Jzz ** 2
    g_4cycle_est = 4.0 * abs(Jpm) ** 2 / Jzz
    schottky_T_factor = 0.4168
    T_hex = schottky_T_factor * g_hex
    T_4cycle = schottky_T_factor * g_4cycle_est

    class_color = {0: "tab:red", 1: "tab:cyan", 2: "tab:green", 3: "tab:olive"}
    class_label = {
        0: r"$|\varphi|_{\pi}=0$  (bare, $\boldsymbol{\varphi}=\mathbf{0}$)",
        1: r"$|\varphi|_{\pi}=1$  (3 corners, partial twist)",
        2: r"$|\varphi|_{\pi}=2$  (3 corners)",
        3: r"$|\varphi|_{\pi}=3$  (corner $(\pi,\pi,\pi)$)",
    }

    def n_pi_of(p):
        return sum(1 for x in p if abs(x - pi) < 1e-8)

    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    seen_classes = set()
    for k, p in enumerate(cube_corners):
        n_pi = n_pi_of(p)
        col = class_color[n_pi]
        lw = 2.0 if p == bare_phi else 1.0
        alpha = 1.0 if p == bare_phi else 0.65
        zorder = 5 if p == bare_phi else 3
        lbl = class_label[n_pi] if n_pi not in seen_classes else None
        seen_classes.add(n_pi)
        ax.plot(T, ftlm_data[p]["C"], lw=lw, alpha=alpha, color=col,
                label=lbl, zorder=zorder)
    ax.fill_between(T, C_avg - C_avg_err, C_avg + C_avg_err,
                    color="tab:blue", alpha=0.20)
    ax.plot(T, C_avg, lw=2.6, color="tab:blue",
            label=r"twist-averaged $\overline{C}(T)$", zorder=6)

    ax.axvline(g_hex, color="black", ls="--", lw=1.2, alpha=0.85,
               label=fr"$g_{{\mathrm{{hex}}}} = 12|J_\pm|^3/J_{{zz}}^2 \approx {g_hex:.3g}$")
    ax.axvline(g_4cycle_est, color="tab:orange", ls=":", lw=1.2, alpha=0.85,
               label=fr"$\sim 4|J_\pm|^2/J_{{zz}} \approx {g_4cycle_est:.3g}$ (4-cycle scale)")

    ax.set_xscale("log")
    ax.set_xlabel(r"$T \, / \, J_{zz}$")
    ax.set_ylabel(r"specific heat $C(T)$")
    ax.set_title(rf"16-site pyrochlore, $J_\pm={Jpm:+.3g}\,J_{{zz}}$: $C(T)$ resolved by corner parity class")
    ax.legend(frameon=False, loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(figs / f"fig_specific_heat{suf}.pdf")
    fig.savefig(figs / f"fig_specific_heat{suf}.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.8, 4.3))
    diff = bare["C"] - C_avg
    ax.plot(T, diff, lw=2.0, color="tab:purple",
            label=r"$C_{\rm bare}(T) - \overline{C}(T)$")
    ax.fill_between(T, -C_spread, +C_spread, color="gray", alpha=0.25,
                    label=r"corner spread $\pm\sigma_\varphi[C(T;\varphi)]$")
    ax.axhline(0, color="black", lw=0.5)
    ax.axvline(g_hex, color="black", ls="--", lw=1.0, alpha=0.85,
               label=fr"$g_{{\mathrm{{hex}}}} = 12|J_\pm|^3/J_{{zz}}^2 \approx {g_hex:.3g}$")
    ax.axvline(g_4cycle_est, color="tab:orange", ls=":", lw=1.0, alpha=0.85,
               label=fr"$\sim 4|J_\pm|^2/J_{{zz}} \approx {g_4cycle_est:.3g}$ (4-cycle scale)")
    ax.set_xscale("log")
    ax.set_xlabel(r"$T \, / \, J_{zz}$")
    ax.set_ylabel(r"$\Delta C(T)$")
    ax.set_title(r"Finite-size correction extracted by twist averaging")
    ax.legend(frameon=False, loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(figs / f"fig_specific_heat_diff{suf}.pdf")
    fig.savefig(figs / f"fig_specific_heat_diff{suf}.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    mask_low = T < 0.18
    seen_classes = set()
    for k, p in enumerate(cube_corners):
        n_pi = n_pi_of(p)
        col = class_color[n_pi]
        lw = 2.0 if p == bare_phi else 1.1
        alpha = 1.0 if p == bare_phi else 0.7
        zorder = 5 if p == bare_phi else 3
        lbl = class_label[n_pi] if n_pi not in seen_classes else None
        seen_classes.add(n_pi)
        ax.plot(T[mask_low], ftlm_data[p]["C"][mask_low], lw=lw, alpha=alpha, color=col,
                label=lbl, zorder=zorder)
    ax.fill_between(T[mask_low], (C_avg - C_avg_err)[mask_low], (C_avg + C_avg_err)[mask_low],
                    color="tab:blue", alpha=0.20)
    ax.plot(T[mask_low], C_avg[mask_low], lw=2.4, color="tab:blue",
            label=r"twist-averaged $\overline{C}(T)$", zorder=6)

    Tp_bare = T[mask_low][np.argmax(bare["C"][mask_low])]
    Tp_avg = T[mask_low][np.argmax(C_avg[mask_low])]
    ax.axvline(Tp_bare, color="tab:red", ls=(0, (3, 3)), lw=0.8, alpha=0.7)
    ax.axvline(Tp_avg, color="tab:blue", ls=(0, (3, 3)), lw=0.8, alpha=0.7)

    ax.axvline(g_hex, color="black", ls="--", lw=1.4,
               label=fr"$g_{{\mathrm{{hex}}}} = 12|J_\pm|^3/J_{{zz}}^2 \approx {g_hex:.3g}$")
    ax.axvline(g_4cycle_est, color="tab:orange", ls=":", lw=1.4,
               label=fr"$4|J_\pm|^2/J_{{zz}} \approx {g_4cycle_est:.3g}$ (spurious 4-cycle)")
    ax.axvline(schottky_T_factor * g_hex, color="black", ls=(0, (1, 2, 3, 2)), lw=1.0, alpha=0.7,
               label=fr"$0.42\,g_{{\mathrm{{hex}}}}\approx{T_hex:.3g}$ (Schottky peak from $g_{{\mathrm{{hex}}}}$)")

    ax.set_xscale("log")
    ax.set_xlabel(r"$T \, / \, J_{zz}$")
    ax.set_ylabel(r"specific heat $C(T)$")
    ax.set_title(r"Low-$T$ region: spurious 4-cycle vs genuine hexagonal ring-exchange scales")
    ax.legend(frameon=False, loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(figs / f"fig_specific_heat_lowT{suf}.pdf")
    fig.savefig(figs / f"fig_specific_heat_lowT{suf}.png", dpi=160)
    plt.close(fig)

    e0_per_corner = np.array([eigs_data[p][0] for p in cube_corners])
    ground_deg_per_corner = []
    gap8 = []
    for p in cube_corners:
        e_rel = eigs_data[p] - eigs_data[p][0]
        k, g = robust_manifold_gap(e_rel)
        ground_deg_per_corner.append(k)
        gap8.append(g)
    ground_deg_per_corner = np.array(ground_deg_per_corner)
    gap8 = np.array(gap8)  # true gap to the first excited manifold (ground degeneracy varies per corner -- NOT a fixed 8)
    Tp_per_corner = []
    for p in cube_corners:
        c = ftlm_data[p]["C"]
        Tp_per_corner.append(T[mask_low][np.argmax(c[mask_low])])
    Tp_per_corner = np.array(Tp_per_corner)
    n_pi_per_corner = np.array([n_pi_of(p) for p in cube_corners])

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0))
    xs = np.arange(len(cube_corners))
    cs = [class_color[n] for n in n_pi_per_corner]

    axes[0].bar(xs, e0_per_corner - e0_per_corner.min(),
                color=cs, edgecolor="black", linewidth=0.5)
    axes[0].set_xticks(xs)
    axes[0].set_xticklabels([
        "(" + ",".join("$\\pi$" if abs(x - pi) < 1e-8 else "0" for x in p) + ")"
        for p in cube_corners
    ], rotation=30, fontsize=8)
    axes[0].set_ylabel(r"$E_0(\boldsymbol{\varphi}) - \min_\varphi E_0$  $/J_{zz}$")
    axes[0].set_title(r"(a) ground-state energy")
    axes[0].grid(alpha=0.3, axis="y")

    axes[1].bar(xs, gap8, color=cs, edgecolor="black", linewidth=0.5)
    axes[1].axhline(g_4cycle_est, color="tab:orange", ls=":", lw=1.2,
                    label=fr"$4|J_\pm|^2/J_{{zz}} \approx {g_4cycle_est:.3g}$")
    axes[1].axhline(g_hex, color="black", ls="--", lw=1.2,
                    label=fr"$g_{{\mathrm{{hex}}}} \approx {g_hex:.3g}$")
    for xi, (gv, dgv) in enumerate(zip(gap8, ground_deg_per_corner)):
        axes[1].text(xi, max(gv, 1.15e-3), str(int(dgv)),
                     ha="center", va="bottom", fontsize=7)
    axes[1].set_xticks(xs)
    axes[1].set_xticklabels([
        "(" + ",".join("$\\pi$" if abs(x - pi) < 1e-8 else "0" for x in p) + ")"
        for p in cube_corners
    ], rotation=30, fontsize=8)
    axes[1].set_ylabel(r"gap to first excited manifold  $/J_{zz}$")
    axes[1].set_title("(b) gap above the quasi-degenerate ground manifold\n(label = ground-manifold degeneracy)")
    axes[1].set_yscale("log")
    axes[1].set_ylim(1e-3, 0.2)
    axes[1].legend(frameon=False, loc="lower right", fontsize=8)
    axes[1].grid(alpha=0.3, axis="y", which="both")

    axes[2].bar(xs, Tp_per_corner, color=cs, edgecolor="black", linewidth=0.5)
    axes[2].axhline(g_4cycle_est, color="tab:orange", ls=":", lw=1.2,
                    label=fr"$4|J_\pm|^2/J_{{zz}} \approx {g_4cycle_est:.3g}$")
    axes[2].axhline(g_hex, color="black", ls="--", lw=1.2,
                    label=fr"$g_{{\mathrm{{hex}}}} \approx {g_hex:.3g}$")
    axes[2].set_xticks(xs)
    axes[2].set_xticklabels([
        "(" + ",".join("$\\pi$" if abs(x - pi) < 1e-8 else "0" for x in p) + ")"
        for p in cube_corners
    ], rotation=30, fontsize=8)
    axes[2].set_ylabel(r"low-$T$ $C(T)$ peak position  $/J_{zz}$")
    axes[2].set_title(r"(c) low-$T$ $C(T)$ peak")
    axes[2].set_yscale("log")
    axes[2].legend(frameon=False, loc="lower right", fontsize=8)
    axes[2].grid(alpha=0.3, axis="y", which="both")

    handles = [plt.Rectangle((0, 0), 1, 1, color=class_color[n]) for n in (0, 1, 2, 3)]
    labels = [class_label[n] for n in (0, 1, 2, 3)]
    fig.legend(handles, labels, loc="upper center", ncol=4, fontsize=9,
               frameon=False, bbox_to_anchor=(0.5, 1.04))
    fig.tight_layout()
    fig.savefig(figs / f"fig_corner_classes{suf}.pdf", bbox_inches="tight")
    fig.savefig(figs / f"fig_corner_classes{suf}.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    diagnostics_scale = {
        "g_hex_HFB": float(g_hex),
        "g_hex_HFB_formula": "12 * |Jpm|^3 / Jzz^2",
        "g_4cycle_estimate": float(g_4cycle_est),
        "g_4cycle_formula": "4 * |Jpm|^2 / Jzz",
        "Schottky_T_peak_from_g_hex": float(T_hex),
        "Schottky_T_peak_from_4cycle": float(T_4cycle),
        "C_peak_T_bare_lowT": float(Tp_bare),
        "C_peak_T_twist_avg_lowT": float(Tp_avg),
        "ratio_Tpeak_bare_to_g_hex": float(Tp_bare / g_hex),
        "ratio_Tpeak_avg_to_g_hex": float(Tp_avg / g_hex),
    }
    (figs / f"diagnostics_scales{suf}.json").write_text(json.dumps(diagnostics_scale, indent=2))

    n_eigs_show = min(48, min(len(v) for v in eigs_data.values()))
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    corner_x = np.arange(len(cube_corners))
    for n in range(n_eigs_show):
        ys = [eigs_data[p][n] for p in cube_corners]
        col = plt.cm.plasma(0.05 + 0.85 * n / n_eigs_show)
        ax.plot(corner_x, ys, "o-", lw=0.7, ms=3.5, color=col, alpha=0.7)
    xtl = []
    for p in cube_corners:
        n_pi = sum(1 for x in p if abs(x - pi) < 1e-8)
        if p == (0.0, 0.0, 0.0):
            xtl.append(r"$(0,0,0)$")
        elif p == (pi, pi, pi):
            xtl.append(r"$(\pi,\pi,\pi)$")
        else:
            comps = ["0" if abs(x) < 1e-8 else r"\pi" for x in p]
            xtl.append(f"$({','.join(comps)})$")
    ax.set_xticks(corner_x)
    ax.set_xticklabels(xtl, rotation=30, fontsize=8)
    ax.set_xlabel(r"twist corner $\boldsymbol{\varphi}$")
    ax.set_ylabel(r"$E_n(\boldsymbol{\varphi})$")
    ax.set_title(rf"Lowest {n_eigs_show} eigenvalues vs twist corner")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(figs / f"fig_spectrum_vs_twist{suf}.pdf")
    fig.savefig(figs / f"fig_spectrum_vs_twist{suf}.png", dpi=160)
    plt.close(fig)

    e0 = np.array([eigs_data[p][0] for p in cube_corners])
    e1 = np.array([eigs_data[p][1] for p in cube_corners])
    # intra-manifold spread: energy of the last state assigned to the ground
    # manifold minus e0 -- should sit at the machine-precision floor,
    # confirming the manifold really is degenerate (not assumed to be size 8)
    intra_manifold_spread = np.array([
        eigs_data[p][max(int(ground_deg_per_corner[i]) - 1, 0)] - eigs_data[p][0]
        for i, p in enumerate(cube_corners)
    ])

    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    ax.plot(corner_x, gap8, "o-", lw=1.5, ms=6, color="tab:blue",
            label=r"gap to first excited manifold (true gap above ice manifold)")
    ax.plot(corner_x, intra_manifold_spread, "s-", lw=1.5, ms=6, color="tab:red",
            label=r"spread within the quasi-degenerate ground manifold")
    ax.plot(corner_x, e1 - e0, "^-", lw=1.2, ms=5, color="gray",
            label=r"$E_1-E_0$")
    ax.set_xticks(corner_x)
    ax.set_xticklabels(xtl, rotation=30, fontsize=8)
    ax.set_xlabel(r"twist corner $\boldsymbol{\varphi}$")
    ax.set_ylabel(r"$\Delta E$")
    ax.set_yscale("symlog", linthresh=1e-7)
    ax.set_title(r"Spurious vs physical gaps on the 16-site cluster")
    ax.legend(frameon=False, fontsize=9, loc="best")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(figs / f"fig_gaps_vs_twist{suf}.pdf")
    fig.savefig(figs / f"fig_gaps_vs_twist{suf}.png", dpi=160)
    plt.close(fig)

    e0_mean = e0.mean()
    e0_spread = e0.max() - e0.min()

    n_sites = summary["n_sites"]
    N_ice = exact_ice_manifold_count(n_sites, tuple(summary["dim"]))
    S_ice_per_site = float(np.log(N_ice) / n_sites)

    plateau_mask = (T > 0.02) & (T < 0.25)
    T_plateau = T[plateau_mask]
    i_bare_min = int(np.argmin(bare["C"][plateau_mask]))
    i_avg_min = int(np.argmin(C_avg[plateau_mask]))

    diagnostics = {
        "Jpm": Jpm, "Jzz": Jzz,
        "n_corners": len(cube_corners),
        "ground_state_E0_phi0": float(e0[0]),
        "ground_state_E0_corner_mean": float(e0_mean),
        "ground_state_E0_corner_spread": float(e0_spread),
        "ground_state_E0_downward_bias": float(e0_mean - e0[0]),
        "ground_manifold_degeneracy_phi0": int(ground_deg_per_corner[0]),
        "ground_manifold_degeneracy_per_corner": [int(x) for x in ground_deg_per_corner],
        "ground_manifold_degeneracy_range": [int(ground_deg_per_corner.min()), int(ground_deg_per_corner.max())],
        "gap_to_next_manifold_phi0": float(gap8[0]),
        "gap_to_next_manifold_per_corner": [float(x) for x in gap8],
        "gap_to_next_manifold_corner_mean": float(np.mean(gap8)),
        "C_peak_T_bare": float(T[np.argmax(bare["C"])]),
        "C_peak_T_twist_avg": float(T[np.argmax(C_avg)]),
        "C_peak_value_bare": float(np.max(bare["C"])),
        "C_peak_value_twist_avg": float(np.max(C_avg)),
        "S_inf_T_bare": float(bare["S"][-1]),
        "S_inf_T_twist_avg": float(S_avg[-1]),
        "two_log2_per_site_x16": 16 * np.log(2),
        "N_ice_exact_2in2out_states": int(N_ice),
        "S_ice_exact_per_site": S_ice_per_site,
        "S_pauling_per_site": float(0.5 * np.log(1.5)),
        "ice_plateau_T_bare": float(T_plateau[i_bare_min]),
        "ice_plateau_C_bare": float(bare["C"][plateau_mask][i_bare_min]),
        "ice_plateau_S_per_site_bare": float(bare["S"][plateau_mask][i_bare_min] / n_sites),
        "ice_plateau_T_twist_avg": float(T_plateau[i_avg_min]),
        "ice_plateau_C_twist_avg": float(C_avg[plateau_mask][i_avg_min]),
        "ice_plateau_S_per_site_twist_avg": float(S_avg[plateau_mask][i_avg_min] / n_sites),
    }
    (figs / f"diagnostics{suf}.json").write_text(json.dumps(diagnostics, indent=2))

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.plot(T, bare["S"] / 16, lw=2.0, color="tab:red",
            label=r"bare $S_{\varphi=0}(T)/N$")
    ax.plot(T, S_avg / 16, lw=2.0, color="tab:blue",
            label=r"twist-averaged $\overline{S}(T)/N$")
    ax.axhline(0.5 * np.log(3.0 / 2.0), color="black", ls="--", lw=0.8,
               label=r"Pauling ice plateau $\frac{1}{2}\ln\frac{3}{2}$ (bulk)")
    ax.axhline(S_ice_per_site, color="tab:green", ls="-.", lw=1.1,
               label=rf"exact 16-site ice count $\ln({N_ice})/16={S_ice_per_site:.4f}$")
    ax.axhline(np.log(2.0), color="gray", ls=":", lw=0.8,
               label=r"$\ln 2$ (full)")
    ax.set_xscale("log")
    ax.set_xlabel(r"$T \, / \, J_{zz}$")
    ax.set_ylabel(r"entropy per site $S(T)/N$")
    ax.set_title(rf"Entropy on the 16-site pyrochlore at $J_\pm={Jpm:+.3g}\,J_{{zz}}$: bare vs twist-averaged")
    ax.legend(frameon=False, loc="best", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(figs / f"fig_entropy{suf}.pdf")
    fig.savefig(figs / f"fig_entropy{suf}.png", dpi=160)
    plt.close(fig)

    print("Wrote figures to", figs)
    print(json.dumps(diagnostics, indent=2))


if __name__ == "__main__":
    main()
