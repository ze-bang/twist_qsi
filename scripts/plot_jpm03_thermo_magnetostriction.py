#!/usr/bin/env python3
"""
Plot twist-averaged thermodynamics from ``twist_avg_analysis.npz`` and, when present,
quadrupole thermal-expansion coefficients from ``quad_qh_twist_avg_analysis.npz``.

All temperature axes use a **log** scale (QED's log-spaced T grid).

Thermal expansion panels plot
``α_Q = (⟨QH⟩ − ⟨Q⟩⟨H⟩) / T²`` from static connected Q–H response
(``run_quad_static_ftlm_scan.py`` + ``analyze_jpm03_quad_twist_avg.py``).

Outputs PDF + PNG next to the npz unless ``--out`` is given.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ALPHA_EG_KEYS = ("Eg_Q1_twist_avg", "Eg_Q2_twist_avg")
ALPHA_T2G_KEYS = ("T2g_Q_xy_twist_avg", "T2g_Q_xz_twist_avg", "T2g_Q_yz_twist_avg")
ALPHA_LABELS = {
    "Eg_Q1_twist_avg": r"$E_g\,Q_1$",
    "Eg_Q2_twist_avg": r"$E_g\,Q_2$",
    "T2g_Q_xy_twist_avg": r"$T_{2g}\,Q_{xy}$",
    "T2g_Q_xz_twist_avg": r"$T_{2g}\,Q_{xz}$",
    "T2g_Q_yz_twist_avg": r"$T_{2g}\,Q_{yz}$",
}


def _find_flct_files(root: Path) -> list[Path]:
    return sorted(root.rglob("mtpq_spin/flct*.dat"))


def _try_load_mtpq_sz(root: Path) -> tuple[np.ndarray | None, np.ndarray | None]:
    for p in _find_flct_files(root):
        if "_Sx" in p.name or "_Sy" in p.name:
            continue
        try:
            data = np.loadtxt(p)
        except OSError:
            continue
        if data.ndim != 2 or data.shape[1] < 3:
            continue
        beta = data[:, 0]
        sz = data[:, 1]
        return beta, sz
    return None, None


def _style_log_T(ax: plt.Axes, t_min: float) -> None:
    ax.set_xscale("log")
    ax.set_xlim(left=max(t_min * 0.98, 1e-12))
    ax.set_xlabel(r"$T\,/\,J_{zz}$")


def _plot_j3_family(
    ax: plt.Axes,
    T: np.ndarray,
    Y: np.ndarray,
    Yerr: np.ndarray | None,
    j3: np.ndarray,
    colors: np.ndarray,
    *,
    ylabel: str,
    title: str,
    t_min: float,
) -> None:
    for i, jv in enumerate(j3):
        c = colors[i] if len(j3) > 1 else "tab:blue"
        lab = rf"$J_3={float(jv):.3g}$"
        ax.plot(T, Y[i], color=c, lw=1.6, label=lab)
        if Yerr is not None and np.any(Yerr[i] > 0):
            ax.fill_between(T, Y[i] - Yerr[i], Y[i] + Yerr[i], color=c, alpha=0.18, lw=0)
    _style_log_T(ax, t_min)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(frameon=True, fontsize=8)
    ax.grid(True, which="both", alpha=0.35)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root",
        type=str,
        default="/home/pc_linux/exact_diagonalization_clean/twist_qsi_demo/output/jpm01_j3_scan_symm_light",
        help="directory containing twist_avg_analysis.npz and summary.json",
    )
    ap.add_argument(
        "--quad-subdir",
        type=str,
        default="quad_qh_static",
        help="hamiltonian subdirectory for α_Q HDF5 (default: quad_qh_static)",
    )
    ap.add_argument(
        "--quad-npz",
        type=str,
        default="",
        help="override quad twist-average npz (default: <root>/quad_qh_twist_avg_analysis.npz)",
    )
    ap.add_argument("--out", type=str, default="", help="output base path without extension")
    args = ap.parse_args()

    root = Path(args.root)
    npz_path = root / "twist_avg_analysis.npz"
    if not npz_path.is_file():
        raise SystemExit(f"missing {npz_path} — run analyze_jpm03_twist_j3.py first")

    d = np.load(npz_path)
    T = d["T"]
    j3 = d["j3"]
    C = d["C_twist_avg"]
    E = d["E_twist_avg"]
    S = d["S_twist_avg"]
    Ce = d["C_twist_avg_err"]
    Jpm = float(d["Jpm"])
    Jzz = float(d["Jzz"])
    t_min = float(np.min(T[T > 0])) if np.any(T > 0) else 1e-3

    summary = json.loads((root / "summary.json").read_text())
    use_symm = summary.get("use_symm", False)
    thermo_mode = summary.get("thermo_mode", "ftlm")
    thermo_label = {
        "full": "full ED spectrum",
        "ftlm": "FTLM",
        "both": "full ED + FTLM",
    }.get(thermo_mode, thermo_mode)

    dEdT = np.stack([np.gradient(E[i], T, edge_order=2) for i in range(E.shape[0])])
    C_over_T = C / np.maximum(T, 1e-12)

    quad_path = Path(args.quad_npz) if args.quad_npz else (root / "quad_qh_twist_avg_analysis.npz")
    if args.quad_subdir != "quad_qh_static" and not args.quad_npz:
        alt = root / f"quad_qh_twist_avg_{args.quad_subdir}.npz"
        if alt.is_file():
            quad_path = alt
    has_quad = quad_path.is_file()
    qd = np.load(quad_path) if has_quad else None

    beta_mtpq, sz_mtpq = _try_load_mtpq_sz(root)

    out_base = Path(args.out) if args.out else (root / "fig_jpm03_thermo_magnetostriction")
    out_base.parent.mkdir(parents=True, exist_ok=True)

    n_rows = 5 if has_quad else 3
    fig = plt.figure(figsize=(12.0, 3.1 * n_rows), constrained_layout=True)
    gs = fig.add_gridspec(n_rows, 2, height_ratios=[1.0] * n_rows)

    colors = plt.cm.viridis(np.linspace(0.15, 0.85, max(len(j3), 1)))

    ax_c = fig.add_subplot(gs[0, 0])
    ax_s = fig.add_subplot(gs[0, 1])
    ax_e = fig.add_subplot(gs[1, 0])
    ax_ct = fig.add_subplot(gs[1, 1])
    ax_dedt = fig.add_subplot(gs[2, 0])
    ax_spin = fig.add_subplot(gs[2, 1])

    _plot_j3_family(
        ax_c, T, C, Ce, j3, colors,
        ylabel=r"$C$",
        title=rf"Specific heat (twist avg, {thermo_label})",
        t_min=t_min,
    )
    _plot_j3_family(
        ax_s, T, S, None, j3, colors,
        ylabel=r"$S$",
        title=r"Entropy density",
        t_min=t_min,
    )
    _plot_j3_family(
        ax_e, T, E, None, j3, colors,
        ylabel=r"$\langle e\rangle$ (intensive)",
        title=r"Energy density",
        t_min=t_min,
    )
    _plot_j3_family(
        ax_ct, T, C_over_T, None, j3, colors,
        ylabel=r"$C/T$",
        title=r"$C/T$ (illustrative thermodynamic scale)",
        t_min=t_min,
    )

    for i, jv in enumerate(j3):
        c = colors[i] if len(j3) > 1 else "tab:blue"
        lab = rf"$J_3={float(jv):.3g}$"
        ax_dedt.plot(T, dEdT[i], color=c, lw=1.4, ls="-", label=lab + r" $\mathrm{d}\langle e\rangle/\mathrm{d}T$")
        ax_dedt.plot(T, C[i], color=c, lw=1.0, ls=":", alpha=0.75)
    ax_dedt.plot([], [], "k:", lw=1.2, label=r"$C$ (dotted)")
    _style_log_T(ax_dedt, t_min)
    ax_dedt.set_ylabel(r"$\mathrm{d}\langle e\rangle/\mathrm{d}T$")
    ax_dedt.set_title(r"Energy slope vs $T$ (compare to $C$)")
    ax_dedt.legend(frameon=True, fontsize=7)
    ax_dedt.grid(True, which="both", alpha=0.35)

    if beta_mtpq is not None and sz_mtpq is not None:
        T_m = 1.0 / np.maximum(beta_mtpq, 1e-12)
        ax_spin.plot(T_m, sz_mtpq.real, "o-", ms=2.5, lw=1.0, label=r"$\langle S^z\rangle$ (mTPQ)")
        _style_log_T(ax_spin, float(np.min(T_m[T_m > 0])))
        ax_spin.set_ylabel(r"$\langle S^z\rangle$")
        ax_spin.set_title("Spin (mTPQ fluctuation file)")
        ax_spin.legend(fontsize=8)
        ax_spin.grid(True, which="both", alpha=0.35)
    elif has_quad:
        ax_spin.axis("off")
    else:
        ax_spin.axis("off")
        ax_spin.text(
            0.5,
            0.55,
            "No quadrupole $\\alpha_Q$ data.\n\n"
            "Run ``run_quad_static_ftlm_scan.py`` and\n"
            "``analyze_jpm03_quad_twist_avg.py`` on this scan root.",
            ha="center",
            va="center",
            fontsize=10,
            transform=ax_spin.transAxes,
        )

    if has_quad and qd is not None:
        Tq = qd["T"]
        j3q = qd["j3"]
        tq_min = float(np.min(Tq[Tq > 0])) if np.any(Tq > 0) else t_min
        ax_eg = fig.add_subplot(gs[3, :])
        ax_t2g = fig.add_subplot(gs[4, :])

        ax_eg.cla()
        linestyles = ["-", "--"]
        for i, jv in enumerate(j3q):
            c = colors[i] if len(j3q) > 1 else "tab:blue"
            for ki, key in enumerate(ALPHA_EG_KEYS):
                err_key = f"{key}_err" if f"{key}_err" in qd.files else None
                y = qd[key][i]
                ax_eg.plot(
                    Tq,
                    y,
                    color=c,
                    lw=1.6,
                    ls=linestyles[ki % len(linestyles)],
                    label=rf"{ALPHA_LABELS[key]}, $J_3={float(jv):.3g}$",
                )
                if err_key:
                    ye = qd[err_key][i]
                    if np.any(ye > 0):
                        ax_eg.fill_between(Tq, y - ye, y + ye, color=c, alpha=0.12, lw=0)
        _style_log_T(ax_eg, tq_min)
        ax_eg.set_ylabel(r"$\alpha_Q$")
        ax_eg.set_title(
            r"Eg quadrupole $\alpha_Q=(\langle QH\rangle-\langle Q\rangle\langle H\rangle)/T^2$ "
            r"(static connected Q–H, twist avg)"
        )
        ax_eg.legend(frameon=True, fontsize=7, ncol=2)
        ax_eg.grid(True, which="both", alpha=0.35)

        ax_t2g.cla()
        linestyles_t = ["-", "--", "-."]
        for i, jv in enumerate(j3q):
            c = colors[i] if len(j3q) > 1 else "tab:blue"
            for ki, key in enumerate(ALPHA_T2G_KEYS):
                err_key = f"{key}_err" if f"{key}_err" in qd.files else None
                y = qd[key][i]
                ax_t2g.plot(
                    Tq,
                    y,
                    color=c,
                    lw=1.6,
                    ls=linestyles_t[ki % len(linestyles_t)],
                    label=rf"{ALPHA_LABELS[key]}, $J_3={float(jv):.3g}$",
                )
                if err_key:
                    ye = qd[err_key][i]
                    if np.any(ye > 0):
                        ax_t2g.fill_between(Tq, y - ye, y + ye, color=c, alpha=0.12, lw=0)
        _style_log_T(ax_t2g, tq_min)
        ax_t2g.set_ylabel(r"$\alpha_Q$")
        ax_t2g.set_title(
            r"$T_{2g}$ quadrupole $\alpha_Q$ (static connected Q–H, twist avg)"
        )
        ax_t2g.legend(frameon=True, fontsize=7, ncol=2)
        ax_t2g.grid(True, which="both", alpha=0.35)

    sym_note = "streaming ``--symm``" if use_symm else "no ``--symm``"
    quad_note = " + $\\alpha_Q$" if has_quad else ""
    fig.suptitle(
        rf"16-site pyrochlore $(1\!\times\!1\!\times\!1)$, $J_\pm={Jpm:+.2g}\,J_{{zz}}$, "
        rf"twist avg ({thermo_label}{quad_note}), {sym_note}",
        fontsize=11,
    )

    for ext in (".pdf", ".png"):
        p = out_base.with_suffix(ext)
        fig.savefig(p, dpi=160)
        print(f"wrote {p}")
    plt.close(fig)


if __name__ == "__main__":
    main()
