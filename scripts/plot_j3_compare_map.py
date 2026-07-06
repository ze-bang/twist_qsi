#!/usr/bin/env python3
"""6×3 comparison grid: J3=0 vs J3=0.08 source-field responses.

Layout (rows):
  0 – clean J3=0.08 baseline (no source)
  1 – clean J3=0   baseline (no source)
  2 – Eg source, J3=0.08, Jpm=-0.3/-0.2/-0.1 overlaid
  3 – Eg source, J3=0,    Jpm=-0.3/-0.2/-0.1 overlaid
  4 – T2g source, J3=0.08, Jpm=-0.3/-0.2/-0.1 overlaid
  5 – T2g source, J3=0,    Jpm=-0.3/-0.2/-0.1 overlaid

Columns: Eg α_Q | T2g α_Q | C(T)
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/home/pc_linux/exact_diagonalization_clean/twist_qsi_demo/output")

# ── data paths ──────────────────────────────────────────────────────────────
BASELINE_J308 = ROOT / "jpm03_j3_verify_pbc_qsus/J3_0.0800/phi_0.000pi_0.000pi_0.000pi/ham/full_thermo_quad/result.npz"
BASELINE_J300 = ROOT / "jpm03_j3_verify_pbc_qsus/J3_0.0000/phi_0.000pi_0.000pi_0.000pi/ham/full_thermo_quad/result.npz"

EG_J308 = {
    "-0.3": ROOT / "jpm03_j3_verify_pbc_qsus/source_field_campaign/EgQ1_1e-3/result.npz",
    "-0.2": ROOT / "jpm02_j3_scan_symm_light/source_field_campaign/EgQ1_1e-3/result.npz",
    "-0.1": ROOT / "jpm01_j3_scan_symm_light/source_field_campaign/EgQ1_1e-3/result.npz",
}
EG_J300 = {
    "-0.3": ROOT / "jpm03_j3_verify_pbc_qsus/source_field_campaign_j3zero/Jpm03_J3zero_EgQ1/result.npz",
    "-0.2": ROOT / "jpm02_j3_scan_symm_light/source_field_campaign_j3zero/Jpm02_J3zero_EgQ1/result.npz",
    "-0.1": ROOT / "jpm01_j3_scan_symm_light/source_field_campaign_j3zero/Jpm01_J3zero_EgQ1/result.npz",
}
T2G_J308 = {
    "-0.3": ROOT / "jpm03_j3_verify_pbc_qsus/source_field_campaign/T2gXZ_1e-3/result.npz",
    "-0.2": ROOT / "jpm02_j3_scan_symm_light/source_field_campaign/T2gXZ_1e-3/result.npz",
    "-0.1": ROOT / "jpm01_j3_scan_symm_light/source_field_campaign/T2gXZ_1e-3/result.npz",
}
T2G_J300 = {
    "-0.3": ROOT / "jpm03_j3_verify_pbc_qsus/source_field_campaign_j3zero/Jpm03_J3zero_T2gXZ/result.npz",
    "-0.2": ROOT / "jpm02_j3_scan_symm_light/source_field_campaign_j3zero/Jpm02_J3zero_T2gXZ/result.npz",
    "-0.1": ROOT / "jpm01_j3_scan_symm_light/source_field_campaign_j3zero/Jpm01_J3zero_T2gXZ/result.npz",
}

EG_OPS  = [("Eg_Q1", r"$Q_1$"), ("Eg_Q2", r"$Q_2$")]
T2G_OPS = [("T2g_Q_xy", r"$Q_{xy}$"), ("T2g_Q_xz", r"$Q_{xz}$"), ("T2g_Q_yz", r"$Q_{yz}$")]

JPM_COLORS = {"-0.3": "#1f77b4", "-0.2": "#ff7f0e", "-0.1": "#d62728"}

OUT_DIR = ROOT / "jpm03_j3_verify_pbc_qsus/source_field_campaign/plots"


def load(path: Path) -> dict:
    d = np.load(path)
    return {
        "T": d["temperatures"],
        "alpha": {op: d[f"{op}_alpha"] for op, _ in EG_OPS + T2G_OPS},
        "C": d["specific_heat"],
    }


def style(ax, *, zero_line=True):
    ax.set_xscale("log")
    ax.set_xlim(left=5e-3)
    if zero_line:
        ax.axhline(0, color="k", lw=0.5, alpha=0.4)
    ax.grid(True, which="both", alpha=0.25)


def plot_baseline_row(axes, title: str, data: dict):
    ax_eg, ax_t2g, ax_c = axes
    eg_ls = ["-", "--"]
    t2g_ls = ["-", "--", "-."]
    eg_col = ["#1f77b4", "#d62728"]
    t2g_col = ["#1f77b4", "#d62728", "#2ca02c"]
    for i, (op, lbl) in enumerate(EG_OPS):
        ax_eg.plot(data["T"], data["alpha"][op], color=eg_col[i], ls=eg_ls[i], lw=2, label=lbl)
    for i, (op, lbl) in enumerate(T2G_OPS):
        ax_t2g.plot(data["T"], data["alpha"][op], color=t2g_col[i], ls=t2g_ls[i], lw=2, label=lbl)
    ax_c.plot(data["T"], data["C"], color="k", lw=2)
    for ax in (ax_eg, ax_t2g):
        style(ax)
        ax.legend(fontsize=8)
    style(ax_c, zero_line=False)
    ax_eg.set_title(title + r" — $E_g\;\alpha_Q$", fontsize=10)
    ax_t2g.set_title(title + r" — $T_{2g}\;\alpha_Q$", fontsize=10)
    ax_c.set_title(title + r" — $C(T)$", fontsize=10)


def plot_overlay_row(axes, title: str, dataset: dict[str, Path], src_op: str):
    """Overlay Jpm=-0.3/-0.2/-0.1 curves; src_op='Eg_Q1' or 'T2g_Q_xz'."""
    ax_eg, ax_t2g, ax_c = axes
    eg_ls  = {"Eg_Q1": "-", "Eg_Q2": "--"}
    t2g_ls = {"T2g_Q_xy": "-", "T2g_Q_xz": "--", "T2g_Q_yz": "-."}
    for jpm, path in dataset.items():
        c = JPM_COLORS[jpm]
        d = load(path)
        for op, lbl in EG_OPS:
            ax_eg.plot(d["T"], d["alpha"][op], color=c, ls=eg_ls[op], lw=1.8,
                       label=f"$J_{{\\pm}}={jpm}$ {lbl}" if op == "Eg_Q1" else None)
        for op, lbl in T2G_OPS:
            ax_t2g.plot(d["T"], d["alpha"][op], color=c, ls=t2g_ls[op], lw=1.8,
                        label=f"$J_{{\\pm}}={jpm}$ {lbl}" if op == "T2g_Q_xz" else None)
        ax_c.plot(d["T"], d["C"], color=c, lw=1.8, label=f"$J_{{\\pm}}={jpm}$")
    for ax in (ax_eg, ax_t2g):
        style(ax)
        ax.legend(fontsize=7, ncol=1)
    style(ax_c, zero_line=False)
    ax_c.legend(fontsize=8)
    ax_eg.set_title(title + r" — $E_g\;\alpha_Q$", fontsize=10)
    ax_t2g.set_title(title + r" — $T_{2g}\;\alpha_Q$", fontsize=10)
    ax_c.set_title(title + r" — $C(T)$", fontsize=10)


def main():
    fig, axes = plt.subplots(6, 3, figsize=(16, 22), sharex=True)

    # ── row 0: clean J3=0.08 ───────────────────────────────────────────────
    plot_baseline_row(axes[0], r"Clean $J_3=0.08,\;J_{\pm}=-0.3$", load(BASELINE_J308))

    # ── row 1: clean J3=0 ─────────────────────────────────────────────────
    plot_baseline_row(axes[1], r"Clean $J_3=0,\;J_{\pm}=-0.3$", load(BASELINE_J300))

    # ── row 2: Eg source, J3=0.08 ─────────────────────────────────────────
    plot_overlay_row(axes[2],
                     r"$E_g$ source ($10^{-3}Q_1$), $J_3=0.08$",
                     EG_J308, "Eg_Q1")

    # ── row 3: Eg source, J3=0 ────────────────────────────────────────────
    plot_overlay_row(axes[3],
                     r"$E_g$ source ($10^{-3}Q_1$), $J_3=0$",
                     EG_J300, "Eg_Q1")

    # ── row 4: T2g source, J3=0.08 ────────────────────────────────────────
    plot_overlay_row(axes[4],
                     r"$T_{2g}$ source ($10^{-3}Q_{xz}$), $J_3=0.08$",
                     T2G_J308, "T2g_Q_xz")

    # ── row 5: T2g source, J3=0 ───────────────────────────────────────────
    plot_overlay_row(axes[5],
                     r"$T_{2g}$ source ($10^{-3}Q_{xz}$), $J_3=0$",
                     T2G_J300, "T2g_Q_xz")

    for col in range(3):
        axes[-1, col].set_xlabel(r"$T\,/\,|J_{zz}|$", fontsize=11)

    for row in range(6):
        axes[row, 0].set_ylabel(r"$\alpha_Q$", fontsize=9)
        axes[row, 2].set_ylabel(r"$C(T)$", fontsize=9)

    fig.suptitle(
        r"Induced quadrupolar response: $J_3=0.08$ vs $J_3=0$ comparison"
        "\n"
        r"(source $\lambda=10^{-3}$, $J_{zz}=1$)",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in (".png", ".pdf"):
        p = OUT_DIR / f"j3_compare_6x3{ext}"
        fig.savefig(p, dpi=180, bbox_inches="tight")
        print(f"wrote {p}")
    plt.close(fig)


if __name__ == "__main__":
    main()
