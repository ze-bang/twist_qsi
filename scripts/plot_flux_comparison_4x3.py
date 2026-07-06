#!/usr/bin/env python3
"""4x5 comparison grid.

Layout
------
Row 0  Clean baseline: Jpm=-0.3, J3=0,    no source
Row 1  Clean baseline: Jpm=-0.3, J3=0.08, no source
Row 2  Eg  source (lambda=1e-3), J3=0, Jpm=-0.3/-0.2/-0.1/-0.04/-0.02/+0.02/+0.04 overlaid
Row 3  T2g source (lambda=1e-3), J3=0, Jpm=-0.3/-0.2/-0.1/-0.04/-0.02/+0.02/+0.04 overlaid

Columns
-------
Col 0  Eg  alpha_Q    (all Eg ops for baseline rows; alpha_Eg for overlay rows)
Col 1  Eg  expect     (all Eg ops for baseline rows; <Q1> for overlay rows)
Col 2  T2g alpha_Q    (all T2g ops for baseline rows; alpha_T2g for overlay rows)
Col 3  T2g expect     (all T2g ops for baseline rows; <Qxz> for overlay rows)
Col 4  C(T)
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/home/pc_linux/exact_diagonalization_clean/twist_qsi_demo/output")

# ── data paths ─────────────────────────────────────────────────────────────
BASELINE_J300 = ROOT / "jpm03_j3_verify_pbc_qsus/J3_0.0000/phi_0.000pi_0.000pi_0.000pi/ham/full_thermo_quad/result.npz"
BASELINE_J308 = ROOT / "jpm03_j3_verify_pbc_qsus/J3_0.0800/phi_0.000pi_0.000pi_0.000pi/ham/full_thermo_quad/result.npz"

# J3=0 Eg-source cases: (label, color, linestyle, path)
EG_SRC = [
    (r"$J_\pm=-0.3$  ($\pi$-flux)", "#1f77b4", "-",
     ROOT / "jpm03_j3_verify_pbc_qsus/source_field_campaign_j3zero/Jpm03_J3zero_EgQ1/result.npz"),
    (r"$J_\pm=-0.2$  ($\pi$-flux)", "#ff7f0e", "-",
     ROOT / "jpm02_j3_scan_symm_light/source_field_campaign_j3zero/Jpm02_J3zero_EgQ1/result.npz"),
    (r"$J_\pm=-0.1$  ($\pi$-flux)", "#d62728", "-",
     ROOT / "jpm01_j3_scan_symm_light/source_field_campaign_j3zero/Jpm01_J3zero_EgQ1/result.npz"),
    (r"$J_\pm=-0.04$ ($\pi$-flux)", "#9467bd", "-",
     ROOT / "jpm_neg004_piflux_j3zero/source_field_campaign/Eg_Q1_1e-3/result.npz"),
    (r"$J_\pm=-0.02$ ($\pi$-flux)", "#e377c2", "-",
     ROOT / "jpm_neg002_piflux_j3zero/source_field_campaign/Eg_Q1_1e-3/result.npz"),
    (r"$J_\pm=+0.02$ (0-flux)",     "#bcbd22", "--",
     ROOT / "jpm_pos002_zeroflux_j3zero/source_field_campaign/Eg_Q1_1e-3/result.npz"),
    (r"$J_\pm=+0.04$ (0-flux)",     "#2ca02c", "--",
     ROOT / "jpm_pos004_zeroflux_j3zero/source_field_campaign/Eg_Q1_1e-3/result.npz"),
]

# J3=0 T2g-source cases
T2G_SRC = [
    (r"$J_\pm=-0.3$  ($\pi$-flux)", "#1f77b4", "-",
     ROOT / "jpm03_j3_verify_pbc_qsus/source_field_campaign_j3zero/Jpm03_J3zero_T2gXZ/result.npz"),
    (r"$J_\pm=-0.2$  ($\pi$-flux)", "#ff7f0e", "-",
     ROOT / "jpm02_j3_scan_symm_light/source_field_campaign_j3zero/Jpm02_J3zero_T2gXZ/result.npz"),
    (r"$J_\pm=-0.1$  ($\pi$-flux)", "#d62728", "-",
     ROOT / "jpm01_j3_scan_symm_light/source_field_campaign_j3zero/Jpm01_J3zero_T2gXZ/result.npz"),
    (r"$J_\pm=-0.04$ ($\pi$-flux)", "#9467bd", "-",
     ROOT / "jpm_neg004_piflux_j3zero/source_field_campaign/T2g_Q_xz_1e-3/result.npz"),
    (r"$J_\pm=-0.02$ ($\pi$-flux)", "#e377c2", "-",
     ROOT / "jpm_neg002_piflux_j3zero/source_field_campaign/T2g_Q_xz_1e-3/result.npz"),
    (r"$J_\pm=+0.02$ (0-flux)",     "#bcbd22", "--",
     ROOT / "jpm_pos002_zeroflux_j3zero/source_field_campaign/T2g_Q_xz_1e-3/result.npz"),
    (r"$J_\pm=+0.04$ (0-flux)",     "#2ca02c", "--",
     ROOT / "jpm_pos004_zeroflux_j3zero/source_field_campaign/T2g_Q_xz_1e-3/result.npz"),
]

EG_OPS  = [("Eg_Q1",    r"$Q_1$"),    ("Eg_Q2",    r"$Q_2$")]
T2G_OPS = [("T2g_Q_xy", r"$Q_{xy}$"), ("T2g_Q_xz", r"$Q_{xz}$"), ("T2g_Q_yz", r"$Q_{yz}$")]

OUT_DIR = ROOT / "jpm03_j3_verify_pbc_qsus/source_field_campaign/plots"


# ── helpers ─────────────────────────────────────────────────────────────────
def load(path: Path) -> dict | None:
    if not path.exists():
        return None
    d = np.load(path)
    out = {"T": d["temperatures"], "C": d["specific_heat"]}
    for op, _ in EG_OPS + T2G_OPS:
        out[f"{op}_alpha"] = d[f"{op}_alpha"]
        out[f"{op}_expect"] = d[f"{op}_expect"]
    return out


def ax_style(ax, zero_line=True):
    ax.set_xscale("log")
    ax.set_xlim(1e-3, 5.0)
    if zero_line:
        ax.axhline(0, color="k", lw=0.5, alpha=0.4, zorder=0)
    ax.grid(True, which="both", alpha=0.2)
    ax.tick_params(labelsize=8)


def plot_baseline(axes, title: str, data: dict):
    ax_eg_a, ax_eg_q, ax_t2g_a, ax_t2g_q, ax_c = axes
    eg_cols = ["#1f77b4", "#d62728"]
    eg_ls   = ["-", "--"]
    t2g_cols = ["#1f77b4", "#d62728", "#2ca02c"]
    t2g_ls   = ["-", "--", "-."]
    for i, (op, lbl) in enumerate(EG_OPS):
        ax_eg_a.plot(data["T"], data[f"{op}_alpha"],
                     color=eg_cols[i], ls=eg_ls[i], lw=1.8, label=lbl)
        ax_eg_q.plot(data["T"], data[f"{op}_expect"],
                     color=eg_cols[i], ls=eg_ls[i], lw=1.8, label=lbl)
    for i, (op, lbl) in enumerate(T2G_OPS):
        ax_t2g_a.plot(data["T"], data[f"{op}_alpha"],
                      color=t2g_cols[i], ls=t2g_ls[i], lw=1.8, label=lbl)
        ax_t2g_q.plot(data["T"], data[f"{op}_expect"],
                      color=t2g_cols[i], ls=t2g_ls[i], lw=1.8, label=lbl)
    ax_c.plot(data["T"], data["C"], color="k", lw=2)
    for ax in (ax_eg_a, ax_eg_q, ax_t2g_a, ax_t2g_q):
        ax_style(ax)
        ax.legend(fontsize=8)
    ax_style(ax_c, zero_line=False)
    ax_eg_a.set_title(title + r"  —  $E_g\;\alpha_Q$", fontsize=9)
    ax_eg_q.set_title(title + r"  —  $E_g\;\langle Q\rangle$", fontsize=9)
    ax_t2g_a.set_title(title + r"  —  $T_{2g}\;\alpha_Q$", fontsize=9)
    ax_t2g_q.set_title(title + r"  —  $T_{2g}\;\langle Q\rangle$", fontsize=9)
    ax_c.set_title(title + r"  —  $C(T)$", fontsize=9)


def plot_overlay(axes, title: str, cases, src_irrep: str):
    """Overlay multiple Jpm cases. src_irrep: 'Eg' or 'T2g'."""
    ax_eg_a, ax_eg_q, ax_t2g_a, ax_t2g_q, ax_c = axes

    for label, color, lsbase, path in cases:
        d = load(path)
        if d is None:
            print(f"  [skip – not ready] {label}")
            continue

        ax_eg_a.plot(d["T"], d["Eg_Q1_alpha"],
                     color=color, ls=lsbase, lw=1.8, label=label)
        ax_eg_q.plot(d["T"], d["Eg_Q1_expect"],
                     color=color, ls=lsbase, lw=1.8, label=label)
        ax_t2g_a.plot(d["T"], d["T2g_Q_xz_alpha"],
                      color=color, ls=lsbase, lw=1.8, label=label)
        ax_t2g_q.plot(d["T"], d["T2g_Q_xz_expect"],
                      color=color, ls=lsbase, lw=1.8, label=label)

        ax_c.plot(d["T"], d["C"], color=color, ls=lsbase, lw=1.8)

    for ax in (ax_eg_a, ax_eg_q, ax_t2g_a, ax_t2g_q):
        ax_style(ax)
        ax.legend(fontsize=7, ncol=1)
    ax_style(ax_c, zero_line=False)

    ax_eg_a.set_title(title + r"  —  $E_g\;\alpha_{Q_1}$", fontsize=9)
    ax_eg_q.set_title(title + r"  —  $E_g\;\langle Q_1\rangle$", fontsize=9)
    ax_t2g_a.set_title(title + r"  —  $T_{2g}\;\alpha_{Q_{xz}}$", fontsize=9)
    ax_t2g_q.set_title(title + r"  —  $T_{2g}\;\langle Q_{xz}\rangle$", fontsize=9)
    ax_c.set_title(title + r"  —  $C(T)$", fontsize=9)


# ── main ────────────────────────────────────────────────────────────────────
def main():
    fig, axes = plt.subplots(4, 5, figsize=(24, 18), sharex=True)
    fig.subplots_adjust(hspace=0.38, wspace=0.32)

    # Row 0: clean Jpm=-0.3, J3=0
    plot_baseline(axes[0],
                  r"Clean $J_\pm=-0.3,\;J_3=0$",
                  load(BASELINE_J300))

    # Row 1: clean Jpm=-0.3, J3=0.08
    plot_baseline(axes[1],
                  r"Clean $J_\pm=-0.3,\;J_3=0.08$",
                  load(BASELINE_J308))

    # Row 2: Eg source, J3=0, all Jpm overlaid
    plot_overlay(axes[2],
                 r"$E_g$ source ($\lambda=10^{-3}Q_1$), $J_3=0$",
                 EG_SRC, "Eg")

    # Row 3: T2g source, J3=0, all Jpm overlaid
    plot_overlay(axes[3],
                 r"$T_{2g}$ source ($\lambda=10^{-3}Q_{xz}$), $J_3=0$",
                 T2G_SRC, "T2g")

    # Axis labels
    for col in range(5):
        axes[-1, col].set_xlabel(r"$T\;/\;|J_{zz}|$", fontsize=10)
    for row in range(4):
        axes[row, 0].set_ylabel(r"$\alpha_Q$", fontsize=9)
        axes[row, 1].set_ylabel(r"$\langle Q\rangle$", fontsize=9)
        axes[row, 2].set_ylabel(r"$\alpha_Q$", fontsize=9)
        axes[row, 3].set_ylabel(r"$\langle Q\rangle$", fontsize=9)
        axes[row, 4].set_ylabel(r"$C(T)$",     fontsize=9)

    # Row labels on the left
    row_labels = [
        r"Row 1: clean, $J_3=0$",
        r"Row 2: clean, $J_3=0.08$",
        r"Row 3: $E_g$ source",
        r"Row 4: $T_{2g}$ source",
    ]
    for row, lbl in enumerate(row_labels):
        axes[row, 0].annotate(lbl, xy=(-0.22, 0.5),
                              xycoords="axes fraction",
                              rotation=90, va="center", ha="center",
                              fontsize=8, color="0.4")

    fig.suptitle(
        r"Quadrupolar $\alpha_Q$, $\langle Q\rangle$, and $C(T)$: flux phase comparison"
        "\n"
        r"($J_{zz}=1$, source $\lambda=10^{-3}$, solid = $\pi$-flux, dashed = 0-flux)",
        fontsize=12,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in (".png", ".pdf"):
        p = OUT_DIR / f"flux_comparison_4x5{ext}"
        fig.savefig(p, dpi=180, bbox_inches="tight")
        print(f"wrote {p}")
    plt.close(fig)


if __name__ == "__main__":
    main()
