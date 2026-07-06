#!/usr/bin/env python3
"""Plot H-decomposition breakdown for each source-field case.

For every result.npz in the H_decomp_campaign directories, generate a
1×3 figure showing:
  col 0 – ⟨Q_source⟩(T)
  col 1 – C(T) specific heat
  col 2 – α_Q decomposed: α_total, α_{Q,Ising}, α_{Q,pm}, α_{Q,J3} (if present)

Output: H_decomp_campaign/plots/ next to each campaign root.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]

# Map source_key → operator label for titles / axes
_LABEL = {
    "Eg_Q1":   r"$E_g\,Q_1$",
    "Eg_Q2":   r"$E_g\,Q_2$",
    "T2g_Q_xz": r"$T_{2g}\,Q_{xz}$",
    "T2g_Q_xy": r"$T_{2g}\,Q_{xy}$",
    "T2g_Q_yz": r"$T_{2g}\,Q_{yz}$",
}

# Component display names and colours
_COMP_STYLE = {
    "H_Ising": dict(label=r"$\alpha_{Q,\mathrm{Ising}}$", color="#e6194b", ls="--",  lw=1.4),
    "H_pm":    dict(label=r"$\alpha_{Q,\pm}$",            color="#3cb44b", ls="-.",  lw=1.4),
    "H_J3":    dict(label=r"$\alpha_{Q,J_3}$",            color="#4363d8", ls=":",   lw=1.4),
}
_TOTAL_STYLE = dict(color="black", lw=2.0, ls="-", label=r"$\alpha_Q$ (total)")


def _jpm_label(path: Path) -> str:
    """Extract a human-readable title from the scan root directory name."""
    d = path.name
    replacements = {
        "jpm_pos004_zeroflux_j3zero": r"$J_{\pm}=+0.04$ (0-flux)",
        "jpm_pos002_zeroflux_j3zero": r"$J_{\pm}=+0.02$ (0-flux)",
        "jpm_neg002_piflux_j3zero":   r"$J_{\pm}=-0.02$ ($\pi$-flux)",
        "jpm_neg004_piflux_j3zero":   r"$J_{\pm}=-0.04$ ($\pi$-flux)",
        "jpm01_j3_scan_symm_light":   r"$J_{\pm}=-0.1$",
        "jpm02_j3_scan_symm_light":   r"$J_{\pm}=-0.2$",
        "jpm03_j3_verify_pbc_qsus":   r"$J_{\pm}=-0.3$",
    }
    return replacements.get(d, d)


def _j3_label(tag: str) -> str:
    if "j3_0.08" in tag:
        return r"$J_3=0.08$"
    return r"$J_3=0$"


def plot_one(npz_path: Path, out_dir: Path) -> Path:
    d = np.load(npz_path, allow_pickle=True)
    T = d["temperatures"]

    # Identify which operator was sourced
    src_keys = [k for k in d.keys() if k.startswith("source_")]
    if not src_keys:
        print(f"  [skip] no source key in {npz_path}")
        return None
    # Use the first (usually only) source key
    src_key = src_keys[0]
    q_name = src_key[len("source_"):]   # e.g. "Eg_Q1"

    q_label = _LABEL.get(q_name, q_name)
    lambda_val = float(d[src_key])

    # Find available H components for this Q
    comp_alpha = {}
    for comp in ["H_Ising", "H_pm", "H_J3"]:
        key = f"{q_name}_x_{comp}_alpha"
        if key in d:
            comp_alpha[comp] = d[key]

    # Total alpha and expectation
    alpha_total = d.get(f"{q_name}_alpha", None)
    q_expect    = d.get(f"{q_name}_expect", None)
    cv          = d["specific_heat"]

    # Scan-root label
    scan_root  = npz_path.parents[2]  # …/H_decomp_campaign/TAG/result.npz  → 2 up = scan_root
    tag        = npz_path.parent.name
    jpm_lbl    = _jpm_label(scan_root)
    j3_lbl     = _j3_label(tag)
    title_base = f"{jpm_lbl},  {j3_lbl},  source {q_label}  (λ={lambda_val:.0e})"

    # ---- figure -----------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    fig.suptitle(title_base, fontsize=11, y=1.01)

    # --- panel 0: ⟨Q_source⟩ ---
    ax = axes[0]
    if q_expect is not None:
        ax.plot(T, q_expect, color="steelblue", lw=1.8)
    ax.set_xscale("log")
    ax.set_xlabel(r"$T / J_{zz}$")
    ax.set_ylabel(rf"$\langle {q_label[1:-1]} \rangle$")
    ax.set_title(r"Expectation value $\langle Q \rangle$")
    ax.axhline(0, color="gray", lw=0.6, ls="--")
    ax.xaxis.set_major_formatter(ticker.LogFormatterSciNotation(minor_thresholds=(2, 0.5)))

    # --- panel 1: C(T) ---
    ax = axes[1]
    ax.plot(T, cv, color="darkorange", lw=1.8)
    ax.set_xscale("log")
    ax.set_xlabel(r"$T / J_{zz}$")
    ax.set_ylabel(r"$C(T) / k_B$")
    ax.set_title("Specific heat")
    ax.xaxis.set_major_formatter(ticker.LogFormatterSciNotation(minor_thresholds=(2, 0.5)))

    # --- panel 2: α decomposition ---
    ax = axes[2]
    if alpha_total is not None:
        ax.plot(T, alpha_total, **_TOTAL_STYLE)
    for comp, arr in comp_alpha.items():
        style = dict(_COMP_STYLE[comp])
        ax.plot(T, arr, **style)

    # Show sum of components as dashed check
    if comp_alpha:
        arr_sum = sum(comp_alpha.values())
        ax.plot(T, arr_sum, color="gray", lw=0.8, ls=":", alpha=0.7, label="Σ components")

    ax.set_xscale("log")
    ax.axhline(0, color="gray", lw=0.6, ls="--")
    ax.set_xlabel(r"$T / J_{zz}$")
    ax.set_ylabel(r"$\alpha_Q \cdot J_{zz}$")
    ax.set_title(r"$\alpha_Q$ decomposition")
    ax.legend(fontsize=8, loc="best")
    ax.xaxis.set_major_formatter(ticker.LogFormatterSciNotation(minor_thresholds=(2, 0.5)))

    for ax in axes:
        ax.set_xlim(T.min(), T.max())
        ax.grid(True, which="both", ls=":", lw=0.4, alpha=0.5)

    fig.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    fname = f"{tag}.pdf"
    out_path = out_dir / fname
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {out_path.relative_to(ROOT)}")
    return out_path


# ---------------------------------------------------------------------------
# Collect all H_decomp result.npz files
# ---------------------------------------------------------------------------
def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(ROOT / "twist_qsi_demo" / "output"),
                    help="output root to search")
    args = ap.parse_args()

    search_root = Path(args.root)
    npz_files = sorted(search_root.rglob("H_decomp_campaign/*/result.npz"))

    if not npz_files:
        print("No H_decomp result.npz files found.")
        sys.exit(1)

    print(f"Found {len(npz_files)} H_decomp result files:")
    for p in npz_files:
        print(f"  {p.relative_to(ROOT)}")
    print()

    # Group by scan_root → put plots in scan_root/H_decomp_campaign/plots/
    saved = []
    for npz_path in npz_files:
        scan_root = npz_path.parents[2]
        out_dir   = scan_root / "H_decomp_campaign" / "plots"
        out = plot_one(npz_path, out_dir)
        if out:
            saved.append(out)

    print(f"\nDone. {len(saved)} plots saved.")
    # Also write a quick summary of plot locations
    for p in saved:
        print(f"  {p.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
