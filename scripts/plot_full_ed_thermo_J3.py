#!/usr/bin/env python3
"""Plot exact thermodynamics (E, C, S) vs T from full-ED npz files for the
J3 grid 0.00, 0.02, 0.04, 0.06, 0.08.

Reads NPZ outputs of ``run_full_ed_thermo_J3.py``.  Missing files are
skipped so the script can be re-run incrementally while the batch is
in flight.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

DEFAULT_ROOT = Path(
    "twist_qsi_demo/output/jpm03_j3_verify_pbc_qsus"
)
J3_GRID = ["0.0000", "0.0200", "0.0400", "0.0600", "0.0800"]
PHI = "phi_0.000pi_0.000pi_0.000pi"
N_SITES = 16


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--out-prefix", type=Path,
                    default=DEFAULT_ROOT / "full_ed_thermo")
    args = ap.parse_args()

    cases = []
    for j3 in J3_GRID:
        p = args.root / f"J3_{j3}" / PHI / "ham" / "full_thermo_pyed" / "result.npz"
        if not p.exists():
            print(f"  skipping J3={j3} (no result yet at {p})")
            continue
        d = np.load(p)
        cases.append((float(j3), d))
        print(f"  loaded J3={j3}: E0={float(d['eigenvalues'][0]):.6f}")

    if not cases:
        print("no results yet")
        return

    cmap = plt.colormaps.get_cmap("viridis")
    # Sample colors evenly across the J3 grid
    n = len(cases)
    colors = [cmap(0.05 + 0.9 * (i / max(n - 1, 1))) for i in range(n)]

    # ---- E, C, S vs T ----
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharex=True)
    for (j3, d), c in zip(cases, colors):
        T = d["temperatures"]
        E = d["energy"]
        C = d["specific_heat"]
        S = d["entropy"]
        label = f"$J_3={j3:.2f}$"
        axes[0].plot(T, E / N_SITES, color=c, lw=1.6, label=label)
        axes[1].plot(T, C / N_SITES, color=c, lw=1.6, label=label)
        axes[2].plot(T, S / N_SITES, color=c, lw=1.6, label=label)

    axes[0].set_ylabel("$E/N$")
    axes[1].set_ylabel("$C/N$")
    axes[2].set_ylabel("$S/N$")
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel("$T/J_{zz}$")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8, loc="best")
    axes[2].axhline(np.log(2), color="k", lw=0.6, ls=":")
    axes[2].text(2.5, np.log(2) * 0.98, r"$\ln 2$", fontsize=9,
                 va="top", ha="left")
    fig.suptitle("Exact full-ED thermodynamics, 16-site pyrochlore (PBC)")
    fig.tight_layout()
    out = args.out_prefix.with_suffix(".png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Wrote {out}")

    # ---- Specific heat closeup ----
    fig2, ax = plt.subplots(figsize=(7.5, 4.6))
    for (j3, d), c in zip(cases, colors):
        ax.plot(d["temperatures"], d["specific_heat"] / N_SITES, color=c,
                lw=1.8, label=f"$J_3={j3:.2f}$")
    ax.set_xscale("log")
    ax.set_xlabel("$T/J_{zz}$")
    ax.set_ylabel("$C/N$")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=9)
    ax.set_title("Specific heat per site, exact ED, 16-site pyrochlore")
    fig2.tight_layout()
    out2 = args.out_prefix.with_name(args.out_prefix.name + "_C").with_suffix(".png")
    fig2.savefig(out2, dpi=150, bbox_inches="tight")
    print(f"Wrote {out2}")

    # ---- Eigenvalue distribution / density of states ----
    fig3, ax = plt.subplots(figsize=(7.5, 4.6))
    for (j3, d), c in zip(cases, colors):
        ev = d["eigenvalues"]
        bins = np.linspace(ev.min(), ev.max(), 200)
        ax.hist(ev, bins=bins, histtype="step", color=c, lw=1.4,
                density=True, label=f"$J_3={j3:.2f}$")
    ax.set_xlabel("$E$")
    ax.set_ylabel("$\\rho(E)$")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    ax.set_title("Density of states (exact 16-site spectrum)")
    fig3.tight_layout()
    out3 = args.out_prefix.with_name(args.out_prefix.name + "_DOS").with_suffix(".png")
    fig3.savefig(out3, dpi=150, bbox_inches="tight")
    print(f"Wrote {out3}")


if __name__ == "__main__":
    main()
