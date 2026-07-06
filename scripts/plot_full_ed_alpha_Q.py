#!/usr/bin/env python3
"""Plot exact full-ED quadrupolar thermal expansion α_Q from run_full_ed_quad_qh_J3.py."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

EG_OPS = [("Eg_Q1", r"$Q_1$ (Eg)"), ("Eg_Q2", r"$Q_2$ (Eg)")]
T2G_OPS = [
    ("T2g_Q_xy", r"$Q_{xy}$"),
    ("T2g_Q_xz", r"$Q_{xz}$"),
    ("T2g_Q_yz", r"$Q_{yz}$"),
]


def _style_log_T(ax, t_min: float) -> None:
    ax.set_xscale("log")
    ax.set_xlim(left=max(t_min, 1e-4))
    ax.set_xlabel(r"$T\,/\,|J_{zz}|$")


def load_case(npz_path: Path, label: str) -> dict:
    d = np.load(npz_path)
    return {
        "label": label,
        "T": d["temperatures"],
        "alpha": {op: d[f"{op}_alpha"] for op, _ in EG_OPS + T2G_OPS},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--j3-0", type=Path, required=True)
    ap.add_argument("--j3-08", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    cases = [
        load_case(args.j3_0, r"$J_3=0$"),
        load_case(args.j3_08, r"$J_3=0.08$"),
    ]
    T = cases[0]["T"]
    t_min = float(np.min(T[T > 0]))

    fig, axes = plt.subplots(2, 1, figsize=(8.5, 7), sharex=True)
    ax_eg, ax_t2g = axes
    colors = ["#1f77b4", "#d62728"]
    lstyles = ["-", "--", "-."]

    for panel, ops, ax, title in [
        (0, EG_OPS, ax_eg, r"Eg quadrupole $\alpha_Q$"),
        (1, T2G_OPS, ax_t2g, r"$T_{2g}$ quadrupole $\alpha_Q$"),
    ]:
        for ci, case in enumerate(cases):
            c = colors[ci]
            for li, (op, op_label) in enumerate(ops):
                y = case["alpha"][op]
                ax.plot(
                    case["T"],
                    y,
                    color=c,
                    lw=1.8,
                    ls=lstyles[li % len(lstyles)],
                    label=rf"{op_label}, {case['label']}",
                )
        _style_log_T(ax, t_min)
        ax.set_ylabel(r"$\alpha_Q = (\langle QH\rangle_c)/T^2$")
        ax.set_title(title + r" — exact full ED (16-site, PBC)")
        ax.axhline(0.0, color="k", lw=0.6, alpha=0.4)
        ax.legend(frameon=True, fontsize=8, ncol=2)
        ax.grid(True, which="both", alpha=0.35)

    fig.suptitle(
        r"Quadrupolar thermal expansion: $\alpha_Q=(\langle QH\rangle-\langle Q\rangle\langle H\rangle)/T^2$",
        fontsize=11,
    )
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for ext in (".png", ".pdf"):
        p = args.out.with_suffix(ext)
        fig.savefig(p, dpi=160, bbox_inches="tight")
        print(f"wrote {p}")
    plt.close(fig)


if __name__ == "__main__":
    main()
