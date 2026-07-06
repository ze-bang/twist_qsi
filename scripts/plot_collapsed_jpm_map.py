#!/usr/bin/env python3
"""Collapsed 4x3 comparison grid for clean and source-field Jpm sweeps."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

EG_OPS = [("Eg_Q1", r"$Q_1$"), ("Eg_Q2", r"$Q_2$")]
T2G_OPS = [("T2g_Q_xy", r"$Q_{xy}$"), ("T2g_Q_xz", r"$Q_{xz}$"), ("T2g_Q_yz", r"$Q_{yz}$")]


def load_case(path: Path) -> dict:
    d = np.load(path)
    return {
        "T": d["temperatures"],
        "alpha": {op: d[f"{op}_alpha"] for op, _ in EG_OPS + T2G_OPS},
        "specific_heat": d["specific_heat"],
    }


def parse_case_spec(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise ValueError(f"invalid case spec '{spec}' (expected LABEL=PATH)")
    label, path = spec.rsplit("=", 1)
    label = label.strip()
    if not label:
        raise ValueError(f"invalid case spec '{spec}' (empty label)")
    return label, Path(path)


def style(ax, t_min: float, zero_line: bool = True) -> None:
    ax.set_xscale("log")
    ax.set_xlim(left=max(t_min, 1e-4))
    if zero_line:
        ax.axhline(0.0, color="k", lw=0.6, alpha=0.35)
    ax.grid(True, which="both", alpha=0.3)


def plot_single_row(axes_row, title: str, case: dict, t_min: float) -> None:
    eg_colors = ["#1f77b4", "#d62728"]
    t2g_colors = ["#1f77b4", "#d62728", "#2ca02c"]
    t2g_styles = ["-", "--", "-."]

    ax_eg, ax_t2g, ax_c = axes_row
    for i, (op, label) in enumerate(EG_OPS):
        ax_eg.plot(case["T"], case["alpha"][op], color=eg_colors[i], lw=2.0, ls="--" if i else "-", label=label)
    for i, (op, label) in enumerate(T2G_OPS):
        ax_t2g.plot(case["T"], case["alpha"][op], color=t2g_colors[i], lw=2.0, ls=t2g_styles[i], label=label)
    ax_c.plot(case["T"], case["specific_heat"], color="black", lw=2.0, label=r"$C(T)$")

    style(ax_eg, t_min)
    style(ax_t2g, t_min)
    style(ax_c, t_min, zero_line=False)
    ax_eg.set_ylabel(r"$\alpha_Q$")
    ax_t2g.set_ylabel(r"$\alpha_Q$")
    ax_c.set_ylabel(r"$C(T)$")
    ax_eg.set_title(title + r" - $E_g$")
    ax_t2g.set_title(title + r" - $T_{2g}$")
    ax_c.set_title(title + r" - specific heat")
    ax_eg.legend(frameon=True, fontsize=8)
    ax_t2g.legend(frameon=True, fontsize=8)
    ax_c.legend(frameon=True, fontsize=8)


def plot_overlay_row(axes_row, title: str, cases: list[tuple[str, dict]], t_min: float) -> None:
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd"]
    eg_styles = {"Eg_Q1": "-", "Eg_Q2": "--"}
    t2g_styles = {"T2g_Q_xy": "-", "T2g_Q_xz": "--", "T2g_Q_yz": "-."}

    ax_eg, ax_t2g, ax_c = axes_row
    for i, (label, case) in enumerate(cases):
        color = colors[i % len(colors)]
        for op, op_label in EG_OPS:
            ax_eg.plot(
                case["T"],
                case["alpha"][op],
                color=color,
                lw=2.0,
                ls=eg_styles[op],
                label=f"{label} {op_label}",
            )
        for op, op_label in T2G_OPS:
            ax_t2g.plot(
                case["T"],
                case["alpha"][op],
                color=color,
                lw=2.0,
                ls=t2g_styles[op],
                label=f"{label} {op_label}",
            )
        ax_c.plot(case["T"], case["specific_heat"], color=color, lw=2.0, label=label)

    style(ax_eg, t_min)
    style(ax_t2g, t_min)
    style(ax_c, t_min, zero_line=False)
    ax_eg.set_ylabel(r"$\alpha_Q$")
    ax_t2g.set_ylabel(r"$\alpha_Q$")
    ax_c.set_ylabel(r"$C(T)$")
    ax_eg.set_title(title + r" - $E_g$")
    ax_t2g.set_title(title + r" - $T_{2g}$")
    ax_c.set_title(title + r" - specific heat")
    ax_eg.legend(frameon=True, fontsize=7, ncol=2)
    ax_t2g.legend(frameon=True, fontsize=6, ncol=3)
    ax_c.legend(frameon=True, fontsize=8)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", type=Path, required=True)
    ap.add_argument("--j3-08", type=Path, required=True)
    ap.add_argument("--eg-jpm", action="append", default=[], help="repeat LABEL=PATH for Eg-source Jpm sweep")
    ap.add_argument("--t2g-jpm", action="append", default=[], help="repeat LABEL=PATH for T2g-source Jpm sweep")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    if len(args.eg_jpm) == 0 or len(args.t2g_jpm) == 0:
        ap.error("need at least one --eg-jpm and one --t2g-jpm case")

    baseline = load_case(args.baseline)
    j3_08 = load_case(args.j3_08)
    eg_cases = [(label, load_case(path)) for label, path in (parse_case_spec(spec) for spec in args.eg_jpm)]
    t2g_cases = [(label, load_case(path)) for label, path in (parse_case_spec(spec) for spec in args.t2g_jpm)]

    t_min = float(np.min(baseline["T"][baseline["T"] > 0]))
    fig, axes = plt.subplots(4, 3, figsize=(16, 14), sharex=True)

    plot_single_row(axes[0], r"Baseline: $J_3=0$", baseline, t_min)
    plot_single_row(axes[1], r"Clean: $J_3=0.08$", j3_08, t_min)
    plot_overlay_row(axes[2], r"$E_g$ source: $-10^{-3}Q_1$", eg_cases, t_min)
    plot_overlay_row(axes[3], r"$T_{2g}$ source: $-10^{-3}Q_{xz}$", t2g_cases, t_min)

    for col in range(3):
        axes[-1, col].set_xlabel(r"$T\,/\,|J_{zz}|$")

    fig.suptitle(
        r"Exact full-ED thermal expansion and specific heat across clean and source-field $J_{\pm}$ sweeps",
        fontsize=13,
    )
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for ext in (".png", ".pdf"):
        out_path = args.out.with_suffix(ext)
        fig.savefig(out_path, dpi=180, bbox_inches="tight")
        print(f"wrote {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
