#!/usr/bin/env python3
"""Scenario-grid comparison for exact alpha_Q and specific heat curves."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

EG_OPS = [("Eg_Q1", r"$Q_1$"), ("Eg_Q2", r"$Q_2$")]
T2G_OPS = [("T2g_Q_xy", r"$Q_{xy}$"), ("T2g_Q_xz", r"$Q_{xz}$"), ("T2g_Q_yz", r"$Q_{yz}$")]


def load_case(path: Path, label: str) -> dict:
    d = np.load(path)
    return {
        "label": label,
        "T": d["temperatures"],
        "alpha": {op: d[f"{op}_alpha"] for op, _ in EG_OPS + T2G_OPS},
        "specific_heat": d["specific_heat"],
    }


def parse_scenario_spec(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise ValueError(f"invalid --scenario '{spec}' (expected TITLE=PATH)")
    title, path = spec.rsplit("=", 1)
    title = title.strip()
    if not title:
        raise ValueError(f"invalid --scenario '{spec}' (empty title)")
    return title, Path(path)


def style(ax, t_min: float) -> None:
    ax.set_xscale("log")
    ax.set_xlim(left=max(t_min, 1e-4))
    ax.axhline(0.0, color="k", lw=0.6, alpha=0.35)
    ax.grid(True, which="both", alpha=0.3)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--scenario",
        action="append",
        default=[],
        help="repeatable TITLE=PATH scenario spec; if omitted, falls back to legacy 4-row arguments",
    )
    ap.add_argument("--j3-0", type=Path)
    ap.add_argument("--j3-08", type=Path)
    ap.add_argument("--eg-source", type=Path)
    ap.add_argument("--t2g-source", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    if args.scenario:
        scenarios = [parse_scenario_spec(spec) for spec in args.scenario]
        scenarios = [(title, load_case(path, title)) for title, path in scenarios]
    else:
        missing = [
            name for name in ("j3_0", "j3_08", "eg_source", "t2g_source")
            if getattr(args, name) is None
        ]
        if missing:
            joined = ", ".join(f"--{name.replace('_', '-')}" for name in missing)
            ap.error(f"missing required legacy arguments: {joined}")
        scenarios = [
            ("$J_3=0$", load_case(args.j3_0, "$J_3=0$")),
            ("$J_3=0.08$", load_case(args.j3_08, "$J_3=0.08$")),
            (r"$E_g$ source: $-10^{-3}Q_1$", load_case(args.eg_source, r"$E_g$ source")),
            (r"$T_{2g}$ source: $-10^{-3}Q_{xz}$", load_case(args.t2g_source, r"$T_{2g}$ source")),
        ]

    t_min = float(np.min(scenarios[0][1]["T"][scenarios[0][1]["T"] > 0]))
    n_rows = len(scenarios)
    fig, axes = plt.subplots(n_rows, 3, figsize=(16, 3.2 * n_rows + 1.5), sharex=True)
    axes = np.atleast_2d(axes)

    eg_styles = ["-", "--"]
    t2g_styles = ["-", "--", "-."]
    eg_colors = ["#1f77b4", "#d62728"]
    t2g_colors = ["#1f77b4", "#d62728", "#2ca02c"]

    for row, (title, case) in enumerate(scenarios):
        ax_eg = axes[row, 0]
        ax_t2g = axes[row, 1]
        ax_c = axes[row, 2]

        for i, (op, label) in enumerate(EG_OPS):
            ax_eg.plot(
                case["T"], case["alpha"][op],
                color=eg_colors[i], ls=eg_styles[i], lw=2.0,
                label=label,
            )
        for i, (op, label) in enumerate(T2G_OPS):
            ax_t2g.plot(
                case["T"], case["alpha"][op],
                color=t2g_colors[i], ls=t2g_styles[i], lw=2.0,
                label=label,
            )

        style(ax_eg, t_min)
        style(ax_t2g, t_min)
        style(ax_c, t_min)
        ax_eg.set_ylabel(r"$\alpha_Q$")
        ax_t2g.set_ylabel(r"$\alpha_Q$")
        ax_c.set_ylabel(r"$C(T)$")
        ax_eg.set_title(title + " — $E_g$")
        ax_t2g.set_title(title + r" — $T_{2g}$")
        ax_c.set_title(title + r" — specific heat")
        ax_eg.legend(frameon=True, fontsize=8)
        ax_t2g.legend(frameon=True, fontsize=8)
        ax_c.plot(case["T"], case["specific_heat"], color="black", lw=2.0, label=r"$C(T)$")
        ax_c.legend(frameon=True, fontsize=8)

    axes[-1, 0].set_xlabel(r"$T\,/\,|J_{zz}|$")
    axes[-1, 1].set_xlabel(r"$T\,/\,|J_{zz}|$")
    axes[-1, 2].set_xlabel(r"$T\,/\,|J_{zz}|$")

    fig.suptitle(
        r"Exact full-ED thermal expansion $\alpha_Q=(\langle QH\rangle-\langle Q\rangle\langle H\rangle)/T^2$",
        fontsize=13,
    )
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for ext in (".png", ".pdf"):
        p = args.out.with_suffix(ext)
        fig.savefig(p, dpi=180, bbox_inches="tight")
        print(f"wrote {p}")
    plt.close(fig)


if __name__ == "__main__":
    main()
