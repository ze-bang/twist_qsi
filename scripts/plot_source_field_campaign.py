#!/usr/bin/env python3
"""Plot completed source-field exact-ED alpha_Q curves."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

EG_OPS = [("Eg_Q1", r"$Q_1$"), ("Eg_Q2", r"$Q_2$")]
T2G_OPS = [("T2g_Q_xy", r"$Q_{xy}$"), ("T2g_Q_xz", r"$Q_{xz}$"), ("T2g_Q_yz", r"$Q_{yz}$")]


def _style_log_T(ax, t_min: float) -> None:
    ax.set_xscale("log")
    ax.set_xlim(left=max(t_min, 1e-4))
    ax.set_xlabel(r"$T\,/\,|J_{zz}|$")


def load_case(path: Path, label: str) -> dict:
    d = np.load(path)
    sources = {k.replace("source_", ""): float(d[k]) for k in d.files if k.startswith("source_")}
    return {
        "label": label,
        "T": d["temperatures"],
        "alpha": {op: d[f"{op}_alpha"] for op, _ in EG_OPS + T2G_OPS},
        "sources": sources,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", type=Path, required=True, help="clean J3=0.08 exact result")
    ap.add_argument("--case", action="append", default=[], help="LABEL=path to completed source-field npz")
    ap.add_argument("--skip-missing", action="store_true",
                    help="skip missing --case paths instead of failing")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    cases = [load_case(args.clean, "clean $J_3=0.08$")]
    for spec in args.case:
        if "=" not in spec:
            raise SystemExit(f"invalid --case '{spec}', expected LABEL=PATH")
        label, raw_path = spec.split("=", 1)
        path = Path(raw_path)
        if not path.exists():
            if args.skip_missing:
                print(f"skipping missing case: {label} -> {path}")
                continue
            raise SystemExit(f"missing case path: {path}")
        cases.append(load_case(path, label))

    T = cases[0]["T"]
    t_min = float(np.min(T[T > 0]))
    fig, axes = plt.subplots(2, 1, figsize=(9, 7.5), sharex=True)
    ax_eg, ax_t2g = axes
    colors = ["black", "#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#8c564b"]
    line_styles = ["-", "--", "-.", ":"]

    for ax, ops, title in [
        (ax_eg, EG_OPS, r"$E_g$ response"),
        (ax_t2g, T2G_OPS, r"$T_{2g}$ response"),
    ]:
        for ci, case in enumerate(cases):
            color = colors[ci % len(colors)]
            for oi, (op, op_label) in enumerate(ops):
                ax.plot(
                    case["T"],
                    case["alpha"][op],
                    color=color,
                    lw=1.7,
                    ls=line_styles[oi % len(line_styles)],
                    label=rf"{op_label}, {case['label']}",
                )
        _style_log_T(ax, t_min)
        ax.set_ylabel(r"$\alpha_Q$")
        ax.set_title(title + r" — exact full ED with source fields")
        ax.axhline(0.0, color="k", lw=0.6, alpha=0.4)
        ax.grid(True, which="both", alpha=0.35)
        ax.legend(frameon=True, fontsize=7, ncol=2)

    fig.suptitle(
        r"Completed source-field campaign vs clean exact result",
        fontsize=11,
    )
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for ext in (".png", ".pdf"):
        p = args.out.with_suffix(ext)
        fig.savefig(p, dpi=170, bbox_inches="tight")
        print(f"wrote {p}")
    plt.close(fig)


if __name__ == "__main__":
    main()
