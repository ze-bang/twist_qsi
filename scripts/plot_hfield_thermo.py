"""
Plot twist-averaged C(T) at various transverse-field strengths h_perp
(from run_hfield_thermo_sweep.py), and the position of the low-T
("spurious/ring-exchange") specific-heat peak as a function of h_perp.

Reads ../hfield_thermo/data/summary.json, writes into
../hfield_thermo/figs/ (kept separate from paper/figs).
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def read_ftlm(h5path):
    with h5py.File(h5path, "r") as f:
        T = f["thermodynamics/temperatures"][...]
        C = f["thermodynamics/specific_heat"][...]
    return T, C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "hfield_thermo" / "data"))
    ap.add_argument("--figs", default=str(ROOT / "hfield_thermo" / "figs"))
    ap.add_argument("--low-T-cutoff", type=float, default=0.18,
                    help="restrict the low-T peak search to T below this")
    ap.add_argument("--plot-T-max", type=float, default=0.25,
                    help="truncate the C(T) plot here -- confirmed converged "
                         "against an 800-eigenvalue reference up to this point; "
                         "beyond it the 400-eigenvalue spectrum sum is not "
                         "converged against the true (2^16-state) high-T "
                         "ice-ordering peak, and shows a spurious re-heating bump")
    args = ap.parse_args()

    data_root = Path(args.data)
    figs = Path(args.figs)
    figs.mkdir(parents=True, exist_ok=True)

    summary = json.loads((data_root / "summary.json").read_text())
    flux = summary["flux"]
    Jpm = -(summary["jxy"] * 2) / 4.0

    by_h = defaultdict(list)
    for r in summary["rows"]:
        by_h[r["h_perp"]].append(r["ftlm_h5"])

    h_values = sorted(by_h.keys())
    C_by_h = {}
    T_ref = None
    for h in h_values:
        C_stack = []
        for h5path in by_h[h]:
            T, C = read_ftlm(h5path)
            T_ref = T
            C_stack.append(C)
        C_by_h[h] = np.mean(C_stack, axis=0)

    # --- (a) C(T) curves, sequential (single-hue, light -> dark) by h ---
    plot_mask = T_ref <= args.plot_T_max
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    cmap = plt.cm.Blues
    hmax = max(h_values)
    for h in h_values:
        frac = 0.30 + 0.65 * (h / hmax if hmax > 0 else 0.0)
        color = cmap(frac)
        lw = 2.4 if h == 0.0 else 1.6
        ax.plot(T_ref[plot_mask], C_by_h[h][plot_mask], color=color, lw=lw,
                label=fr"$h_\perp={h:.2f}\,J_{{zz}}$" if h in (h_values[0], h_values[-1]) or
                h in (0.02, 0.04, 0.06, 0.10) else None,
                zorder=3 + (h == 0.0))
    ax.set_xscale("log")
    ax.set_xlabel(r"$T\,/\,J_{zz}$")
    ax.set_ylabel(r"specific heat $\overline{C}(T)$")
    ax.set_title(rf"Twist-averaged $C(T)$ vs transverse field $h_\perp$ ({flux}, $J_\pm={Jpm:+.3g}\,J_{{zz}}$)")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left", title=r"$h_\perp$ (light$\to$dark = increasing)")
    ax.grid(alpha=0.25)
    ax.text(0.98, 0.03,
            f"spectrum truncated to {summary.get('n_eigs', '?')} eigenvalues "
            f"(converged vs.\n800-eig. reference for $T\\lesssim{args.plot_T_max}$)\n"
            "(true high-$T$ ice peak at $T\\sim0.26$ not resolved here)",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7.2,
            color="#52514e", style="italic")
    fig.tight_layout()
    fig.savefig(figs / "fig_CT_vs_hfield.png", dpi=170, bbox_inches="tight")
    fig.savefig(figs / "fig_CT_vs_hfield.pdf", bbox_inches="tight")
    plt.close(fig)

    # --- (b) low-T peak position vs h_perp, LINEAR axes ---
    # The T grid is log-spaced (150 bins) -- log(T) is nearly uniform, so a
    # parabolic fit to the 3 points around the grid-level argmax in log(T)
    # gives a sub-grid-resolution peak location instead of a staircase
    # artifact from reading off the nearest bin.
    def refine_peak(T, C, mask):
        Tm, Cm = T[mask], C[mask]
        i = np.argmax(Cm)
        if i == 0 or i == len(Cm) - 1:
            return Tm[i], Cm[i]
        x = np.log(Tm[i - 1:i + 2])
        y = Cm[i - 1:i + 2]
        # vertex of the parabola through the 3 points
        denom = (x[0] - x[1]) * (x[0] - x[2]) * (x[1] - x[2])
        a = (x[2] * (y[1] - y[0]) + x[1] * (y[0] - y[2]) + x[0] * (y[2] - y[1])) / denom
        b = (x[2]**2 * (y[0] - y[1]) + x[1]**2 * (y[2] - y[0]) + x[0]**2 * (y[1] - y[2])) / denom
        if abs(a) < 1e-300:
            return Tm[i], Cm[i]
        x_vertex = -b / (2 * a)
        c = y[0] - a * x[0]**2 - b * x[0]
        return float(np.exp(x_vertex)), float(a * x_vertex**2 + b * x_vertex + c)

    mask_low = T_ref < args.low_T_cutoff
    refined = [refine_peak(T_ref, C_by_h[h], mask_low) for h in h_values]
    peak_T = np.array([r[0] for r in refined])
    peak_C = np.array([r[1] for r in refined])

    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    ax.plot(h_values, peak_T, "o-", color="#2a78d6", ms=7, lw=2.0, mec="white", mew=0.7)
    ax.set_xlabel(r"$h_\perp\,/\,J_{zz}$")
    ax.set_ylabel(r"low-$T$ peak position  $T_{\rm peak}(h_\perp)\,/\,J_{zz}$")
    ax.set_title(rf"Low-$T$ specific-heat peak vs field strength ({flux}, $J_\pm={Jpm:+.3g}\,J_{{zz}}$)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(figs / "fig_Tpeak_vs_hfield.png", dpi=170, bbox_inches="tight")
    fig.savefig(figs / "fig_Tpeak_vs_hfield.pdf", bbox_inches="tight")
    plt.close(fig)

    diagnostics = {"flux": flux, "Jpm": Jpm, "h_values": h_values,
                   "peak_T": peak_T.tolist(), "peak_C": peak_C.tolist()}
    (figs / "diagnostics_hfield_thermo.json").write_text(json.dumps(diagnostics, indent=2))
    print("Wrote figures to", figs)
    print(json.dumps(diagnostics, indent=2))


if __name__ == "__main__":
    main()
