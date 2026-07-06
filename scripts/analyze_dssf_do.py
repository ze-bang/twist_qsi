"""
Aggregate and plot the polarized-neutron DSSF (NSF + SF channels) for the
16-site (1,1,1) cubic pyrochlore cluster in the *dipolar-octupolar (DO)
QSI* convention. Compares the bare phi = 0 spectrum against the
{0, pi}^3 twist-corner average.

Reads the per-corner .npz files written by python_dssf_do.py.

Outputs (under paper/figs/):
    fig_dssf_do_NSF_SF.pdf, .png      bare vs twist-averaged NSF + SF at
                                      the cubic-X-point and L-point
    fig_dssf_do_NSF_SF_lowfreq.pdf,.png   low-omega zooms
    diagnostics_dssf_do.json          peak positions, sum rules
"""
from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

X_KEYS = ["X1", "X2", "X3"]


def n_pi_of(phi, pi=np.pi):
    return sum(1 for x in phi if abs(x - pi) < 1e-8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(ROOT / "output" / "dssf_do"))
    ap.add_argument("--demo-root", default=str(ROOT / "output" / "demo"))
    ap.add_argument("--figs", default=str(ROOT / "paper" / "figs"))
    ap.add_argument("--suffix", default="",
                    help="suffix appended to figure / json filenames (e.g. _jpm005)")
    args = ap.parse_args()

    root = Path(args.root)
    demo_root = Path(args.demo_root)
    figs = Path(args.figs)
    figs.mkdir(parents=True, exist_ok=True)
    suf = args.suffix

    summary = json.loads((root / "summary.json").read_text())

    pi = float(np.pi)
    cube_corners = list(product([0.0, pi], repeat=3))

    dssf: dict = {}
    for rec in summary["twists"]:
        phi = tuple(rec["phi"])
        d = np.load(rec["npz"])
        e = {"omega": d["omega"], "E0": float(d["E0"])}
        for chan in ("NSF", "SF"):
            for q in ("Gamma", "X1", "X2", "X3", "L"):
                key = f"{chan}_{q}"
                e[key] = d[f"S_{key}"]
                e[f"sum_{key}"] = float(d[f"sum_{key}"][0])
        dssf[phi] = e

    demo_summary = json.loads((demo_root / "summary.json").read_text())
    Jpm = demo_summary["Jpm"]
    Jzz = demo_summary["Jzz"]
    g_hex = 12.0 * abs(Jpm) ** 3 / Jzz**2
    g_4cycle = 4.0 * abs(Jpm) ** 2 / Jzz

    omega = dssf[cube_corners[0]]["omega"]
    eta = float(summary["eta"])

    bare_phi = (0.0, 0.0, 0.0)

    def x_avg(corner_dict, channel: str):
        s = np.zeros_like(omega)
        for k in X_KEYS:
            s += corner_dict[f"{channel}_{k}"]
        return s / len(X_KEYS)

    def stack(channel: str, q: str):
        if q == "X":
            return np.stack(
                [x_avg(dssf[p], channel) for p in cube_corners], axis=0
            )
        return np.stack(
            [dssf[p][f"{channel}_{q}"] for p in cube_corners], axis=0
        )

    bare_NSF_X = x_avg(dssf[bare_phi], "NSF")
    bare_SF_X = x_avg(dssf[bare_phi], "SF")
    bare_NSF_L = dssf[bare_phi]["NSF_L"]
    bare_SF_L = dssf[bare_phi]["SF_L"]

    NSF_X = stack("NSF", "X")
    SF_X = stack("SF", "X")
    NSF_L = stack("NSF", "L")
    SF_L = stack("SF", "L")

    NSF_X_avg, NSF_X_sd = NSF_X.mean(0), NSF_X.std(0)
    SF_X_avg, SF_X_sd = SF_X.mean(0), SF_X.std(0)
    NSF_L_avg, NSF_L_sd = NSF_L.mean(0), NSF_L.std(0)
    SF_L_avg, SF_L_sd = SF_L.mean(0), SF_L.std(0)

    class_color = {0: "tab:red", 1: "tab:cyan", 2: "tab:green", 3: "tab:olive"}
    class_label = {
        0: r"$|\varphi|_\pi=0$ (bare $\boldsymbol{\varphi}{=}\mathbf{0}$)",
        1: r"$|\varphi|_\pi=1$ (3 corners)",
        2: r"$|\varphi|_\pi=2$ (3 corners)",
        3: r"$|\varphi|_\pi=3$ corner $(\pi,\pi,\pi)$",
    }

    # -------------------------------------------------------------------
    # Main figure: 2 (NSF/SF) x 2 (X / L) = 4 panels, bare-vs-twist style
    # -------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 8.6), sharex=True)

    def _draw_bare_vs_avg(ax, bare, avg, spread, title, ylabel):
        ax.plot(omega, bare, lw=2.4, color="tab:red",
                label=r"bare $\boldsymbol{\varphi}{=}\mathbf{0}$")
        ax.fill_between(omega, avg - spread, avg + spread,
                        color="tab:blue", alpha=0.20,
                        label=r"$\pm\sigma_\varphi$")
        ax.plot(omega, avg, lw=2.4, color="tab:blue",
                label=r"twist-averaged")
        ax.axvline(g_hex, color="black", ls="--", lw=1.2, alpha=0.85,
                   label=fr"$g_{{\rm hex}}\!\approx\!{g_hex:.3g}$")
        ax.axvline(g_4cycle, color="tab:orange", ls=":", lw=1.2, alpha=0.85,
                   label=fr"$4|J_\pm|^2/J_{{zz}}\!\approx\!{g_4cycle:.3g}$")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xlim(-0.02, omega.max())
        ax.legend(frameon=False, loc="upper right", fontsize=8)
        ax.grid(alpha=0.3)

    _draw_bare_vs_avg(
        axes[0, 0], bare_NSF_X, NSF_X_avg, NSF_X_sd,
        title=r"(a) NSF, $\mathbf{q}=X$",
        ylabel=r"$S^{\rm NSF}_{\rm DO}(X,\omega)$",
    )
    _draw_bare_vs_avg(
        axes[0, 1], bare_SF_X, SF_X_avg, SF_X_sd,
        title=r"(b) SF, $\mathbf{q}=X$",
        ylabel=r"$S^{\rm SF}_{\rm DO}(X,\omega)$",
    )
    _draw_bare_vs_avg(
        axes[1, 0], bare_NSF_L, NSF_L_avg, NSF_L_sd,
        title=r"(c) NSF, $\mathbf{q}=L\!=\!(\pi,\pi,\pi)$",
        ylabel=r"$S^{\rm NSF}_{\rm DO}(L,\omega)$",
    )
    _draw_bare_vs_avg(
        axes[1, 1], bare_SF_L, SF_L_avg, SF_L_sd,
        title=r"(d) SF, $\mathbf{q}=L\!=\!(\pi,\pi,\pi)$",
        ylabel=r"$S^{\rm SF}_{\rm DO}(L,\omega)$",
    )
    for ax in axes[-1]:
        ax.set_xlabel(r"$\omega \,/\, J_{zz}$")

    fig.suptitle(
        rf"$T\!=\!0$ DO-QSI polarized-neutron DSSF on the 16-site "
        rf"$(1,1,1)_{{\rm cubic}}$ cluster, "
        rf"$J_\pm\!=\!{Jpm:+.3g}\,J_{{zz}}$, "
        rf"vertical polarisation $\hat n\!\parallel\![1,\bar1,0]$, "
        rf"continued-fraction Lanczos ($\eta\!=\!{eta:.3g}\,J_{{zz}}$). "
        rf"NSF: dipolar component along $\hat n$; SF: along $\hat n_2\!=\!\hat q\!\times\!\hat n$.",
        fontsize=10.5,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(figs / f"fig_dssf_do_NSF_SF{suf}.pdf")
    fig.savefig(figs / f"fig_dssf_do_NSF_SF{suf}.png", dpi=170)
    plt.close(fig)

    # -------------------------------------------------------------------
    # Per-corner figure (4 panels, NSF+SF at X and L, all 8 corners shown)
    # -------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 8.6), sharex=True)

    def _draw_per_corner(ax, st, avg, title, ylabel):
        seen = set()
        for k, p in enumerate(cube_corners):
            n_pi = n_pi_of(p, pi)
            col = class_color[n_pi]
            lw = 2.2 if p == bare_phi else 1.0
            alpha = 1.0 if p == bare_phi else 0.55
            zorder = 5 if p == bare_phi else 3
            lbl = class_label[n_pi] if n_pi not in seen else None
            seen.add(n_pi)
            ax.plot(omega, st[k], lw=lw, alpha=alpha, color=col,
                    label=lbl, zorder=zorder)
        ax.plot(omega, avg, lw=2.6, color="tab:blue",
                label=r"twist-avg", zorder=6)
        ax.axvline(g_hex, color="black", ls="--", lw=1.2, alpha=0.85)
        ax.axvline(g_4cycle, color="tab:orange", ls=":", lw=1.2, alpha=0.85)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xlim(-0.02, omega.max())
        ax.legend(frameon=False, loc="upper right", fontsize=8)
        ax.grid(alpha=0.3)

    _draw_per_corner(
        axes[0, 0], NSF_X, NSF_X_avg,
        r"(a) NSF, $\mathbf{q}=X$, per corner", r"$S^{\rm NSF}_{\rm DO}(X,\omega)$",
    )
    _draw_per_corner(
        axes[0, 1], SF_X, SF_X_avg,
        r"(b) SF, $\mathbf{q}=X$, per corner", r"$S^{\rm SF}_{\rm DO}(X,\omega)$",
    )
    _draw_per_corner(
        axes[1, 0], NSF_L, NSF_L_avg,
        r"(c) NSF, $\mathbf{q}=L$, per corner", r"$S^{\rm NSF}_{\rm DO}(L,\omega)$",
    )
    _draw_per_corner(
        axes[1, 1], SF_L, SF_L_avg,
        r"(d) SF, $\mathbf{q}=L$, per corner", r"$S^{\rm SF}_{\rm DO}(L,\omega)$",
    )
    for ax in axes[-1]:
        ax.set_xlabel(r"$\omega \,/\, J_{zz}$")

    fig.suptitle(
        rf"Per-corner DO-QSI NSF/SF on the 16-site cluster "
        rf"($J_\pm\!=\!{Jpm:+.3g}\,J_{{zz}}$): each $\boldsymbol{{\varphi}}$ "
        rf"corner coloured by parity class $|\varphi|_\pi$, "
        rf"twist-averaged in heavy blue.",
        fontsize=10.5,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(figs / f"fig_dssf_do_NSF_SF_per_corner{suf}.pdf")
    fig.savefig(figs / f"fig_dssf_do_NSF_SF_per_corner{suf}.png", dpi=170)
    plt.close(fig)

    # -------------------------------------------------------------------
    # Low-frequency zoom: 2x2 zooms
    # -------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 8.6), sharex=True)

    def _draw_lowfreq(ax, st, avg, title, ylabel):
        seen = set()
        for k, p in enumerate(cube_corners):
            n_pi = n_pi_of(p, pi)
            col = class_color[n_pi]
            lw = 2.2 if p == bare_phi else 1.1
            alpha = 1.0 if p == bare_phi else 0.65
            zorder = 5 if p == bare_phi else 3
            lbl = class_label[n_pi] if n_pi not in seen else None
            seen.add(n_pi)
            ax.plot(omega, st[k], lw=lw, alpha=alpha, color=col,
                    label=lbl, zorder=zorder)
        ax.plot(omega, avg, lw=2.6, color="tab:blue",
                label="twist-avg", zorder=6)
        ax.axvline(g_hex, color="black", ls="--", lw=1.4,
                   label=fr"$g_{{\rm hex}}\!\approx\!{g_hex:.3g}$")
        ax.axvline(g_4cycle, color="tab:orange", ls=":", lw=1.4,
                   label=fr"$4|J_\pm|^2/J_{{zz}}\!\approx\!{g_4cycle:.3g}$")
        ax.set_xlim(-0.005, 0.30)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(frameon=False, loc="upper right", fontsize=8)
        ax.grid(alpha=0.3)

    _draw_lowfreq(
        axes[0, 0], NSF_X, NSF_X_avg,
        r"(a) NSF, $\mathbf{q}=X$, low-$\omega$ zoom",
        r"$S^{\rm NSF}_{\rm DO}(X,\omega)$",
    )
    _draw_lowfreq(
        axes[0, 1], SF_X, SF_X_avg,
        r"(b) SF, $\mathbf{q}=X$, low-$\omega$ zoom",
        r"$S^{\rm SF}_{\rm DO}(X,\omega)$",
    )
    _draw_lowfreq(
        axes[1, 0], NSF_L, NSF_L_avg,
        r"(c) NSF, $\mathbf{q}=L$, low-$\omega$ zoom",
        r"$S^{\rm NSF}_{\rm DO}(L,\omega)$",
    )
    _draw_lowfreq(
        axes[1, 1], SF_L, SF_L_avg,
        r"(d) SF, $\mathbf{q}=L$, low-$\omega$ zoom",
        r"$S^{\rm SF}_{\rm DO}(L,\omega)$",
    )
    for ax in axes[-1]:
        ax.set_xlabel(r"$\omega \,/\, J_{zz}$")

    fig.suptitle(
        rf"Low-$\omega$ zooms of the DO-QSI NSF/SF DSSF "
        rf"($J_\pm\!=\!{Jpm:+.3g}\,J_{{zz}}$, $\eta\!=\!{eta:.3g}$).",
        fontsize=10.5,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(figs / f"fig_dssf_do_NSF_SF_lowfreq{suf}.pdf")
    fig.savefig(figs / f"fig_dssf_do_NSF_SF_lowfreq{suf}.png", dpi=170)
    plt.close(fig)

    # -------------------------------------------------------------------
    # Diagnostics
    # -------------------------------------------------------------------
    def peak_pos(spec, om):
        mask = om >= 0.0
        if not mask.any():
            return float("nan")
        i = int(np.argmax(spec[mask]))
        return float(om[mask][i])

    diagnostics = {
        "Jpm": Jpm, "Jzz": Jzz,
        "g_hex": float(g_hex),
        "g_4cycle_estimate": float(g_4cycle),
        "n_corners": len(cube_corners),
        "broadening_eta": eta,
        "lanczos_steps": int(summary["lanczos_steps"]),
    }
    for chan, qlbl, bare, avg, sd, st in [
        ("NSF", "X", bare_NSF_X, NSF_X_avg, NSF_X_sd, NSF_X),
        ("SF",  "X", bare_SF_X,  SF_X_avg,  SF_X_sd,  SF_X),
        ("NSF", "L", bare_NSF_L, NSF_L_avg, NSF_L_sd, NSF_L),
        ("SF",  "L", bare_SF_L,  SF_L_avg,  SF_L_sd,  SF_L),
    ]:
        diagnostics[f"bare_{chan}_{qlbl}_peak_omega"] = peak_pos(bare, omega)
        diagnostics[f"twist_avg_{chan}_{qlbl}_peak_omega"] = peak_pos(avg, omega)
        diagnostics[f"bare_{chan}_{qlbl}_peak_height"] = float(bare.max())
        diagnostics[f"twist_avg_{chan}_{qlbl}_peak_height"] = float(avg.max())
    (figs / f"diagnostics_dssf_do{suf}.json").write_text(
        json.dumps(diagnostics, indent=2),
    )

    print("=== DO-QSI NSF+SF DSSF analysis complete ===")
    for k, v in diagnostics.items():
        if isinstance(v, float):
            print(f"  {k:40s} {v:.4g}")
        else:
            print(f"  {k:40s} {v}")
    print(f"Wrote figures to {figs}")


if __name__ == "__main__":
    main()
