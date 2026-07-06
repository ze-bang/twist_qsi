"""
Aggregate and plot the SzSz dynamical structure factor S^{zz}(q, omega)
for the 16-site (1,1,1) cubic pyrochlore cluster, contrasting the bare
phi = 0 spectrum against the {0, pi}^3 twist-corner average.

Reads the per-corner .npz files written by python_dssf.py (continued-
fraction Lanczos against the same twisted Hamiltonians used by run_demo.py
for C(T) and S(T)). The C++ ED `dssf ground_state_dssf` engine has a
known issue with the lift-and-shift ladder-basis SzSz operator pair
that produces ~1e-22 spectral weight on this cluster, so we use the
Python continued-fraction implementation as the production path.

Outputs (under paper/figs/):
    fig_dssf_szsz.pdf, .png           main figure: bare vs twist average
                                      at the cluster X-point (with the
                                      three cubic-equivalent X copies
                                      averaged on each corner)
    fig_dssf_szsz_per_corner.pdf,.png all 8 twist corners coloured by
                                      parity class
    diagnostics_dssf.json             summary of peak positions, sum
                                      rules, spectral weights
"""
from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path

import h5py
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
    ap.add_argument("--root", default=str(ROOT / "output" / "dssf_py"))
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

    dssf = {}
    for rec in summary["twists"]:
        phi = tuple(rec["phi"])
        d = np.load(rec["npz"])
        dssf[phi] = {
            "omega": d["omega"],
            "E0": float(d["E0"]),
            "Gamma": d["S_Gamma"],
            "X1": d["S_X1"],
            "X2": d["S_X2"],
            "X3": d["S_X3"],
            "L":  d["S_L"],
            "sum_X1": float(d["sum_X1"][0]),
            "sum_X2": float(d["sum_X2"][0]),
            "sum_X3": float(d["sum_X3"][0]),
            "sum_L":  float(d["sum_L"][0]),
        }

    # Pull Jpm, Jzz from demo summary so we can mark the photon scale
    demo_summary = json.loads((demo_root / "summary.json").read_text())
    Jpm = demo_summary["Jpm"]
    Jzz = demo_summary["Jzz"]
    g_hex = 12.0 * abs(Jpm) ** 3 / Jzz**2
    g_4cycle = 4.0 * abs(Jpm) ** 2 / Jzz

    omega = dssf[cube_corners[0]]["omega"]
    domega = omega[1] - omega[0]
    eta = float(summary["eta"])

    def x_avg(corner):
        s = np.zeros_like(omega)
        for k in X_KEYS:
            s += corner[k]
        return s / len(X_KEYS)

    bare_phi = (0.0, 0.0, 0.0)
    bare_X = x_avg(dssf[bare_phi])
    bare_L = dssf[bare_phi]["L"]

    X_stack = np.stack([x_avg(dssf[p]) for p in cube_corners], axis=0)
    X_avg = X_stack.mean(axis=0)
    X_spread = X_stack.std(axis=0)

    L_stack = np.stack([dssf[p]["L"] for p in cube_corners], axis=0)
    L_avg = L_stack.mean(axis=0)
    L_spread = L_stack.std(axis=0)

    static_X_per_corner = np.array([
        (dssf[p]["sum_X1"] + dssf[p]["sum_X2"] + dssf[p]["sum_X3"]) / 3.0
        for p in cube_corners
    ])
    static_L_per_corner = np.array([dssf[p]["sum_L"] for p in cube_corners])

    class_color = {0: "tab:red", 1: "tab:cyan", 2: "tab:green", 3: "tab:olive"}
    class_label = {
        0: r"$|\varphi|_\pi=0$  (bare $\boldsymbol{\varphi}=\mathbf{0}$)",
        1: r"$|\varphi|_\pi=1$  (3 corners)",
        2: r"$|\varphi|_\pi=2$  (3 corners)",
        3: r"$|\varphi|_\pi=3$  (corner $(\pi,\pi,\pi)$)",
    }

    # ---------- Figure 1: bare vs twist-averaged S^{zz}(q, omega) ---------
    # Top row: X-point (cubic-allowed on bare cluster, "spinon" scale).
    # Bottom row: L-point (only allowed via (pi,pi,pi) twist, lights up
    #            the lower-energy ring-exchange / 4-cycle structure).
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 8.6))

    def _draw_panel_per_corner(ax, stack, avg, spread, title, ylabel):
        seen = set()
        for k, p in enumerate(cube_corners):
            n_pi = n_pi_of(p, pi)
            col = class_color[n_pi]
            lw = 2.2 if p == bare_phi else 1.0
            alpha = 1.0 if p == bare_phi else 0.55
            zorder = 5 if p == bare_phi else 3
            lbl = class_label[n_pi] if n_pi not in seen else None
            seen.add(n_pi)
            ax.plot(omega, stack[k], lw=lw, alpha=alpha, color=col,
                    label=lbl, zorder=zorder)
        ax.fill_between(omega, avg - spread, avg + spread,
                        color="tab:blue", alpha=0.18)
        ax.plot(omega, avg, lw=2.6, color="tab:blue",
                label=r"twist-averaged $\overline{S^{zz}}$", zorder=6)
        ax.axvline(g_hex, color="black", ls="--", lw=1.2, alpha=0.85,
                   label=fr"$g_{{\rm hex}}\approx{g_hex:.3g}$")
        ax.axvline(g_4cycle, color="tab:orange", ls=":", lw=1.2, alpha=0.85,
                   label=fr"$4|J_\pm|^2/J_{{zz}}\approx{g_4cycle:.3g}$")
        ax.set_xlabel(r"$\omega \,/\, J_{zz}$")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xlim(-0.02, omega.max())
        ax.legend(frameon=False, loc="upper right", fontsize=8)
        ax.grid(alpha=0.3)

    def _draw_bare_vs_avg(ax, bare, avg, spread, title, ylabel):
        ax.plot(omega, bare, lw=2.4, color="tab:red",
                label=r"bare $S^{zz}_{\varphi=0}$")
        ax.fill_between(omega, avg - spread, avg + spread,
                        color="tab:blue", alpha=0.20,
                        label=r"twist corner spread $\pm\sigma_\varphi$")
        ax.plot(omega, avg, lw=2.4, color="tab:blue",
                label=r"twist-averaged $\overline{S^{zz}}$")
        ax.axvline(g_hex, color="black", ls="--", lw=1.2, alpha=0.85,
                   label=fr"$g_{{\rm hex}}\approx{g_hex:.3g}$")
        ax.axvline(g_4cycle, color="tab:orange", ls=":", lw=1.2, alpha=0.85,
                   label=fr"$4|J_\pm|^2/J_{{zz}}\approx{g_4cycle:.3g}$")
        ax.set_xlabel(r"$\omega \,/\, J_{zz}$")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xlim(-0.02, omega.max())
        ax.legend(frameon=False, loc="upper right", fontsize=8)
        ax.grid(alpha=0.3)

    _draw_panel_per_corner(
        axes[0, 0], X_stack, X_avg, X_spread,
        title=r"(a) $\mathbf{q}=X$, per twist corner + average",
        ylabel=r"$S^{zz}(X,\omega)$",
    )
    _draw_bare_vs_avg(
        axes[0, 1], bare_X, X_avg, X_spread,
        title=r"(b) $\mathbf{q}=X$, bare vs twist-averaged",
        ylabel=r"$S^{zz}(X,\omega)$",
    )
    _draw_panel_per_corner(
        axes[1, 0], L_stack, L_avg, L_spread,
        title=r"(c) $\mathbf{q}=L\!=\!(\pi,\pi,\pi)$, per twist corner + average",
        ylabel=r"$S^{zz}(L,\omega)$",
    )
    _draw_bare_vs_avg(
        axes[1, 1], bare_L, L_avg, L_spread,
        title=r"(d) $\mathbf{q}=L$, bare vs twist-averaged",
        ylabel=r"$S^{zz}(L,\omega)$",
    )

    fig.suptitle(
        rf"$T\!=\!0$ SzSz dynamical structure factor on the 16-site "
        rf"$(1,1,1)_{{\rm cubic}}$ pyrochlore cluster, "
        rf"$J_\pm\!=\!{Jpm:+.3g}\,J_{{zz}}$ "
        rf"(continued-fraction Lanczos, $\eta\!=\!{eta:.3g}\,J_{{zz}}$). "
        rf"Top: X-point (cubic-symm. avg of $\hat x,\hat y,\hat z$). "
        rf"Bottom: L-point.",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(figs / f"fig_dssf_szsz{suf}.pdf")
    fig.savefig(figs / f"fig_dssf_szsz{suf}.png", dpi=170)
    plt.close(fig)

    # ---------- Figure 2: zooms (low-omega L, low-omega X log scale) ------
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.6))

    ax = axes[0]
    seen = set()
    for k, p in enumerate(cube_corners):
        n_pi = n_pi_of(p, pi)
        col = class_color[n_pi]
        lw = 2.2 if p == bare_phi else 1.1
        alpha = 1.0 if p == bare_phi else 0.65
        zorder = 5 if p == bare_phi else 3
        lbl = class_label[n_pi] if n_pi not in seen else None
        seen.add(n_pi)
        ax.plot(omega, L_stack[k], lw=lw, alpha=alpha, color=col,
                label=lbl, zorder=zorder)
    ax.plot(omega, L_avg, lw=2.6, color="tab:blue",
            label=r"twist-avg", zorder=6)
    ax.axvline(g_hex, color="black", ls="--", lw=1.4,
               label=fr"$g_{{\rm hex}}\approx{g_hex:.3g}$")
    ax.axvline(g_4cycle, color="tab:orange", ls=":", lw=1.4,
               label=fr"$4|J_\pm|^2/J_{{zz}}\approx{g_4cycle:.3g}$")
    ax.set_xlim(-0.005, 0.25)
    ax.set_xlabel(r"$\omega \,/\, J_{zz}$")
    ax.set_ylabel(r"$S^{zz}(L,\omega)$")
    ax.set_title(r"(a) $\mathbf{q}=L$, low-$\omega$ zoom (photon-relevant scale)")
    ax.legend(frameon=False, loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    seen = set()
    for k, p in enumerate(cube_corners):
        n_pi = n_pi_of(p, pi)
        col = class_color[n_pi]
        lw = 2.2 if p == bare_phi else 1.1
        alpha = 1.0 if p == bare_phi else 0.65
        zorder = 5 if p == bare_phi else 3
        lbl = class_label[n_pi] if n_pi not in seen else None
        seen.add(n_pi)
        ax.plot(omega, X_stack[k], lw=lw, alpha=alpha, color=col,
                label=lbl, zorder=zorder)
    ax.plot(omega, X_avg, lw=2.6, color="tab:blue",
            label=r"twist-avg", zorder=6)
    ax.axvline(g_hex, color="black", ls="--", lw=1.4,
               label=fr"$g_{{\rm hex}}\approx{g_hex:.3g}$")
    ax.axvline(g_4cycle, color="tab:orange", ls=":", lw=1.4,
               label=fr"$4|J_\pm|^2/J_{{zz}}\approx{g_4cycle:.3g}$")
    ax.axvline(2 * abs(Jpm), color="tab:purple", ls="-.", lw=1.0, alpha=0.7,
               label=fr"$2|J_\pm|\approx{2*abs(Jpm):.3g}$")
    ax.set_xlim(0.7, 1.7)
    ax.set_xlabel(r"$\omega \,/\, J_{zz}$")
    ax.set_ylabel(r"$S^{zz}(X,\omega)$")
    ax.set_title(r"(b) $\mathbf{q}=X$, spinon-scale zoom")
    ax.legend(frameon=False, loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)

    fig.suptitle(
        rf"Zoomed views of the SzSz DSSF, "
        rf"16-site $(1,1,1)_{{\rm cubic}}$ at $J_\pm\!=\!{Jpm:+.3g}\,J_{{zz}}$, "
        rf"$\eta\!=\!{eta:.3g}\,J_{{zz}}$",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(figs / f"fig_dssf_szsz_lowfreq{suf}.pdf")
    fig.savefig(figs / f"fig_dssf_szsz_lowfreq{suf}.png", dpi=170)
    plt.close(fig)

    # ---------- Diagnostics ----------------------------------------------
    def peak_pos(spec, omega):
        mask = omega >= 0.0
        if not mask.any():
            return float("nan")
        idx = np.argmax(spec[mask])
        return float(omega[mask][idx])

    diagnostics = {
        "Jpm": Jpm,
        "Jzz": Jzz,
        "g_hex": float(g_hex),
        "g_4cycle_estimate": float(g_4cycle),
        "n_corners": len(cube_corners),
        "broadening_eta": eta,
        "lanczos_steps": int(summary["lanczos_steps"]),
        "bare_X_peak_omega": peak_pos(bare_X, omega),
        "twist_avg_X_peak_omega": peak_pos(X_avg, omega),
        "bare_X_peak_height": float(bare_X.max()),
        "twist_avg_X_peak_height": float(X_avg.max()),
        "bare_L_peak_omega": peak_pos(bare_L, omega),
        "twist_avg_L_peak_omega": peak_pos(L_avg, omega),
        "bare_L_peak_height": float(bare_L.max()),
        "twist_avg_L_peak_height": float(L_avg.max()),
        "bare_X_static_avg": float(np.mean([dssf[bare_phi]["sum_X1"],
                                            dssf[bare_phi]["sum_X2"],
                                            dssf[bare_phi]["sum_X3"]])),
        "twist_avg_X_static_avg": float(static_X_per_corner.mean()),
        "bare_L_static": float(dssf[bare_phi]["sum_L"]),
        "twist_avg_L_static": float(static_L_per_corner.mean()),
        "per_corner_X_static_avg": {
            str(p): float(s) for p, s in zip(cube_corners, static_X_per_corner)
        },
        "per_corner_L_static": {
            str(p): float(s) for p, s in zip(cube_corners, static_L_per_corner)
        },
        "ratio_bare_X_peak_over_g_hex": peak_pos(bare_X, omega) / g_hex,
        "ratio_twist_X_peak_over_g_hex": peak_pos(X_avg, omega) / g_hex,
        "ratio_bare_L_peak_over_g_hex": peak_pos(bare_L, omega) / g_hex,
        "ratio_twist_L_peak_over_g_hex": peak_pos(L_avg, omega) / g_hex,
    }
    (figs / f"diagnostics_dssf{suf}.json").write_text(
        json.dumps(diagnostics, indent=2)
    )

    print("=== SzSz DSSF analysis complete ===")
    print(f"Bare X peak:        omega={diagnostics['bare_X_peak_omega']:.4f} Jzz, "
          f"height={diagnostics['bare_X_peak_height']:.2f}, "
          f"static={diagnostics['bare_X_static_avg']:.4f}")
    print(f"Twist-avg X peak:   omega={diagnostics['twist_avg_X_peak_omega']:.4f} Jzz, "
          f"height={diagnostics['twist_avg_X_peak_height']:.2f}, "
          f"static={diagnostics['twist_avg_X_static_avg']:.4f}")
    print(f"Bare L peak:        omega={diagnostics['bare_L_peak_omega']:.4f} Jzz, "
          f"height={diagnostics['bare_L_peak_height']:.2f}, "
          f"static={diagnostics['bare_L_static']:.4f}")
    print(f"Twist-avg L peak:   omega={diagnostics['twist_avg_L_peak_omega']:.4f} Jzz, "
          f"height={diagnostics['twist_avg_L_peak_height']:.2f}, "
          f"static={diagnostics['twist_avg_L_static']:.4f}")
    print(f"g_hex:                          {g_hex:.4g} Jzz")
    print(f"4|Jpm|^2/Jzz (4-cycle):         {g_4cycle:.4g} Jzz")
    print(f"Wrote figures to {figs}")


if __name__ == "__main__":
    main()
