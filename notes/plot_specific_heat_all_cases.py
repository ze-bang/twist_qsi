#!/usr/bin/env python3
"""Plot C(T) for every recomputed finite-size scenario.

The figure uses the same standalone recomputation module as the note: no old
project scripts or git history are used.  It also writes the raw curves to
``notes/recomputed_specific_heat_curves.npz`` for later inspection.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import recompute_finite_size_artifact as R

HERE = Path(__file__).resolve().parent
FIGS = HERE / "figs"
FIGS.mkdir(exist_ok=True)

C_BARE = "#e67e22"
C_CLEAN = "#27ae60"
C_HEX = "#8e44ad"

plt.rcParams.update({
    "font.size": 9.5,
    "axes.titlesize": 10,
    "axes.labelsize": 9.5,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "mathtext.fontset": "cm",
    "font.family": "serif",
    "axes.linewidth": 0.85,
    "savefig.bbox": "tight",
    "savefig.dpi": 220,
})


def curve_key(basis, shape, jpm, mode):
    key = f"{basis}_{shape[0]}{shape[1]}{shape[2]}_J{jpm:+.2f}_{mode}"
    return key.replace("+", "p").replace("-", "m").replace(".", "p")


def load_cached_curves(path):
    if not path.exists():
        return None
    z = np.load(path, allow_pickle=False)
    T = z["T"]
    jpms = [-0.10, -0.05, 0.05]
    clusters = [
        ("cubic", (1, 1, 1), "16-site cubic", 16),
        ("fcc", (2, 2, 2), "32-site FCC", 32),
    ]
    records = []
    arrays = {"T": T}
    for basis, shape, label, n_sites in clusters:
        for jpm in jpms:
            for mode in ("all", "delta0"):
                key = curve_key(basis, shape, jpm, mode)
                ckey = f"{key}_C_per_site"
                if ckey not in z:
                    return None
                C = z[ckey]
                arrays[ckey] = C
                records.append({
                    "key": key,
                    "basis": basis,
                    "shape": shape,
                    "label": label,
                    "n_sites": n_sites,
                    "Jpm": jpm,
                    "mode": mode,
                    "Tpk": R.refined_peak(T, C),
                    "g4": 4 * jpm * jpm,
                    "ghex": 12 * abs(jpm) ** 3,
                    "Cmax_per_site": float(C.max()),
                })
    print(f"loaded cached curves from {path}", flush=True)
    return T, records, arrays


def compute_curves():
    cache = HERE / "recomputed_specific_heat_curves.npz"
    cached = load_cached_curves(cache)
    if cached is not None:
        return cached

    T = np.geomspace(1e-4, 0.09, 1000)
    jpms = [-0.10, -0.05, 0.05]
    clusters = [
        ("cubic", (1, 1, 1), "16-site cubic"),
        ("fcc", (2, 2, 2), "32-site FCC"),
    ]
    data = {"T": T}
    records = []
    for basis, shape, label in clusters:
        cl = R.build_cluster(basis, shape)
        print(f"built {label}: N={cl.n_sites}, ice={cl.n_ice}", flush=True)
        pt = R.sw_order23(cl, verbose=True)
        print(f"rows {label}: H2={len(pt['H2']['c'])}, H3={len(pt['H3']['c'])}", flush=True)
        for jpm in jpms:
            for mode in ("all", "delta0"):
                H = R.assemble(cl, pt, jpm, mode)
                E = np.linalg.eigvalsh(H)
                C = R.specific_heat(E, T) / cl.n_sites
                Tpk = R.refined_peak(T, C)
                key = curve_key(basis, shape, jpm, mode)
                data[f"{key}_C_per_site"] = C
                data[f"{key}_E"] = E
                records.append({
                    "key": key,
                    "basis": basis,
                    "shape": shape,
                    "label": label,
                    "n_sites": cl.n_sites,
                    "Jpm": jpm,
                    "mode": mode,
                    "Tpk": Tpk,
                    "g4": 4 * jpm * jpm,
                    "ghex": 12 * abs(jpm) ** 3,
                    "Cmax_per_site": float(C.max()),
                })
                print(
                    f"{label} Jpm={jpm:+.2f} {mode:6s}: "
                    f"Tpk={Tpk:.6g}, Cmax/N={C.max():.6g}",
                    flush=True,
                )
    np.savez_compressed(cache, **data)
    return T, records, data


def main():
    T, records, arrays = compute_curves()
    jpms = [-0.10, -0.05, 0.05]
    cluster_rows = [("cubic", "16-site cubic"), ("fcc", "32-site FCC")]

    fig, axes = plt.subplots(2, 3, figsize=(12.0, 6.8), sharex=True)
    for ir, (basis, row_label) in enumerate(cluster_rows):
        for jc, jpm in enumerate(jpms):
            ax = axes[ir, jc]
            for mode, color, label in (
                ("all", C_BARE, "bare"),
                ("delta0", C_CLEAN, r"clean ($\delta=0$)"),
            ):
                rec = next(
                    r for r in records
                    if r["basis"] == basis and abs(r["Jpm"] - jpm) < 1e-12 and r["mode"] == mode
                )
                C = arrays[f"{rec['key']}_C_per_site"]
                ax.plot(T, C, color=color, lw=2.0, label=label)
                ax.axvline(rec["Tpk"], color=color, lw=0.8, ls=":", alpha=0.9)
            g4 = 4 * jpm * jpm
            ghex = 12 * abs(jpm) ** 3
            ax.axvline(g4, color="black", lw=0.9, ls="-.", alpha=0.75)
            ax.axvline(ghex, color=C_HEX, lw=1.0, ls="--", alpha=0.85)
            if ir == 0:
                ax.set_title(rf"$J_\pm={jpm:+.2f}\,J_{{zz}}$")
            if jc == 0:
                ax.set_ylabel(row_label + "\n" + r"$C(T)/N$")
            if ir == 1:
                ax.set_xlabel(r"$T/J_{zz}$")
            ax.set_xscale("log")
            ax.grid(alpha=0.18, lw=0.6)
            ax.spines[["top", "right"]].set_visible(False)
            if ir == 0 and jc == 0:
                handles = [
                    plt.Line2D([0], [0], color=C_BARE, lw=2.0, label="bare"),
                    plt.Line2D([0], [0], color=C_CLEAN, lw=2.0, label=r"clean ($\delta=0$)"),
                    plt.Line2D([0], [0], color="black", lw=0.9, ls="-.", label=r"$g_4$"),
                    plt.Line2D([0], [0], color=C_HEX, lw=1.0, ls="--", label=r"$g_{\rm hex}$"),
                ]
            ymax = ax.get_ylim()[1]
            ax.text(g4, 0.92 * ymax, r"$g_4$", ha="center", va="top", fontsize=7.5, color="black")
            ax.text(ghex, 0.78 * ymax, r"$g_{\rm hex}$", ha="center", va="top", fontsize=7.5, color=C_HEX)

    fig.suptitle("Specific heat curves for every independently recomputed scenario", y=0.995, fontsize=13)
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.955),
               ncol=4, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    for ext in ("pdf", "png"):
        out = FIGS / f"fig_specific_heat_all_cases.{ext}"
        fig.savefig(out)
        print("wrote", out)
    plt.close(fig)


if __name__ == "__main__":
    main()
