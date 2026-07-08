#!/usr/bin/env python3
"""Result figures for the PRL (paper/figs/).  Uses the same from-scratch ED
module as the pedagogical notes, so every number is reproducible."""
from __future__ import annotations
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import recompute_finite_size_artifact as R

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "paper" / "figs"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
    "mathtext.fontset": "cm", "font.family": "serif",
    "axes.linewidth": 0.8, "savefig.bbox": "tight", "savefig.dpi": 300,
    "legend.fontsize": 7.5, "xtick.labelsize": 8, "ytick.labelsize": 8,
})
C_BARE = "#e67e22"
C_CLEAN = "#27ae60"
C_HEX = "#8e44ad"


def entropy(E, T):
    E = np.asarray(E, float); E = E - E.min()
    b = 1.0 / T[:, None]
    w = np.exp(-b * E[None, :]); Z = w.sum(1)
    Emean = (w * E[None, :]).sum(1) / Z
    return np.log(Z) + Emean / T


def build():
    data = {}
    for basis, shape in (("cubic", (1, 1, 1)), ("fcc", (2, 2, 2))):
        cl = R.build_cluster(basis, shape)
        pt = R.sw_order23(cl, verbose=False)
        data[(basis, shape)] = (cl, pt)
        print("built", basis, shape, "ice", cl.n_ice, flush=True)
    return data


def fig_results(data):
    cl16, pt16 = data[("cubic", (1, 1, 1))]
    cl32, pt32 = data[("fcc", (2, 2, 2))]
    T = np.geomspace(1e-4, 0.25, 900)

    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.5))

    # ---- (a) C(T) FCC-32, bare vs clean ----
    ax = axes[0]
    jpm = -0.05
    for mode, col, lab in (("all", C_BARE, "bare"), ("delta0", C_CLEAN, "clean")):
        E = np.linalg.eigvalsh(R.assemble(cl32, pt32, jpm, mode))
        C = R.specific_heat(E, T) / cl32.n_sites
        ax.plot(T, C, color=col, lw=1.8, label=lab)
    g4 = 4 * jpm ** 2; ghex = 12 * abs(jpm) ** 3
    for v, t, c in ((g4, r"$g_4$", C_BARE), (ghex, r"$g_{\rm hex}$", C_HEX)):
        ax.axvline(v, color=c, ls="--", lw=0.9, alpha=0.7)
    ax.set_xscale("log")
    ax.set_xlabel(r"$T/J_{zz}$"); ax.set_ylabel(r"$C/N$")
    ax.set_title(r"(a) 32 sites, $J_\pm=-0.05$", loc="left")
    ax.legend(frameon=False, loc="upper right")
    ax.text(g4, ax.get_ylim()[1]*0.9, r"$g_4$", color=C_BARE, fontsize=8, ha="center")
    ax.text(ghex, ax.get_ylim()[1]*0.9, r"$g_{\rm hex}$", color=C_HEX, fontsize=8, ha="center")

    # ---- (b) T_peak/g_hex vs |Jpm|: artifact scaling ----
    ax = axes[1]
    jgrid = np.array([-0.12, -0.10, -0.08, -0.06, -0.05, -0.04, -0.03])
    for cl, pt, mk, name in ((cl16, pt16, "o", "16"), (cl32, pt32, "s", "32")):
        bare, clean = [], []
        for j in jgrid:
            ghex = 12 * abs(j) ** 3
            Eb = np.linalg.eigvalsh(R.assemble(cl, pt, j, "all"))
            Ec = np.linalg.eigvalsh(R.assemble(cl, pt, j, "delta0"))
            bare.append(R.refined_peak(T, R.specific_heat(Eb, T)) / ghex)
            clean.append(R.refined_peak(T, R.specific_heat(Ec, T)) / ghex)
        ax.plot(np.abs(jgrid), bare, mk + "-", color=C_BARE, ms=3.5, lw=1.0,
                label=f"bare, N={name}")
        ax.plot(np.abs(jgrid), clean, mk + "-", color=C_CLEAN, ms=3.5, lw=1.0,
                label=f"clean, N={name}")
    jj = np.linspace(0.028, 0.125, 100)
    ax.plot(jj, 1.0 / (3 * jj), "k:", lw=1.0, label=r"$J_{zz}/3|J_\pm|$")
    ax.set_yscale("log")
    ax.set_xlabel(r"$|J_\pm|/J_{zz}$"); ax.set_ylabel(r"$T_{\rm peak}/g_{\rm hex}$")
    ax.set_title("(b) artifact scaling", loc="left")
    ax.legend(frameon=False, loc="upper right", ncol=1)

    # ---- (c) entropy release ----
    ax = axes[2]
    jpm = -0.05
    for mode, col, lab in (("all", C_BARE, "bare"), ("delta0", C_CLEAN, "clean")):
        E = np.linalg.eigvalsh(R.assemble(cl32, pt32, jpm, mode))
        S = entropy(E, T) / cl32.n_sites
        ax.plot(T, S, color=col, lw=1.8, label=lab)
    ax.axhline(np.log(cl32.n_ice) / cl32.n_sites, color="grey", ls=":", lw=0.9)
    ax.text(1.2e-4, np.log(cl32.n_ice)/cl32.n_sites*1.02,
            r"$\ln(\#\mathrm{ice})/N$", color="grey", fontsize=7)
    ax.set_xscale("log")
    ax.set_xlabel(r"$T/J_{zz}$"); ax.set_ylabel(r"$S/N$")
    ax.set_title(r"(c) photon entropy", loc="left")
    ax.legend(frameon=False, loc="lower right")

    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(w_pad=1.3)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig_results.{ext}")
    plt.close(fig)
    print("wrote fig_results")


def fig_concept():
    """Concept figure for the PRL: scale separation + census, self-contained."""
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.5))
    ax = axes[0]
    jpm = np.linspace(0.01, 0.2, 300); Jzz = 1.0
    ax.plot(jpm, 4 * jpm ** 2, color=C_BARE, lw=1.8,
            label=r"$g_4=4J_\pm^2/J_{zz}$ (winding, 2nd order)")
    ax.plot(jpm, 12 * jpm ** 3, color=C_HEX, lw=1.8,
            label=r"$g_{\rm hex}=12|J_\pm|^3/J_{zz}^2$ (photon, 3rd order)")
    ax.fill_between(jpm, 12 * jpm ** 3, 4 * jpm ** 2, color=C_BARE, alpha=0.08)
    ax.set_yscale("log"); ax.set_xlabel(r"$|J_\pm|/J_{zz}$"); ax.set_ylabel(r"scale $/J_{zz}$")
    ax.set_title("(a) the artifact dominates", loc="left")
    ax.legend(frameon=False, loc="lower right")

    # census bar
    ax = axes[1]
    labels = ["winding\n4-loop", "wrapping\nhexagon", "physical\nhexagon"]
    corner = [0.5, 0.5, 1.0]; d0 = [0.0, 0.0, 1.0]
    x = np.arange(3); bw = 0.36
    ax.bar(x - bw/2, corner, bw, color="#7f8c8d", label="twist avg.")
    ax.bar(x + bw/2, d0, bw, color=C_CLEAN, label=r"$\delta=0$")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_ylabel("coupling kept"); ax.set_ylim(0, 1.15)
    ax.set_title("(b) what each scheme keeps", loc="left")
    ax.legend(frameon=False, loc="upper left")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(w_pad=1.3)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig_concept.{ext}")
    plt.close(fig)
    print("wrote fig_concept")


if __name__ == "__main__":
    data = build()
    fig_concept()
    fig_results(data)
    print("done ->", OUT)
