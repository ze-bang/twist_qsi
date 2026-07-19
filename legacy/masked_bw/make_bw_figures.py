#!/usr/bin/env python3
"""Figures for the masked Brillouin-Wigner all-Jpm protocol (cubic-16 ladder).

House style matches make_figures.py.  Reads masked_bw_16site_results.json +
masked_bw_16site_curves.npz and writes to figs/:
  fig_bw_peaks.pdf         peak scaling: bare vs masked-BW vs g4/ghex
  fig_bw_ct.pdf            C(T) bare vs clean, controlled + beyond-wall
  fig_bw_diagnostics.pdf   new diagnostics vs old eta_min gate
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
FIGS = HERE / "figs"
FIGS.mkdir(exist_ok=True)

# ---- house style (same as make_figures.py) --------------------------------
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "mathtext.fontset": "cm",
    "font.family": "serif",
    "axes.linewidth": 0.9,
    "savefig.bbox": "tight",
    "savefig.dpi": 200,
})

C_ICE = "#27ae60"     # ice / physical / clean
C_ART = "#e67e22"     # artifact / bare
C_HEX = "#8e44ad"     # hexagon scale / Loewdin
C_GREY = "#7f8c8d"
C_RED = "#c0392b"


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(FIGS / f"{name}.{ext}")
    plt.close(fig)


recs = json.loads((HERE / "masked_bw_16site_results.json").read_text())
curves = np.load(HERE / "masked_bw_16site_curves.npz", allow_pickle=True)
T = curves["T"]

pi_recs = sorted((r for r in recs if r["jpm"] < 0), key=lambda r: abs(r["jpm"]))
aj = np.array([abs(r["jpm"]) for r in pi_recs])


def fig_peaks():
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    jj = np.geomspace(0.024, 0.36, 100)
    ax.plot(jj, 4 * jj**2, "--", color="black", lw=1.1,
            label=r"$g_4=4J_\pm^2/J_{zz}$")
    ax.plot(jj, 12 * jj**3, ":", color=C_HEX, lw=1.8,
            label=r"$g_{\mathrm{hex}}=12|J_\pm|^3/J_{zz}^2$")
    ax.plot(aj, [r["bare_peak"] for r in pi_recs], "o-", color=C_ART,
            ms=7, lw=1.6, label="bare low band")
    ok = [r for r in pi_recs if r["bw_clean_converged"] == 90]
    bad = [r for r in pi_recs if r["bw_clean_converged"] < 90]
    ax.plot([abs(r["jpm"]) for r in ok], [r["bw_clean_peak"] for r in ok],
            "s-", color=C_ICE, ms=7, lw=1.6, label="masked BW clean")
    if bad:
        ax.plot([abs(r["jpm"]) for r in bad],
                [r["bw_clean_peak"] for r in bad], "s", ms=8, mfc="none",
                mew=1.6, color=C_ICE, label="unconverged (sector dissolved)")
    lo = [r for r in pi_recs if r.get("loewdin_clean_peak")]
    ax.plot([abs(r["jpm"]) for r in lo], [r["loewdin_clean_peak"] for r in lo],
            "x", ms=9, mew=1.8, color=C_HEX,
            label="masked Löwdin (where defined)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$|J_\pm|/J_{zz}$  ($\pi$-flux)")
    ax.set_ylabel(r"$T_{\mathrm{peak}}/J_{zz}$")
    ax.set_title("Clean gauge peak follows the photon scale")
    ax.legend(fontsize=9, frameon=False, loc="upper left", handlelength=2.2)
    save(fig, "fig_bw_peaks")


def fig_ct():
    picks = [("-0.05", "controlled window"),
             ("-0.18", "beyond the old validity wall")]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for ax, (tag, subtitle) in zip(axes, picks):
        r = next(x for x in pi_recs if f"{x['jpm']:+.2f}" == tag)
        ax.plot(T, curves[f"{tag}/C_bare"] / 16, color=C_ART, lw=1.8,
                label="bare")
        ax.plot(T, curves[f"{tag}/C_bw_clean"] / 16, color=C_ICE, lw=1.8,
                label="masked BW clean")
        ax.axvline(r["g4"], color="black", ls="--", lw=1.0)
        ax.axvline(r["ghex"], color=C_HEX, ls=":", lw=1.6)
        ax.text(r["g4"] * 1.12, ax.get_ylim()[1] * 0.97, r"$g_4$", ha="left",
                va="top", fontsize=10)
        ax.text(r["ghex"] / 1.12, ax.get_ylim()[1] * 0.97,
                r"$g_{\mathrm{hex}}$", ha="right", va="top", fontsize=10,
                color=C_HEX)
        ax.set_xscale("log")
        ax.set_xlim(1e-5, 1)
        ax.set_xlabel(r"$T/J_{zz}$")
        ax.set_title(rf"$J_\pm={tag}\,J_{{zz}}$  ({subtitle})")
    axes[0].set_ylabel(r"$C(T)/N$  (ice manifold)")
    axes[0].legend(fontsize=10, frameon=False)
    fig.tight_layout()
    save(fig, "fig_bw_ct")


def fig_diagnostics():
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))

    ax = axes[0]
    ax.plot(aj, [r["eta_min_theta0"] for r in pi_recs], "X--", color=C_HEX,
            ms=8, lw=1.4, label=r"old gate: $\eta_{\min}$ (Löwdin Gram)")
    ax.plot(aj, [r["bw_clean_max_nonice_weight"] for r in pi_recs], "s-",
            color=C_ICE, ms=7, lw=1.6,
            label=r"BW: max non-ice weight $w^{\mathrm{out}}$")
    ax.axhline(1.0, color=C_GREY, lw=0.8)
    ax.set_xlabel(r"$|J_\pm|/J_{zz}$")
    ax.set_ylabel("diagnostic value")
    ax.set_ylim(-0.04, 1.09)
    ax.set_title("Dressing: smooth where the old gate collapses")
    ax.legend(fontsize=9.5, frameon=False, loc="center left")

    ax = axes[1]
    ax.plot(aj, [r["bw_clean_min_pole_gap"] for r in pi_recs], "o-",
            color=C_RED, ms=7, lw=1.6,
            label=r"$\min_n\,\mathrm{dist}(E_n,\,\mathrm{spec}\,QHQ)$")
    ax.plot(aj, [r["ghex"] for r in pi_recs], ":", color=C_HEX, lw=1.8,
            label=r"$g_{\mathrm{hex}}$")
    ax.set_yscale("log")
    ax.set_xlabel(r"$|J_\pm|/J_{zz}$")
    ax.set_ylabel(r"energy / $J_{zz}$")
    ax.set_title("Isolation: collapse at $-0.30$ is the physics")
    ax.legend(fontsize=9.5, frameon=False, loc="lower left")

    fig.tight_layout()
    save(fig, "fig_bw_diagnostics")


fig_peaks()
fig_ct()
fig_diagnostics()
print("wrote fig_bw_peaks / fig_bw_ct / fig_bw_diagnostics")
