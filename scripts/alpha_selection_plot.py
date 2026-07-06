"""Assemble alpha(T)/2lambda = d<s>_T/dT and exact C(T) from the histograms
produced by alpha_selection_full.py, and draw figN10: the form-factor
selection rule -- uniform source blind to the charge peak, staggered /
[111] sources reading it.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
GP = HERE.parents[1] / "gauge_probe_prl"
OUT = GP / "notes"
FIGS = OUT / "figs"

C_PI = "#1f6feb"; C_ZERO = "#e08e0b"; C_RED = "#d1495b"; C_TEAL = "#2a9d8f"
COLS = {"unif": C_TEAL, "stag": C_RED, "tri": "#6f42c1"}
LABS = {"unif": r"$E_g$ (uniform, $\sum f\neq0$)",
        "stag": r"$T_{2g}$ (staggered, $\sum f=0$)",
        "tri": r"$[111]$ field ($f_i=\hat n\!\cdot\!\hat z_i$, $\sum f=0$)"}

plt.rcParams.update({
    "font.size": 9, "axes.labelsize": 9.5, "axes.titlesize": 9,
    "legend.fontsize": 6.6, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "figure.dpi": 200, "savefig.bbox": "tight"})


def load(J):
    d = np.load(OUT / f"alpha_selection_J{J:+.2f}.npz")
    return d


def s_of_T(d, pname, Tgrid):
    """<s>_T from the pairwise histograms with the regular kernel."""
    e0, ebin, nbins = float(d["e0"]), float(d["ebin"]), int(d["nbins"])
    Ec = np.arange(nbins) * ebin                      # bin centers rel. GS
    # partition function with sector multiplicities
    mult = {8: 1.0, 9: 2.0, 10: 2.0, 11: 2.0}
    out = np.zeros_like(Tgrid)
    for it, T in enumerate(Tgrid):
        b = 1.0 / T
        Z = sum(m * np.exp(-b * (d[f"E{nu}"] - e0)).sum()
                for nu, m in mult.items())
        num = 0.0
        for lo, hi in ((8, 9), (9, 10), (10, 11)):
            H = d[f"H_{lo}_{hi}_{pname}"]
            ii, jj = np.nonzero(H)                     # (hi_bin, lo_bin)
            Ea, Eb_ = Ec[jj], Ec[ii]                   # lo energy, hi energy
            w = H[ii, jj]
            de = Eb_ - Ea
            ker = np.where(np.abs(de) < ebin / 2,
                           b * np.exp(-b * Ea),
                           (np.exp(-b * Ea) - np.exp(-b * Eb_))
                           / np.where(np.abs(de) < 1e-12, 1.0, de))
            num += np.sum(w * ker)
        out[it] = num / Z
    return out


def C_of_T_full(d, Tgrid):
    mult = {8: 1.0, 9: 2.0, 10: 2.0, 11: 2.0}
    e0 = float(d["e0"])
    Es = np.concatenate([np.repeat(d[f"E{nu}"] - e0, int(m) if m == 1 else 2)
                         if False else d[f"E{nu}"] - e0
                         for nu, m in mult.items()])
    ws = np.concatenate([np.full(len(d[f"E{nu}"]), m)
                         for nu, m in mult.items()])
    C = np.zeros_like(Tgrid)
    for it, T in enumerate(Tgrid):
        b = 1.0 / T
        w = ws * np.exp(-b * Es)
        Z = w.sum()
        m1 = (w * Es).sum() / Z
        m2 = (w * Es ** 2).sum() / Z
        C[it] = (m2 - m1 ** 2) / T ** 2
    return C


def main():
    Tgrid = np.geomspace(3e-3, 0.6, 160)
    fig, axs = plt.subplots(2, 2, figsize=(7.0, 4.6), sharex="col")
    for jc, J in enumerate((-0.05, +0.04)):
        d = load(J)
        C = C_of_T_full(d, Tgrid)
        axs[0, jc].plot(Tgrid, C, "k-", lw=1.2)
        axs[0, jc].set_xscale("log")
        axs[0, jc].set_title(f"$J_\\pm={J:+.2f}$ "
                             + (r"($\pi$-flux)" if J < 0 else r"($0$-flux)"))
        axs[0, jc].set_ylabel(r"$C(T)$" if jc == 0 else "")
        for pname in ("unif", "stag", "tri"):
            s = s_of_T(d, pname, Tgrid)
            ds = np.gradient(s, Tgrid)
            # normalize each curve by its |max| in the gauge window for shape
            axs[1, jc].plot(Tgrid, ds / np.max(np.abs(ds)), "-",
                            color=COLS[pname], lw=1.2, label=LABS[pname])
        axs[1, jc].axhline(0, color="0.8", lw=0.7)
        axs[1, jc].set_xscale("log")
        axs[1, jc].set_xlabel(r"$T/J_{zz}$")
        axs[1, jc].set_ylabel(r"$\alpha(T)/|\alpha|_{\max}$" if jc == 0 else "")
        if jc == 0:
            axs[1, jc].legend(frameon=False)
    fig.savefig(FIGS / "figN10_selection.pdf")
    print("wrote figN10_selection.pdf")

    # numbers for the note: charge-window feature size relative to gauge peak
    for J in (-0.05, +0.04):
        d = load(J)
        for pname in ("unif", "stag", "tri"):
            s = s_of_T(d, pname, Tgrid)
            ds = np.gradient(s, Tgrid)
            gauge = np.max(np.abs(ds[Tgrid < 0.06]))
            charge = np.max(np.abs(ds[(Tgrid > 0.12) & (Tgrid < 0.5)]))
            print(f"J={J:+.2f} {pname:4s}: |alpha|_gauge={gauge:9.3f}  "
                  f"|alpha|_charge={charge:9.3f}  ratio charge/gauge="
                  f"{charge/gauge:.3f}")


if __name__ == "__main__":
    main()
