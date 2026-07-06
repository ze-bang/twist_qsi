"""Spinon-sector analysis at the same level as the ice band: decompose the
exact O(lambda^2) response <s>_T into physical CHANNELS using the sector
band structure, from the alpha_selection histograms:

  channel PC ("pair creation")  : ice band (Sz=0)  <->  one-pair band (Sz=1)
  channel SH ("spinon hopping") : two-defect band (Sz=0) <-> one-pair band
                                  (Sz=1)  [X moves/repolarizes an existing
                                  pair: the intra-charge-manifold channel]
  channel HI ("higher")         : everything else.

alpha(T)/2lambda = d<s>/dT is then resolved per channel per drive: this
identifies exactly which microscopic process produces the charge-window
feature, and why the uniform drive lacks it.

Also prints the per-state intra-charge spectral weights
W_SH = sum_{intra} |X|^2 / N_charge  per drive: the quantitative content of
"a spinon carries staggered charge".

Outputs: figN12_channels.pdf + printed numbers.
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

C_TEAL = "#2a9d8f"; C_RED = "#d1495b"; C_PUR = "#6f42c1"
plt.rcParams.update({
    "font.size": 9, "axes.labelsize": 9.5, "axes.titlesize": 9,
    "legend.fontsize": 6.4, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "figure.dpi": 200, "savefig.bbox": "tight"})


def band_edges(E, start, width_guess=0.6):
    """Return (lo, hi) of the band starting at E[start], ending at the
    largest spectral gap within [E[start], E[start]+width_guess]."""
    lo = E[start]
    seg = E[(E >= lo) & (E <= lo + width_guess)]
    if len(seg) < 3:
        return lo, lo + width_guess
    gaps = np.diff(seg)
    k = int(np.argmax(gaps))
    return lo, float(seg[k]) + 1e-9


def kernel(Ea, Eb, T, ebin):
    b = 1.0 / T
    de = Eb - Ea
    return np.where(np.abs(de) < ebin / 2, b * np.exp(-b * Ea),
                    (np.exp(-b * Ea) - np.exp(-b * Eb))
                    / np.where(np.abs(de) < 1e-12, 1.0, de))


def main():
    Tgrid = np.geomspace(3e-3, 0.6, 140)
    fig, axs = plt.subplots(2, 3, figsize=(7.2, 4.4), sharex=True)
    for jr, J in enumerate((-0.05, +0.04)):
        d = np.load(OUT / f"alpha_selection_J{J:+.2f}.npz")
        e0, ebin, nbins = float(d["e0"]), float(d["ebin"]), int(d["nbins"])
        Ec = np.arange(nbins) * ebin
        E8 = d["E8"] - e0
        E9 = d["E9"] - e0
        # band windows
        ice_hi = float(E8[89]) + 1e-6
        W_BAND = 0.55
        c8_lo, c8_hi = float(E8[90]), float(E8[90]) + W_BAND
        c9_lo, c9_hi = float(E9[0]), float(E9[0]) + W_BAND
        print(f"J={J:+.2f}: ice band [0, {ice_hi:.3f}] (90 states); "
              f"Sz=0 two-defect band [{c8_lo:.3f},{c8_hi:.3f}] "
              f"({np.sum((E8>=c8_lo)&(E8<=c8_hi))} states); "
              f"Sz=1 one-pair band [{c9_lo:.3f},{c9_hi:.3f}] "
              f"({np.sum((E9>=c9_lo)&(E9<=c9_hi))} states)")
        mult = {8: 1.0, 9: 2.0, 10: 2.0, 11: 2.0}

        def Z_of(T):
            b = 1.0 / T
            return sum(m * np.exp(-b * (d[f"E{nu}"] - e0)).sum()
                       for nu, m in mult.items())

        for pc, pname in enumerate(("unif", "stag", "tri")):
            # split the (8,9) histogram into channels
            H89 = d[f"H_8_9_{pname}"]
            ii, jj = np.nonzero(H89)
            Ea, Eb_, w = Ec[jj], Ec[ii], H89[np.nonzero(H89)]
            in_ice8 = Ea <= ice_hi
            in_c8 = (Ea >= c8_lo) & (Ea <= c8_hi)
            in_c9 = (Eb_ >= c9_lo) & (Eb_ <= c9_hi)
            ch = {"PC": in_ice8 & in_c9, "SH": in_c8 & in_c9}
            ch["HI"] = ~(ch["PC"] | ch["SH"])
            # other sector pairs entirely -> HI
            extra = []
            for lo, hi in ((9, 10), (10, 11)):
                Hx = d[f"H_{lo}_{hi}_{pname}"]
                i2, j2 = np.nonzero(Hx)
                extra.append((Ec[j2], Ec[i2], Hx[np.nonzero(Hx)]))
            curves = {}
            for cname, m in ch.items():
                s = np.zeros_like(Tgrid)
                for it, T in enumerate(Tgrid):
                    num = np.sum(w[m] * kernel(Ea[m], Eb_[m], T, ebin))
                    if cname == "HI":
                        for (ea, eb, ww) in extra:
                            num += np.sum(ww * kernel(ea, eb, T, ebin))
                    s[it] = num / Z_of(T)
                curves[cname] = np.gradient(s, Tgrid)
            norm = max(np.max(np.abs(v)) for v in curves.values())
            ax = axs[jr, pc]
            for cname, col in (("PC", C_TEAL), ("SH", C_RED), ("HI", "0.6")):
                ax.plot(Tgrid, curves[cname] / norm, "-", color=col, lw=1.2,
                        label={"PC": "pair creation (ice$\\to$pair)",
                               "SH": "spinon hopping (intra-charge)",
                               "HI": "higher"}[cname])
            ax.axhline(0, color="0.85", lw=0.6)
            ax.set_xscale("log")
            if jr == 0:
                ax.set_title({"unif": "uniform", "stag": "staggered",
                              "tri": "[111] field"}[pname])
            if pc == 0:
                ax.set_ylabel(f"$J_\\pm={J:+.2f}$\n"
                              r"$\alpha$ channels (norm.)")
            if jr == 1:
                ax.set_xlabel(r"$T/J_{zz}$")
            if jr == 0 and pc == 0:
                ax.legend(frameon=False)

            # intra-charge spectral weight per charge state
            n_c9 = np.sum((E9 >= c9_lo) & (E9 <= c9_hi))
            W_SH = np.sum(w[ch["SH"]]) / max(n_c9, 1)
            W_PC = np.sum(w[ch["PC"]]) / 90.0
            print(f"   {pname:4s}: W_SH (intra-charge, per pair state) = "
                  f"{W_SH:8.3f};  W_PC (per ice state) = {W_PC:8.3f}")
    fig.savefig(FIGS / "figN12_channels.pdf")
    print("wrote figN12_channels.pdf")


if __name__ == "__main__":
    main()
