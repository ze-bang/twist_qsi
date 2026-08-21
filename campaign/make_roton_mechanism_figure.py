"""FIGURE 2.  The sharp diagnostic: a character map of every bright level.

(a) CHARACTER MAP.  Two independent, exactly computed axes:
      x = d(omega)/dV, the Hellmann-Feynman census of flippable
          hexagons the excitation adds (>0) or removes (<0);
      y = omega / omega_free, the energy in units of what the FREE
          photon theory assigns to that same momentum.
    A photon must sit near y = 1 (it is a photon) and at x < 0 (a flux
    twist depletes resonance).  Every photon-branch level does: they
    fill a band around the free-photon line with negative census.  The
    roton is alone -- half the free-photon energy and the only positive
    census in the spectrum.  Neither axis alone is decisive; their
    conjunction is.

(b) The raw data behind the x axis: level energies tracked through the
    Rokhsar-Kivelson bias V.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif": ["cmr10", "Computer Modern Roman", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "axes.unicode_minus": False,
})

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "campaign" / "outputs" / "ring_model_dssf"
OUTDIR = ROOT / "paper_multiphoton" / "figs"

INK, INK2, INK3 = "#0f172a", "#475569", "#94a3b8"
PHOTON, ROTON, GAUSS = "#4f46e5", "#e11d48", "#d97706"
EDGE = "#cbd5e1"

# free-photon energy per momentum, same lattice (Table I)
WG = {"(1/3,1/3,1/3)": 1.761, "(1/3,1/3,2/3)": 2.116, "L": 2.033,
      "(1/6,1/6,5/6)": 2.273, "X": 2.347}


def key_of(star):
    s = str(star)
    if s in ("L", "X"):
        return s
    try:
        a = np.sort(np.abs(np.asarray(
            eval(s.replace("?", "")), dtype=float)))
    except Exception:
        return None
    for k, ref in (("(1/3,1/3,1/3)", [1 / 3, 1 / 3, 1 / 3]),
                   ("(1/3,1/3,2/3)", [1 / 3, 1 / 3, 2 / 3]),
                   ("(1/6,1/6,5/6)", [1 / 6, 1 / 6, 5 / 6])):
        if np.allclose(a, ref, atol=3e-3):
            return k
    return None


def load_levels():
    p = D / "rk_slopes.json"
    if not p.exists():
        raise SystemExit("rk_slopes.json missing - run rk_slopes_table.py")
    out = []
    for r in json.load(open(p))["rows"]:
        k = key_of(r["star"])
        if k is None:
            continue
        out.append((k, int(r["branch"]), float(r["omega"]),
                    float(r["fit"]), float(r["census"])))
    return out


def panel_map(ax, levels):
    xs_ph, ys_ph, xs_ro, ys_ro = [], [], [], []
    for k, b, om, slope, cen in levels:
        y = om / WG[k]
        roton = (k == "L" and om < 1.4)
        (xs_ro if roton else xs_ph).append(cen)
        (ys_ro if roton else ys_ph).append(y)

    if xs_ph:
        x0, x1 = min(xs_ph) - 0.05, max(xs_ph) + 0.05
        y0, y1 = min(ys_ph) - 0.05, max(ys_ph) + 0.05
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0,
                               facecolor="#e7ebf2", edgecolor="none",
                               zorder=0))
        ax.text(x1, y1 + 0.04, "what a photon does", fontsize=7.8,
                color=INK2, ha="right")

    ax.axhline(1.0, color=GAUSS, ls="--", lw=1.4, zorder=1)
    ax.text(0.12, 1.03, "free-photon energy", fontsize=7.6, color=GAUSS)
    ax.axvline(0.0, color=EDGE, lw=1.0, zorder=1)

    ax.scatter(xs_ph, ys_ph, s=52, color=PHOTON, zorder=4,
               edgecolor="white", lw=0.7)
    ax.scatter(xs_ro, ys_ro, s=115, color=ROTON, zorder=5,
               edgecolor="white", lw=0.9)
    if xs_ro:
        ax.annotate("roton", xy=(xs_ro[0] - 0.02, ys_ro[0] + 0.02),
                    xytext=(xs_ro[0] - 0.34, ys_ro[0] + 0.20),
                    fontsize=10.5, color=ROTON,
                    arrowprops=dict(arrowstyle="->", color=ROTON,
                                    lw=1.3))
    ax.set_xlabel(r"flippable hexagons added, $\ \Delta\langle n^{\rm flip}\rangle$",
                  fontsize=9.3, color=INK)
    ax.set_ylabel(r"$\omega\,/\,\omega_{\rm free\ photon}$",
                  fontsize=9.5, color=INK)
    ax.set_ylim(0.30, 1.72)
    ax.tick_params(labelsize=8.5, colors=INK2, length=3)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(EDGE)
    ax.set_title("(a) every bright level, on two exact axes",
                 loc="left", fontsize=9.6, color=INK)


def panel_knob(ax, levels):
    V = np.linspace(-0.55, 0.55, 5)
    for k, b, om, slope, cen in levels:
        roton = (k == "L" and om < 1.4)
        c = ROTON if roton else PHOTON
        ax.plot(V, om + slope * V, "-", color=c,
                lw=2.3 if roton else 1.3, alpha=1.0 if roton else 0.6,
                zorder=4 if roton else 2)
    ax.axvline(0, color=EDGE, lw=0.9, zorder=0)
    ro = [r for r in levels if r[0] == "L" and r[2] < 1.4]
    if ro:
        ax.text(0.60, ro[0][2] + ro[0][3] * 0.55,
                f"roton\n(slope ${ro[0][3]:+.2f}$)", fontsize=8.3,
                color=ROTON, va="center")
    ax.text(0.60, 2.75, "photon branches", fontsize=8.3,
            color=PHOTON, va="center")
    ax.set_xlim(-0.62, 1.32)
    ax.set_xticks([-0.5, 0, 0.5])
    ax.set_xlabel(r"Rokhsar--Kivelson bias $\ V/K$", fontsize=9.5,
                  color=INK)
    ax.set_ylabel(r"level energy $\ \omega/K$", fontsize=9.5, color=INK)
    ax.tick_params(labelsize=8.5, colors=INK2, length=3)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(EDGE)
    ax.set_title("(b) the raw data behind the census", loc="left",
                 fontsize=9.6, color=INK)


def main():
    levels = load_levels()
    print(f"{len(levels)} dominant levels:")
    for k, b, om, slope, cen in sorted(levels,
                                       key=lambda r: r[2] / WG[r[0]]):
        print(f"  {k:<16s} br{b}  omega={om:6.3f}  "
              f"ratio={om / WG[k]:5.3f}  slope={slope:+6.3f}  "
              f"census={cen:+6.3f}")
    ph = [r for r in levels if not (r[0] == "L" and r[2] < 1.4)]
    print(f"\nphoton ratio range  [{min(r[2]/WG[r[0]] for r in ph):.3f},"
          f" {max(r[2]/WG[r[0]] for r in ph):.3f}]")
    print(f"photon slope range  [{min(r[3] for r in ph):+.3f},"
          f" {max(r[3] for r in ph):+.3f}]")

    fig, ax = plt.subplots(2, 1, figsize=(3.4, 5.4))
    fig.patch.set_facecolor("white")
    panel_map(ax[0], levels)
    panel_knob(ax[1], levels)
    fig.subplots_adjust(left=0.20, right=0.965, top=0.945, bottom=0.088,
                        hspace=0.46)
    for ext in ("pdf", "png"):
        fig.savefig(OUTDIR / f"fig_mechanism.{ext}", dpi=300,
                    bbox_inches="tight")
    print("wrote", OUTDIR / "fig_mechanism.pdf")


if __name__ == "__main__":
    main()
