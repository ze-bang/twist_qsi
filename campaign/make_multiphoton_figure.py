"""Waterfall figure: exact S^zz(q, omega) lineshapes vs the Gaussian line.

One trace per inequivalent momentum: the full broadened structure factor
(blue), the part beyond the leading doublet (green fill = odd-photon
continuum), and the Gaussian prediction (orange dashed line at omega_G,
carrying ALL the weight in that theory).

Data: ring48_full_spectrum.npz + ring48_gaussian_match.json
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "campaign" / "outputs" / "ring_model_dssf"
OUTDIR = ROOT / "paper_multiphoton" / "figs"
OUTDIR.mkdir(parents=True, exist_ok=True)

BLUE, GREEN, ORANGE = "#2a78d6", "#1baf7a", "#eb6834"
INK, INK2, GRID = "#1a1a19", "#55554e", "#dedcd5"
ETA = 0.07          # broadening of the exact sticks, units of K

LABELS = {
    3.0000: r"$(\frac{1}{3}\frac{1}{3}\frac{1}{3})$",
    3.6056: r"$(\frac{1}{3}\frac{1}{3}\frac{2}{3})$",
    3.4641: r"$L$",
    3.8730: r"$(\frac{1}{6}\frac{1}{6}\frac{5}{6})$",
    4.0000: r"$X$",
}
ORDER = [3.0000, 3.6056, 3.4641, 3.8730, 4.0000]


def broaden(grid, energies, weights):
    out = np.zeros_like(grid)
    for e, w in zip(energies, weights):
        if w > 1e-12:
            out += w * np.exp(-0.5 * ((grid - e) / ETA) ** 2)
    return out / (ETA * np.sqrt(2 * np.pi))


def main():
    z = np.load(D / "ring48_full_spectrum.npz", allow_pickle=True)
    ge, gw = z["group_energies"], z["group_weights"]
    gmatch = json.load(open(D / "ring48_gaussian_match.json"))
    scale = gmatch["scale"]
    by_m = {tuple(r["m"]): r for r in gmatch["rows"]}
    m_indices = [tuple(int(v) for v in m) for m in z["m_indices"]]

    # one representative momentum per sigma class
    reps = {}
    for qi, m in enumerate(m_indices):
        r = by_m.get(m)
        if r is None:
            continue
        reps.setdefault(round(r["sigma"][0], 4), (qi, r))

    grid = np.linspace(0.0, 5.4, 1400)
    fig, ax = plt.subplots(figsize=(3.4, 4.9))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#fcfcfb")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)

    step = 1.42
    for row, key in enumerate(reversed(ORDER)):
        qi, r = reps[key]
        w = gw[:, qi]
        tot = w.sum()
        base = row * step
        full = broaden(grid, ge, w / tot)
        # continuum = everything beyond the two dominant groups
        top2 = np.argsort(w)[::-1][:2]
        wc = w.copy()
        wc[top2] = 0.0
        cont = broaden(grid, ge, wc / tot)
        norm = full.max()
        ax.fill_between(grid, base, base + full / norm, color=BLUE,
                        alpha=0.22, lw=0)
        ax.plot(grid, base + full / norm, color=BLUE, lw=1.1)
        ax.fill_between(grid, base, base + cont / norm, color=GREEN,
                        alpha=0.65, lw=0)
        # Gaussian: single doubly degenerate line carrying ALL the weight
        wg = scale * r["sigma"][0]
        ax.plot([wg, wg], [base, base + 1.08], color=ORANGE, lw=1.4,
                ls=(0, (4, 2)))
        f3 = 1.0 - w[top2].sum() / tot
        ax.annotate(f"{100*f3:.0f}\\%" if False else f"{100*f3:.0f}%",
                    (5.32, base + 0.44), ha="right", fontsize=7.6, color=INK)
        ax.annotate(LABELS[key], (-0.14, base + 0.40), ha="right",
                    fontsize=9, color=INK, annotation_clip=False)

    # legend, drawn by hand at the top
    ytop = len(ORDER) * step + 0.18
    ax.plot([0.15, 0.55], [ytop + 0.52] * 2, color=BLUE, lw=1.2)
    ax.fill_between([0.15, 0.55], ytop + 0.38, ytop + 0.66, color=BLUE,
                    alpha=0.22, lw=0)
    ax.annotate("exact $S^{zz}(\\mathbf{q},\\omega)$", (0.66, ytop + 0.52),
                fontsize=7.6, color=INK, va="center")
    ax.fill_between([0.15, 0.55], ytop - 0.02, ytop + 0.20, color=GREEN,
                    alpha=0.65, lw=0)
    ax.annotate("beyond the photon doublet\n($C$-odd residual weight)",
                (0.66, ytop + 0.09), fontsize=7.6, color=INK, va="center")
    ax.plot([3.62, 3.62], [ytop - 0.02, ytop + 0.68], color=ORANGE, lw=1.4,
            ls=(0, (4, 2)))
    ax.annotate("Gaussian theory:\none line, all weight,\n2-fold degenerate",
                (3.76, ytop + 0.33), fontsize=7.6, color=ORANGE, va="center")

    ax.set_xlim(-0.05, 5.4)
    ax.set_ylim(-0.12, ytop + 1.0)
    ax.set_yticks([])
    ax.set_xlabel(r"$\omega\,/\,K$", fontsize=9.5, color=INK)
    ax.tick_params(labelsize=8.5, colors=INK2, length=3)
    ax.annotate(r"$f_{\rm res}$", (5.32, len(ORDER) * step - 0.28),
                ha="right", fontsize=8, color=INK2)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUTDIR / f"fig_multiphoton.{ext}", dpi=300,
                    bbox_inches="tight")
    print("wrote", OUTDIR / "fig_multiphoton.pdf")


if __name__ == "__main__":
    main()
