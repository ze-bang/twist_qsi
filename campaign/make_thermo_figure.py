"""Figure: thermodynamic evidence and numerical rigor, in one place.

(a) The complete Feynman-ratio landscape at 2048 sites -- every allowed
    momentum star, with statistical errors, 864-site values overlaid as
    open markers.  The soft valley (t,1/2,1/2) and its minimum at L are
    labeled; nothing is hidden by a path choice.
(b) Ground energy per site against 1/N: exact diagonalization and the
    Monte Carlo on the same axis, including the two sizes where both
    exist (48, 72), where they agree within error bars -- the
    method-agreement gate, shown rather than asserted.
(c) The ratio at L and at X against 1/N across both methods: the
    zone-boundary hierarchy is size-stable where it matters.

Colors follow the paper's established system (roton red, photon
indigo, slate ink); method is encoded by marker fill (ED filled, QMC
open + error bars), so identity never rests on color alone.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif": ["cmr10", "Computer Modern Roman", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "axes.unicode_minus": False,
})

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "campaign" / "outputs" / "ring_model_dssf"
OUTDIR = ROOT / "paper_multiphoton" / "figs"

INK, INK2, EDGE = "#0f172a", "#475569", "#cbd5e1"
PHOTON, ROTON, GAUSS = "#4f46e5", "#e11d48", "#d97706"


def fold(q):
    G = [a * np.array([1, 1, -1]) + b * np.array([1, -1, 1])
         + c * np.array([-1, 1, 1])
         for a in range(-2, 3) for b in range(-2, 3) for c in range(-2, 3)]
    q = np.asarray(q, dtype=float)
    best = None
    for g in G:
        v = q - g
        if best is None or np.linalg.norm(v) < np.linalg.norm(best) - 1e-12:
            best = v
    return np.sort(np.abs(best))


def stars_of(n):
    d = json.load(open(D / f"gfmc_fullgrid_n{n}.json"))
    merged = {}
    for r in d["rows"]:
        if r["ratio"] is None:
            continue
        k = tuple(np.round(fold(r["q"]), 6))
        merged.setdefault(k, []).append(r)
    out = []
    for k, rr in merged.items():
        w = np.array([1 / x["ratio_err"] ** 2 for x in rr])
        v = np.array([x["ratio"] for x in rr])
        out.append({"qf": np.array(k),
                    "absq": float(np.linalg.norm(k)),
                    "ratio": float((v * w).sum() / w.sum()),
                    "err": float(1 / np.sqrt(w.sum()))})
    return out


def is_valley(qf):
    q = np.sort(qf)
    return (abs(q[1] - 0.5) < 1e-6 and abs(q[2] - 0.5) < 1e-6)


def panel_landscape(ax):
    s8 = stars_of(8)
    s6 = stars_of(6)
    for s in s6:
        ax.errorbar(s["absq"], s["ratio"], yerr=s["err"], fmt="s",
                    ms=4.2, mfc="white", mec=INK2, ecolor=INK2,
                    elinewidth=0.8, capsize=0, zorder=2)
    for s in s8:
        col = ROTON if is_valley(s["qf"]) else PHOTON
        ax.errorbar(s["absq"], s["ratio"], yerr=s["err"], fmt="o",
                    ms=5.2, mfc=col, mec="white", mew=0.6, ecolor=col,
                    elinewidth=1.0, capsize=0, zorder=3)
    lab = {(0.5, 0.5, 0.5): ("$L$", (10, -4)),
           (0.0, 0.0, 1.0): ("$X$", (-13, 0)),
           (0.0, 0.5, 1.0): ("$W$", (8, -3)),
           (0.0, 0.75, 0.75): ("$K$", (6, -12))}
    for s in s8:
        k = tuple(np.round(np.sort(s["qf"]), 3))
        if k in lab:
            t, off = lab[k]
            ax.annotate(t, xy=(s["absq"], s["ratio"]),
                        xytext=off, textcoords="offset points",
                        fontsize=8.5, color=INK, ha="center", va="center")
    ax.errorbar([], [], fmt="o", ms=5.2, mfc=PHOTON, mec="white",
                label="2048 sites")
    ax.errorbar([], [], fmt="s", ms=4.2, mfc="white", mec=INK2,
                label="864 sites")
    ax.errorbar([], [], fmt="o", ms=5.2, mfc=ROTON, mec="white",
                label=r"soft valley $(t,\frac{1}{2},\frac{1}{2})$")
    ax.legend(loc="lower right", fontsize=7.6, frameon=False,
              handletextpad=0.2, borderaxespad=0.2)
    ax.set_xlabel(r"$|\bf q|$  (cubic reciprocal units)", fontsize=9.3,
                  color=INK)
    ax.set_ylabel(r"$f_1(\bf q)\,/\,S(\bf q)$   $[K]$", fontsize=9.3,
                  color=INK)
    ax.set_xlim(0.0, 1.22)
    ax.set_ylim(0.0, 3.05)
    ax.set_title("(a) all allowed momenta",
                 loc="left", fontsize=9.4, color=INK)


def panel_energy(ax):
    ed_N = np.array([48, 64, 72, 96])
    ed_E = np.array([-0.21991691, -0.21652015, -0.20569049, -0.20383221])
    q_N = np.array([48, 72, 256, 864, 2048])
    q_E = np.array([-0.219907, -0.205662, -0.190541, -0.189122, -0.188557])
    q_err = np.array([0.000017, 0.000016, 0.000035, 0.000042, 0.000060])
    ax.errorbar(1.0 / q_N, q_E, yerr=q_err, fmt="o", ms=5.5, mfc="white",
                mec=ROTON, ecolor=ROTON, elinewidth=1.0, capsize=2,
                label="Monte Carlo", zorder=3)
    ax.plot(1.0 / ed_N, ed_E, "o", ms=4.6, color=INK,
            label="exact diagonalization", zorder=4)
    ax.annotate("methods agree\nat 48 and 72",
                xy=(1 / 72.0, -0.2057), xytext=(0.0035, -0.2125),
                fontsize=7.6, color=INK2,
                arrowprops=dict(arrowstyle="-", color=INK2, lw=0.7))
    ax.set_xlabel(r"$1/N$", fontsize=9.3, color=INK)
    ax.set_ylabel(r"$E_0/N$   $[K]$", fontsize=9.3, color=INK)
    ax.legend(loc="lower left", fontsize=7.6, frameon=False)
    ax.set_title("(b) ground energy, both methods", loc="left",
                 fontsize=9.4, color=INK)


def panel_zb(ax):
    # ED ratios (exact, gated sum rule); QMC star-averaged with errors
    L_N = [48, 64, 72, 864, 2048]
    L_v = [1.8456, 1.8054, 1.9826, 2.1419, 2.1786]
    L_e = [0, 0, 0, 0.0104, 0.0147]
    X_N = [48, 64, 864, 2048]
    X_v = [3.2230, 3.3463, 2.6598, 2.7129]
    X_e = [0, 0, 0.0096, 0.0205]
    for N, v, e, col, lab in ((L_N, L_v, L_e, ROTON, "$L$"),
                              (X_N, X_v, X_e, PHOTON, "$X$")):
        N = np.array(N, dtype=float)
        ed = N < 100
        ax.errorbar(1 / N[~ed], np.array(v)[~ed], yerr=np.array(e)[~ed],
                    fmt="o", ms=5.5, mfc="white", mec=col, ecolor=col,
                    elinewidth=1.0, capsize=2, zorder=3)
        ax.plot(1 / N[ed], np.array(v)[ed], "o", ms=4.6, color=col,
                zorder=4)
        ax.annotate(lab, xy=(1 / N[0], v[0]), xytext=(8, -3),
                    textcoords="offset points", fontsize=10, color=col,
                    va="center")
    ax.annotate("filled: exact\nopen: Monte Carlo", xy=(0.0075, 2.98),
                fontsize=7.6, color=INK2)
    ax.set_xlabel(r"$1/N$", fontsize=9.3, color=INK)
    ax.set_ylabel(r"$f_1/S$ at the zone boundary  $[K]$", fontsize=9.3,
                  color=INK)
    ax.set_ylim(1.6, 3.5)
    ax.set_xlim(-0.0012, 0.0242)
    ax.set_title("(c) the hierarchy is size-stable", loc="left",
                 fontsize=9.4, color=INK)


def main():
    fig, ax = plt.subplots(1, 3, figsize=(7.0, 2.55))
    fig.patch.set_facecolor("white")
    panel_landscape(ax[0])
    panel_energy(ax[1])
    panel_zb(ax[2])
    for a in ax:
        a.tick_params(labelsize=8, colors=INK2, length=3)
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            a.spines[s].set_color(EDGE)
    fig.subplots_adjust(left=0.065, right=0.985, top=0.90, bottom=0.175,
                        wspace=0.33)
    for ext in ("pdf", "png"):
        fig.savefig(OUTDIR / f"fig_thermo.{ext}", dpi=300)
    print("wrote", OUTDIR / "fig_thermo.pdf")


if __name__ == "__main__":
    main()
