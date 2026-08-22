"""Figure: the four-size spectroscopy ladder at the L point.

One column per cluster.  Filled markers: the two dominant levels of
the rotonizing L class, marker area proportional to the level's share
of the longitudinal weight at that momentum.  Open markers: the
dominant levels of the NON-rotonizing L class on the two clusters
whose anisotropy provides one -- the within-cluster control at the
same |q| and the same free-theory energy.  All numbers are exact
diagonalization (dense at 48; Lanczos with the sum-rule gate reading
1.0000 at 64, 72, 96).

Data provenance: ring_fixed_fcc223 (48, complete diagonalization),
ring64_results (64), ring128_fcc233 (72), ring96_fcc234 (96).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif": ["cmr10", "Computer Modern Roman", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "axes.unicode_minus": False,
})

ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "paper_multiphoton" / "figs"
INK, INK2, EDGE = "#0f172a", "#475569", "#cbd5e1"
PHOTON, ROTON = "#4f46e5", "#e11d48"

# (sites, [(omega, share, kind)]) kind: r=roton branch, p=upper branch,
# c=non-rotonizing class control
DATA = [
    (48, [(1.024, 0.508, "r"), (1.730, 0.323, "p")]),
    (64, [(0.954, 0.500, "r"), (1.652, 0.313, "p"),
          (1.998, 0.771, "c")]),
    (72, [(1.068, 0.540, "r"), (1.814, 0.178, "p")]),
    (96, [(0.904, 0.544, "r"), (1.647, 0.225, "p"),
          (1.651, 0.387, "c"), (1.730, 0.301, "c")]),
]
RATIO = {48: 0.592, 64: 0.578, 72: 0.589, 96: 0.549}


def main():
    fig, ax = plt.subplots(figsize=(3.4, 2.9))
    fig.patch.set_facecolor("white")
    xs = {48: 0, 64: 1, 72: 2, 96: 3}
    for sites, levels in DATA:
        x = xs[sites]
        for om, sh, kind in levels:
            if kind == "c":
                ax.scatter(x + 0.22, om, s=560 * sh, facecolor="white",
                           edgecolor=INK2, linewidth=1.1, zorder=3)
            else:
                col = ROTON if kind == "r" else PHOTON
                ax.scatter(x, om, s=560 * sh, color=col,
                           edgecolor="white", linewidth=0.7, zorder=4)
        ax.annotate(f"{RATIO[sites]:.3f}",
                    xy=(x, 0.32), fontsize=7.8, color=INK2, ha="center")
    # connect the roton and upper branches across sizes
    for kind, col in (("r", ROTON), ("p", PHOTON)):
        pts = [(xs[s], om) for s, ls in DATA for om, sh, k in ls
               if k == kind]
        ax.plot(*zip(*pts), "-", color=col, lw=1.1, alpha=0.45, zorder=2)
    ax.annotate("soft mode", xy=(3, 0.904), xytext=(2.05, 0.62),
                fontsize=8.6, color=ROTON,
                arrowprops=dict(arrowstyle="-", color=ROTON, lw=0.8))
    ax.annotate("upper transverse branch", xy=(0, 1.730),
                xytext=(-0.60, 2.12), fontsize=8.6, color=PHOTON,
                arrowprops=dict(arrowstyle="-", color=PHOTON, lw=0.8))
    ax.annotate("non-rotonizing\n$L$ class (control)", xy=(1.22, 1.998),
                xytext=(2.25, 2.30), fontsize=7.8, color=INK2,
                arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
    ax.annotate(r"$\omega_{\rm low}/\omega_{\rm high}$", xy=(-0.68, 0.47),
                fontsize=7.4, color=INK2)
    ax.set_xticks(list(xs.values()))
    ax.set_xticklabels(["48", "64", "72", "96"], fontsize=9)
    ax.set_xlabel("cluster size (spins)", fontsize=9.3, color=INK)
    ax.set_ylabel(r"level energy  $\omega/K$", fontsize=9.5, color=INK)
    ax.set_xlim(-0.75, 3.75)
    ax.set_ylim(0.2, 2.62)
    ax.tick_params(labelsize=8.5, colors=INK2, length=3)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(EDGE)
    ax.set_title("$L$-point spectroscopy across four cluster sizes",
                 loc="left", fontsize=9.4, color=INK)
    fig.subplots_adjust(left=0.135, right=0.975, top=0.92, bottom=0.145)
    for ext in ("pdf", "png"):
        fig.savefig(OUTDIR / f"fig_ladder.{ext}", dpi=300)
    print("wrote", OUTDIR / "fig_ladder.pdf")


if __name__ == "__main__":
    main()
