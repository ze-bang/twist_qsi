"""Schematic: the two perfect matchings (virtual paths) of a boundary-winding
four-loop. Matching A uses only bulk bonds (untwistable, N_A = 0); matching B
contains the boundary crossing (N_B = w). The ring coefficient is
c(phi) = -(g4/2) [e^{-i N_A.phi} + e^{-i N_B.phi}]
       = -(g4/2) e^{-i N_A.phi} (1 + e^{-i w.phi}).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

FIGS = Path(__file__).resolve().parents[1] / "paper" / "figs"

C_A = "#1f6feb"   # bulk matching
C_B = "#d1495b"   # boundary matching

fig, axs = plt.subplots(1, 2, figsize=(6.8, 2.7))

# site positions: square loop a-b-c-d; bond b-c crosses the right boundary
pos = {"a": (0.62, 0.25), "b": (0.86, 0.42), "c": (1.14, 0.62), "d": (0.66, 0.68)}
pos_c_img = (0.14, 0.62)   # periodic image of c inside the cell

for ax, matching, col, title in (
        (axs[0], (("a", "b"), ("c", "d")), C_A,
         r"matching $A$: bulk bonds only, $\mathbf{N}_A=\mathbf{0}$"),
        (axs[1], (("b", "c"), ("d", "a")), C_B,
         r"matching $B$: uses the crossing, $\mathbf{N}_B=\mathbf{w}$")):
    # cell
    ax.add_patch(plt.Rectangle((0, 0), 1, 1, fill=False, lw=1.2, color="0.3"))
    ax.axvline(1.0, color="0.3", lw=1.2, ls="--")
    ax.text(1.01, 0.02, "boundary", rotation=90, fontsize=7, color="0.35")
    # loop bonds (all four, light)
    loop = [("a", "b"), ("b", "c"), ("c", "d"), ("d", "a")]
    for (u, v) in loop:
        pu = np.array(pos[u])
        pv = np.array(pos[v] if v != "c" else pos["c"])
        crosses = (u, v) == ("b", "c") or (u, v) == ("c", "d")
        if (u, v) == ("b", "c"):
            ax.plot([pu[0], pv[0]], [pu[1], pv[1]], color="0.75", lw=1.4)
        elif (u, v) == ("c", "d"):
            pci = np.array(pos_c_img)
            ax.plot([pci[0], pos["d"][0]], [pci[1], pos["d"][1]],
                    color="0.75", lw=1.4)
        else:
            ax.plot([pu[0], pv[0]], [pu[1], pv[1]], color="0.75", lw=1.4)
    # highlight the matching dimers
    for (u, v) in matching:
        if (u, v) == ("b", "c"):
            pu, pv = np.array(pos["b"]), np.array(pos["c"])
            ax.plot([pu[0], pv[0]], [pu[1], pv[1]], color=col, lw=3.2,
                    solid_capstyle="round", zorder=5)
        elif (u, v) == ("c", "d"):
            pci = np.array(pos_c_img)
            ax.plot([pci[0], pos["d"][0]], [pci[1], pos["d"][1]], color=col,
                    lw=3.2, solid_capstyle="round", zorder=5)
        else:
            pu, pv = np.array(pos[u]), np.array(pos[v])
            ax.plot([pu[0], pv[0]], [pu[1], pv[1]], color=col, lw=3.2,
                    solid_capstyle="round", zorder=5)
    # sites
    for k, p in pos.items():
        if k == "c":
            ax.plot(*pos_c_img, "o", ms=7, mfc="white", mec="k", zorder=6)
            ax.annotate("c", pos_c_img, textcoords="offset points",
                        xytext=(-9, 2), fontsize=9)
            ax.plot(*p, "o", ms=7, mfc="0.9", mec="0.5", zorder=6)
            ax.annotate("c$'$", p, textcoords="offset points", xytext=(4, 2),
                        fontsize=9, color="0.45")
        else:
            ax.plot(*p, "o", ms=7, mfc="white", mec="k", zorder=6)
            ax.annotate(k, p, textcoords="offset points", xytext=(4, 3),
                        fontsize=9)
    ax.set_title(title, fontsize=9)
    ax.set_xlim(-0.05, 1.3)
    ax.set_ylim(-0.05, 1.05)
    ax.set_aspect("equal")
    ax.axis("off")

axs[0].text(0.02, -0.02,
            r"$S^+_aS^-_b\cdot S^+_cS^-_d$: no twisted bond touched",
            fontsize=7.5, color=C_A, transform=axs[0].transAxes)
axs[1].text(0.02, -0.02,
            r"$S^-_bS^+_c\cdot S^-_dS^+_a$: phase $e^{-i\mathbf{w}\cdot\varphi}$",
            fontsize=7.5, color=C_B, transform=axs[1].transAxes)

fig.suptitle(r"$c_C(\varphi) = -\frac{g_4}{2}\,"
             r"[e^{-i\mathbf{N}_A\cdot\varphi}+e^{-i\mathbf{N}_B\cdot\varphi}]"
             r" = -\frac{g_4}{2}\,(1+e^{-i\mathbf{w}\cdot\varphi})$",
             fontsize=10, y=1.06)
fig.savefig(FIGS / "fig_matchings.pdf", bbox_inches="tight")
print("wrote", FIGS / "fig_matchings.pdf")
