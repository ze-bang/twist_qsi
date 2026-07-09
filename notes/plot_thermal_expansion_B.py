#!/usr/bin/env python3
r"""B(Jpm): PBC (four-loop-contaminated) vs transported-dipole loop projection."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
d = np.load(HERE / "thermal_expansion_B_sweep.npz")
J = d["jpms"]; Bb = d["B_bare"]; Bc = d["B_clean"]

fig, ax = plt.subplots(figsize=(9, 6.2))
# leading references +-6 (four-loop) and +-15 (hexagon)
for y, lab, c in [(6, r"four-loop $|B|=6$", "0.55"), (15, r"hexagon $|B|=15$", "#2a7f2a")]:
    ax.plot([-0.105, -0.004], [y, y], ls="--", lw=1.2, color=c, zorder=1)
    ax.plot([0.004, 0.055], [-y, -y], ls="--", lw=1.2, color=c, zorder=1,
            label=lab if y == 15 else None)
    if y == 15:
        ax.plot([-0.105, -0.004], [y, y], ls="--", lw=1.2, color=c, zorder=1)

mpi, m0 = J < 0, J > 0
# PBC bare
ax.plot(J[mpi], Bb[mpi], "o-", color="#1f6feb", ms=7, lw=2, zorder=3,
        label=r"PBC (bare), $\pi$-flux")
ax.plot(J[m0], Bb[m0], "o-", color="#0b3d91", ms=7, lw=2, zorder=3,
        label=r"PBC (bare), $0$-flux")
# clean loop projection
ax.plot(J[mpi], Bc[mpi], "s-", color="#d1495b", ms=9, lw=2.4, zorder=4,
        label=r"loop-projected ($\delta{=}0$)")
ax.plot(J[m0], Bc[m0], "s-", color="#d1495b", ms=9, lw=2.4, zorder=4)

ax.axhline(0, color="0.4", lw=0.8); ax.axvline(0, color="0.4", lw=0.8)
ax.text(-0.052, 16.2, r"$\pi$-flux: $B_{\rm clean}=+15$", color="#d1495b", fontsize=9.5)
ax.text(0.006, -17.3, r"$0$-flux: $B_{\rm clean}=-15$", color="#d1495b", fontsize=9.5)
ax.annotate("removing the winding\nfour-loop lands $B$ on the\nhexagon value 15",
            xy=(-0.07, 15), xytext=(-0.099, 20.5), fontsize=9, color="#7a1f1f",
            arrowprops=dict(arrowstyle="-|>", color="#d1495b", lw=1.5))

ax.set_xlabel(r"$J_\pm/J_{zz}$  (signed:  $<0$ $\pi$-flux,  $>0$ $0$-flux)")
ax.set_ylabel(r"$B \equiv \kappa\,|J_\pm|\,J_{zz}$")
ax.set_title(r"Thermal-expansion softening $B(J_\pm)$ on the 16-site cluster:"
             "\nPBC vs transported-dipole loop projection (SW ice-manifold)")
ax.legend(fontsize=8.5, loc="lower left", ncol=1, framealpha=0.95)
ax.grid(alpha=0.3)
ax.set_ylim(-24, 24); ax.set_xlim(-0.105, 0.055)
fig.tight_layout()
out = HERE / "figs" / "thermal_expansion_B_pbc_vs_clean.png"
fig.savefig(out, dpi=150)
print("saved", out)
