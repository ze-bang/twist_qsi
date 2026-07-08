#!/usr/bin/env python3
r"""B(Jpm) and gauge scale, ice-manifold bare (PBC) vs zero-transport clean."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NOTES = Path(__file__).resolve().parent
z = np.load(NOTES / "thermal_expansion_B_ice_manifold.npz")
J = z["jpms"]; Bb = z["B_bare"]; Bc = z["B_clean"]
tpb = z["tp0_bare"]; tpc = z["tp0_clean"]

fig, ax = plt.subplots(1, 2, figsize=(13, 5.2))

# (a) B(Jpm)
a = ax[0]
a.axhline(6, color="0.6", ls="--", lw=1.2, label=r"four-loop $|B_4|=6$")
a.axhline(15, color="#6a1b9a", ls="--", lw=1.3, label=r"hexagon $|B_6|=15$")
a.axhline(-6, color="0.6", ls="--", lw=1.2)
a.axhline(-15, color="#6a1b9a", ls="--", lw=1.3)
mpi, m0 = J < 0, J > 0
a.plot(J[mpi], Bc[mpi], "s-", color="#2e7d32", ms=8, lw=2.2, label=r"clean $\delta{=}0$, $\pi$-flux")
a.plot(J[m0], Bc[m0], "s-", color="#1b5e20", ms=8, lw=2.2, label=r"clean $\delta{=}0$, $0$-flux")
a.plot(J[mpi], Bb[mpi], "o", color="#ef6c00", ms=8, label=r"bare PBC, $\pi$-flux")
a.plot(J[m0], Bb[m0], "o-", color="#e65100", ms=8, lw=1.6, label=r"bare PBC, $0$-flux")
a.axhline(0, color="0.4", lw=0.8); a.axvline(0, color="0.4", lw=0.8)
a.set_xlabel(r"$J_\pm/J_{zz}$"); a.set_ylabel(r"$B=\kappa\,|J_\pm|\,J_{zz}$")
a.set_title(r"(a) $E_g$ softening $B$: clean projection $\to |B|=15$")
a.legend(fontsize=7.8, ncol=2, loc="center left"); a.grid(alpha=0.3)
a.set_ylim(-22, 22)

# (b) gauge scale vs |Jpm|
b = ax[1]
aj = np.abs(J)
b.loglog(aj, 4 * aj**2, color="0.6", ls="--", lw=1.2, label=r"$g_4=4J_\pm^2$")
b.loglog(aj, 12 * aj**3, color="#6a1b9a", ls="--", lw=1.3, label=r"$g_{\rm hex}=12|J_\pm|^3$")
b.loglog(aj[mpi], tpb[mpi], "o", color="#ef6c00", ms=7, label=r"bare $T_{\rm pk}$, $\pi$")
b.loglog(aj[m0], tpb[m0], "o", color="#e65100", ms=7, mfc="none", label=r"bare $T_{\rm pk}$, $0$")
b.loglog(aj[mpi], tpc[mpi], "s", color="#2e7d32", ms=7, label=r"clean $T_{\rm pk}$, $\pi$")
b.loglog(aj[m0], tpc[m0], "s", color="#1b5e20", ms=7, mfc="none", label=r"clean $T_{\rm pk}$, $0$")
b.set_xlabel(r"$|J_\pm|/J_{zz}$"); b.set_ylabel(r"gauge peak $T_{\rm pk}$")
b.set_title(r"(b) gauge scale: bare $\sim g_4$, clean $\sim g_{\rm hex}$")
b.legend(fontsize=8, ncol=1); b.grid(alpha=0.3, which="both")

fig.suptitle(r"Ice-manifold $E_g$ thermal expansion (16-site cubic): winding four-loops "
             r"(PBC) vs zero-transport $\delta{=}0$ projection", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.96])
out = NOTES / "figs" / "thermal_expansion_B_ice_manifold.png"
out.parent.mkdir(exist_ok=True)
fig.savefig(out, dpi=150)
print("saved", out)
