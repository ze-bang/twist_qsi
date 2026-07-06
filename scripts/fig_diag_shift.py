"""figN7: the per-state second-order shift -lambda^2 s_n of the ice band
against the band energy E_n, for one pi-flux and one 0-flux coupling.
The opposite slopes ARE the softening/hardening asymmetry."""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import identity
from scipy.sparse.linalg import cg

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ice_pt_lib as ipl
import exact_ed_lib as eel
from kappa_lambda2_exact import band_states, build_Xplus, CACHE, FIGS, NB

C_PI = "#1f6feb"; C_ZERO = "#e08e0b"
plt.rcParams.update({
    "font.size": 9, "axes.labelsize": 9.5, "axes.titlesize": 9.5,
    "legend.fontsize": 7.2, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "figure.dpi": 200, "savefig.bbox": "tight"})

cl = ipl.build_cluster("cubic", (1, 1, 1))
B0 = eel.SzBasis(cl)
B1 = eel.SzBasis(cl, nup=cl.n_sites // 2 + 1)
Xp = build_Xplus(B0, B1)

fig, ax = plt.subplots(figsize=(3.6, 2.9))
for J, col, mk, lab in ((-0.05, C_PI, "o", r"$\pi$-flux, $J_\pm=-0.05$"),
                        (+0.04, C_ZERO, "s", r"$0$-flux, $J_\pm=+0.04$")):
    E, Psi = band_states(B0, J)
    E0abs = E[0]
    Eb = (E - E0abs)[:NB]
    Psik = Psi[:, :NB]
    H1 = B1.H_xxz(J).real.tocsr()
    X = Xp @ Psik
    s = np.empty(NB)
    for n in range(NB):
        A = H1 - (E0abs + Eb[n]) * identity(B1.dim, format="csr")
        y, info = cg(A, X[:, n], rtol=1e-8, maxiter=2000)
        s[n] = 2.0 * float(X[:, n] @ y)      # both Sz sectors
    d = -(s - s.mean())                      # state-dependent part of the shift
    ax.plot(Eb / Eb.max(), d, mk, color=col, ms=3.5, mfc="none", label=lab)
    # guide: linear fit
    p = np.polyfit(Eb / Eb.max(), d, 1)
    xx = np.linspace(0, 1, 10)
    ax.plot(xx, np.polyval(p, xx), "-", color=col, lw=1, alpha=0.6)
ax.axhline(0, color="k", lw=0.7)
ax.set_xlabel(r"band energy $E_n/E_{\rm max}$")
ax.set_ylabel(r"state-dep. shift $-(s_n-\bar s)$   [$\lambda^2 J_{zz}$]")
ax.legend(frameon=False, fontsize=7)
ax.set_title("second-order shift across the ice band", fontsize=9)
fig.savefig(FIGS / "figN7_diagshift.pdf")
print("wrote figN7_diagshift.pdf")
