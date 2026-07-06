"""Re-render figN2 as ground-state ring ENERGY contributions
(coupling x expectation), where cooperation vs competition is visible as
the sign of the hexagon bars."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

GP = Path(__file__).resolve().parents[2] / "gauge_probe_prl"
FIGS = GP / "notes" / "figs"
C_PI = "#1f6feb"; C_ZERO = "#e08e0b"; C_RED = "#d1495b"

plt.rcParams.update({
    "font.size": 9, "axes.labelsize": 9.5, "axes.titlesize": 9.5,
    "legend.fontsize": 7.2, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "figure.dpi": 200, "savefig.bbox": "tight",
})

res = json.load(open(GP / "notes" / "kappa_flux_results.json"))
coh = res["coherence"]

fig, axs = plt.subplots(1, 2, figsize=(6.9, 2.8))
for ax, key, col, ttl in ((axs[0], "-0.1", C_PI, r"$\pi$-flux, $J_\pm=-0.10$"),
                          (axs[1], "0.05", C_ZERO, r"$0$-flux, $J_\pm=+0.05$")):
    J = float(key)
    g4 = 4 * J ** 2
    gam6 = 12 * J ** 3          # signed; dynamical term is -gam6 * Shex
    v4 = np.array(coh[key]["v4"])
    v6c = np.array(coh[key]["v6c"])
    v6w = np.array(coh[key]["v6w"])
    e4 = -g4 * v4 * 1e3          # meV-like units: 10^-3 Jzz
    e6c = -gam6 * v6c * 1e3
    e6w = -gam6 * v6w * 1e3
    rng = np.random.default_rng(3)
    ax.scatter(rng.uniform(-.12, .12, len(e4)), e4, s=14, color=C_RED,
               alpha=0.8, label="36 winding 4-loops")
    ax.scatter(1 + rng.uniform(-.12, .12, len(e6c)), e6c, s=14, color=col,
               alpha=0.85, label="16 contractible hexagons")
    ax.scatter(2 + rng.uniform(-.12, .12, len(e6w)), e6w, s=14, color="0.55",
               alpha=0.7, label="48 wrapping hexagons")
    ax.axhline(0, color="k", lw=0.7)
    ax.set_xticks([0, 1, 2], ["4-loops", "hex (contr.)", "hex (wrap)"])
    ax.set_title(ttl)
    ax.legend(frameon=False, fontsize=6.6, loc="center right")
axs[0].set_ylabel(r"GS ring energy per loop  [$10^{-3}J_{zz}$]")
fig.savefig(FIGS / "figN2_coherence.pdf")
print("re-rendered figN2 (energy contributions)")
