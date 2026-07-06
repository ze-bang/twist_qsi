"""
Visualise the projector kernel
   K[w] = (1/|G|) sum_{phi in G} exp(i w . phi)
for three averaging schemes:
  (A) full 8-corner average G = {0, pi}^3
  (B) green/olive subset G = {phi: |phi|_pi >= 2}  (only the corners that
      individually look 'right' on the 16-site cluster)
  (C) continuous average G = [0, 2pi)^3

The figure shows that scheme (A) gives a clean delta(w_alpha mod 2),
scheme (B) is not a projector at all (random +-1/2 for odd-winding
loops), scheme (C) is the textbook delta(w).
"""
from __future__ import annotations

from itertools import product
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIGS = ROOT / "paper" / "figs"

corners = list(product([0.0, np.pi], repeat=3))
green_olive = [c for c in corners if sum(1 for x in c if abs(x - np.pi) < 1e-6) >= 2]

w_max = 2
w_axis = np.arange(-w_max, w_max + 1)

K_full = np.zeros((len(w_axis), len(w_axis)))
K_go = np.zeros((len(w_axis), len(w_axis)))
for i, wx in enumerate(w_axis):
    for j, wy in enumerate(w_axis):
        w = (wx, wy, 0)
        K_full[i, j] = np.mean([np.cos(np.dot(w, c)) for c in corners])
        K_go[i, j] = np.mean([np.cos(np.dot(w, c)) for c in green_olive])

fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))
for ax, K, title in zip(
    axes,
    (K_full, K_go),
    (r"(a) full 8-corner average  $\frac{1}{8}\sum_{\varphi\in\{0,\pi\}^3} e^{i\mathbf{w}\cdot\varphi}$",
     r"(b) green/olive subset  $\frac{1}{4}\sum_{|\varphi|_\pi\geq 2} e^{i\mathbf{w}\cdot\varphi}$"),
):
    im = ax.imshow(K, cmap="RdBu_r", vmin=-1, vmax=1, origin="lower",
                   extent=[w_axis[0] - 0.5, w_axis[-1] + 0.5,
                           w_axis[0] - 0.5, w_axis[-1] + 0.5])
    for i, wx in enumerate(w_axis):
        for j, wy in enumerate(w_axis):
            color = "white" if abs(K[j, i]) > 0.6 else "black"
            ax.text(wx, wy, f"{K[j, i]:+.2f}", ha="center", va="center",
                    fontsize=10, color=color)
    ax.set_xlabel(r"$w_x$")
    ax.set_ylabel(r"$w_y$  (with $w_z=0$)")
    ax.set_title(title, fontsize=10)
    ax.set_xticks(w_axis)
    ax.set_yticks(w_axis)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="kernel value")

fig.suptitle(r"Projector kernel for two averaging schemes (slice $w_z=0$)",
             fontsize=12, y=1.02)
fig.tight_layout()
fig.savefig(FIGS / "fig_projector_kernels.pdf", bbox_inches="tight")
fig.savefig(FIGS / "fig_projector_kernels.png", dpi=160, bbox_inches="tight")
print(f"Wrote {FIGS}/fig_projector_kernels.{{pdf,png}}")
