"""Simulated measurements: three intensity panels along Gamma-L-X.

(a) Gaussian (free-photon) prediction for S^zz: continuous in q (the theory
    is analytic at every k), single doubly degenerate line at
    omega_G = sqrt(2UK) sigma(k), intensity ~ omega_G  (BSS form factor).
(b) exact compact theory, same observable: columns at the cluster momenta.
(c) exact P-even (Raman/ultrasound) plaquette channel.

No interpretive annotations: the panels are what a detector would record.
Broadening eta = 0.08K, power-law color scale, per-panel normalization.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, PowerNorm

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "campaign" / "outputs" / "ring_model_dssf"
OUTDIR = ROOT / "paper_multiphoton" / "figs"

INK, INK2 = "#1a1a19", "#55554e"
CMAP = LinearSegmentedColormap.from_list(
    "ice", ["#ffffff", "#dbe7f7", "#7aabe0", "#2a78d6", "#123a6b"])
ETA = 0.08
HALFW = 0.085

POS = {(0.0, 0.0, 0.0): 0.0, (1/3, 1/3, 1/3): 2/3, (0.5, 0.5, 0.5): 1.0,
       (1/3, 1/3, 2/3): 4/3, (1/6, 1/6, 5/6): 5/3, (0.0, 0.0, 1.0): 2.0}
TICKS = [0.0, 2/3, 1.0, 4/3, 5/3, 2.0]
TLAB = [r"$\Gamma$", r"$(\frac{1}{3}\frac{1}{3}\frac{1}{3})$", r"$L$",
        r"$(\frac{1}{3}\frac{1}{3}\frac{2}{3})$",
        r"$(\frac{1}{6}\frac{1}{6}\frac{5}{6})$", r"$X$"]


def main():
    z = np.load(D / "ring48_channels.npz", allow_pickle=True)
    ge = z["group_energies"]
    scale = json.load(open(D / "ring48_gaussian_match.json"))["scale"]
    qred = np.round(z["qreduced"], 6)
    tcoord = np.concatenate([np.linspace(0, 1, 41),
                             1 + np.linspace(0, 1, 41)[1:]])
    sig = np.asarray(z["path_sigma"], dtype=float)

    slots = {}
    for qi in range(len(qred)):
        key = tuple(np.round(np.sort(np.abs(qred[qi])), 4))
        for ref, x in POS.items():
            if np.allclose(key, np.sort(np.abs(np.array(ref))), atol=1e-3):
                slots.setdefault(x, qi)

    wgrid = np.linspace(0.0, 5.3, 560)
    xgrid = np.linspace(-0.06, 2.09, 480)

    # (a) Gaussian: continuous line, weight ~ omega_G
    gauss = np.zeros((len(wgrid), len(xgrid)))
    sig_x = np.interp(xgrid, tcoord, sig)
    for j, (x, s) in enumerate(zip(xgrid, sig_x)):
        if 0 <= x <= 2.0 and s > 1e-9:
            wG = scale * s
            gauss[:, j] = wG * np.exp(-0.5 * ((wgrid - wG) / ETA) ** 2)

    # (b), (c) exact columns
    def column_map(name):
        img = np.zeros((len(wgrid), len(xgrid)))
        for x0, qi in slots.items():
            w = z[name][:, qi]
            if w.sum() < 1e-12:
                continue
            line = np.zeros_like(wgrid)
            for e, wt in zip(ge, w):
                if wt > 1e-10:
                    line += wt * np.exp(-0.5 * ((wgrid - e) / ETA) ** 2)
            img[:, np.abs(xgrid - x0) <= HALFW] = line[:, None]
        return img

    panels = [
        (gauss, r"(a) Gaussian theory: $S^{zz}(\mathbf{q},\omega)$"),
        (column_map("odd"),
         r"(b) exact compact theory: $S^{zz}(\mathbf{q},\omega)$"),
        (column_map("even"),
         r"(c) exact: Raman/ultrasound channel $\Phi(\mathbf{q},\omega)$"),
    ]

    fig, axes = plt.subplots(3, 1, figsize=(3.4, 6.7), sharex=True)
    fig.patch.set_facecolor("white")
    for ax, (img, title) in zip(axes, panels):
        ax.pcolormesh(xgrid, wgrid, img, cmap=CMAP,
                      norm=PowerNorm(gamma=0.4, vmin=0, vmax=img.max()),
                      rasterized=True)
        ax.set_ylabel(r"$\omega/K$", fontsize=9, color=INK)
        ax.set_title(title, loc="left", fontsize=8.2, color=INK)
        ax.tick_params(labelsize=8, colors=INK2, length=3)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.set_ylim(0, 5.3)
    axes[2].set_xticks(TICKS)
    axes[2].set_xticklabels(TLAB, fontsize=8)
    axes[2].set_xlim(-0.06, 2.09)

    fig.tight_layout(h_pad=0.9)
    for ext in ("pdf", "png"):
        fig.savefig(OUTDIR / f"fig_maps.{ext}", dpi=300,
                    bbox_inches="tight")
    print("wrote", OUTDIR / "fig_maps.pdf")


if __name__ == "__main__":
    main()
