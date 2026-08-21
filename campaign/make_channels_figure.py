"""Benton-style two-panel figure along Gamma-L-X: neutron (odd) channel vs
Raman (even) channel, exact ED sticks on the continuous Gaussian backdrop.

Data: ring48_channels.npz (validated: curl gate 0.0, momenta projector-gated).
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

BLUE, GREEN, ORANGE = "#2a78d6", "#1baf7a", "#eb6834"
INK, INK2, GRID = "#1a1a19", "#55554e", "#dedcd5"

# path positions of the five inequivalent momenta (reduced coordinates)
POS = {(1/3, 1/3, 1/3): 2/3, (0.5, 0.5, 0.5): 1.0,
       (1/3, 1/3, 2/3): 1.0 + 1/3, (1/6, 1/6, 5/6): 1.0 + 2/3,
       (0.0, 0.0, 1.0): 2.0}          # units: fraction of each segment
LAB = {2/3: r"$(\frac{1}{3}\frac{1}{3}\frac{1}{3})$", 1.0: r"$L$",
       4/3: r"$(\frac{1}{3}\frac{1}{3}\frac{2}{3})$",
       5/3: r"$(\frac{1}{6}\frac{1}{6}\frac{5}{6})$", 2.0: r"$X$"}


def main():
    z = np.load(D / "ring48_channels.npz", allow_pickle=True)
    ge = z["group_energies"]
    scale = json.load(open(D / "ring48_gaussian_match.json"))["scale"]
    qred = np.round(z["qreduced"], 6)
    path, sig = z["path"], z["path_sigma"]
    # path coordinate: 0..1 Gamma-L, 1..2 L-X (uniform parameter)
    tcoord = np.concatenate([np.linspace(0, 1, 41),
                             1 + np.linspace(0, 1, 41)[1:]])

    # map the 12 momenta onto the 5 path positions (star partners collapse)
    slots = {}
    for qi in range(len(qred)):
        key = tuple(np.round(np.sort(np.abs(qred[qi])), 4))
        for ref, x in POS.items():
            if np.allclose(key, np.sort(np.abs(ref)), atol=1e-3):
                slots.setdefault(x, []).append(qi)

    fig, axes = plt.subplots(2, 1, figsize=(3.4, 4.9), sharex=True)
    fig.patch.set_facecolor("white")
    for ax in axes:
        ax.set_facecolor("#fcfcfb")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.tick_params(labelsize=8, colors=INK2, length=3)
        ax.set_ylim(0, 5.3)
        for x in POS.values():
            ax.axvline(x, lw=0.5, color=GRID, zorder=0)

    # ---------- panel (a): odd / neutron ---------------------------------
    ax = axes[0]
    ax.fill_between(tcoord, scale * sig, 5.3, color=GREEN,
                    alpha=0.14, lw=0, zorder=0)
    ax.plot(tcoord, scale * sig, color=ORANGE, lw=1.6, zorder=2)
    for x, qis in slots.items():
        w = z["odd"][:, qis[0]]
        tot = w.sum()
        for gi in np.where(w > 1e-8)[0]:
            ax.scatter([x], [ge[gi]], s=340 * w[gi] / tot, color=BLUE,
                       zorder=3, edgecolors="white", linewidths=0.6)
    ax.annotate("Gaussian $1\\gamma$", (0.06, 0.62), fontsize=7.4,
                color=ORANGE)
    ax.annotate("$3\\gamma$ continuum\n(Gaussian kinematics)", (0.06, 4.35),
                fontsize=7.0, color="#12805a")
    ax.set_ylabel(r"$\omega/K$", fontsize=9, color=INK)
    ax.set_title(r"(a) neutron channel: $S^{zz}$ ($\mathcal{P}$-odd: "
                 r"$1\gamma,3\gamma,\dots$)", loc="left", fontsize=8,
                 color=INK)

    # ---------- panel (b): even / Raman ----------------------------------
    ax = axes[1]
    ax.fill_between(tcoord, scale * sig, scale * z["env2"][:, 1],
                    color=GREEN, alpha=0.14, lw=0, zorder=0)
    ax.plot(tcoord, scale * sig, color=ORANGE, lw=0.9, ls=(0, (2, 2)),
            zorder=1)
    for x, qis in slots.items():
        w = z["even"][:, qis[0]]
        tot = w.sum()
        if tot < 1e-10:
            continue
        for gi in np.where(w / tot > 1e-4)[0]:
            ax.scatter([x], [ge[gi]], s=340 * w[gi] / tot, color=BLUE,
                       zorder=3, edgecolors="white", linewidths=0.6)
    ax.annotate("$2\\gamma$ continuum\n(Gaussian kinematics)", (0.06, 4.35),
                fontsize=7.0, color="#12805a")
    ax.set_ylabel(r"$\omega/K$", fontsize=9, color=INK)
    ax.set_title(r"(b) Raman/ultrasound channel: $\Phi$ "
                 r"($\mathcal{P}$-even: $2\gamma,4\gamma,\dots$)",
                 loc="left", fontsize=8, color=INK)
    ax.set_xticks([0] + sorted(LAB))
    ax.set_xticklabels([r"$\Gamma$"] + [LAB[k] for k in sorted(LAB)],
                       fontsize=8)
    ax.set_xlim(-0.03, 2.06)

    fig.tight_layout(h_pad=1.0)
    for ext in ("pdf", "png"):
        fig.savefig(OUTDIR / f"fig_channels.{ext}", dpi=300,
                    bbox_inches="tight")
    print("wrote", OUTDIR / "fig_channels.pdf")

    # summary numbers for the text
    for x in sorted(slots):
        qi = slots[x][0]
        wo, we = z["odd"][:, qi], z["even"][:, qi]
        bo = ge[np.argmax(wo)] if wo.sum() > 0 else np.nan
        be = ge[np.argmax(we)] if we.sum() > 0 else np.nan
        lowest_even = ge[np.where(we / max(we.sum(), 1e-30) > 1e-3)[0]]
        print(f"x={x:.3f}  odd peak w={bo:.4f}  even peak w={be:.4f}  "
              f"even onset={lowest_even.min() if len(lowest_even) else np.nan:.4f}  "
              f"even total={we.sum():.4f}")


if __name__ == "__main__":
    main()
