"""
Generate fig_cluster_geometry.pdf summarising the geometric audit of pyrochlore
ED clusters relevant to resolving the QSI g_hex = 12|Jpm|^3/Jzz^2 scale.

The figure has three panels:
    (a) per-site density of contractible hexagonal plaquettes nu_c, showing
        that all reasonable clusters preserve the full bulk hexagonal
        coupling (nu_c = 1).
    (b) effective Brillouin-zone resolution k_min ~ 2*pi/L_max as a function
        of cluster shape, with and without twist averaging at N_phi = 2.
        This is the actual bottleneck for resolving the photon Schottky bump.
    (c) the corresponding lowest-photon-mode energy estimate
        omega_min ~ c * k_min compared to the bulk Schottky scale
        T_peak ~ 0.42 * g_hex.

The audit numbers are precomputed by cluster_geometry_audit.py.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        "lines.linewidth": 2.0,
        "lines.markersize": 9,
    }
)

# (label, basis, dims, N_sites, N_hex_contract, dim_Sz0, feasible)
DATA = [
    ("cubic (1,1,1)\n16 sites", "cubic", (1, 1, 1), 16, 16, 12_870, True),
    ("cubic (2,1,1)\n32 sites", "cubic", (2, 1, 1), 32, 32, 601_080_390, True),
    ("FCC (2,2,2)\n32 sites", "fcc", (2, 2, 2), 32, 32, 601_080_390, True),
    ("cubic (2,2,1)\n64 sites", "cubic", (2, 2, 1), 64, 64, int(1.83e18), False),
    ("FCC (4,2,2)\n64 sites", "fcc", (4, 2, 2), 64, 64, int(1.83e18), False),
]

# Photon parameters at Jpm = -0.1, Jzz = 1
JPM = -0.1
JZZ = 1.0
G_HEX = 12.0 * abs(JPM) ** 3 / JZZ**2
T_PEAK_BULK = 0.42 * G_HEX
PHOTON_SPEED = G_HEX  # rough order-of-magnitude estimate, c ~ g_hex * a, a=1

# FCC primitive vectors in cubic units (length = sqrt(2)/2)
A1 = np.array([1.0, 1.0, 0.0]) / 2.0
A2 = np.array([1.0, 0.0, 1.0]) / 2.0
A3 = np.array([0.0, 1.0, 1.0]) / 2.0


def cluster_pbc_vectors(basis, dims):
    L1, L2, L3 = dims
    if basis == "cubic":
        return np.array(
            [
                [L1, 0, 0],
                [0, L2, 0],
                [0, 0, L3],
            ],
            dtype=float,
        )
    elif basis == "fcc":
        return np.array([L1 * A1, L2 * A2, L3 * A3], dtype=float)
    else:
        raise ValueError(basis)


def k_min(basis, dims, n_phi: int = 1):
    """Smallest non-zero |k| on the cluster reciprocal grid, with N_phi twist
    refinement (effective L_eff = N_phi * L)."""
    pbc = cluster_pbc_vectors(basis, dims) * n_phi
    # Reciprocal lattice vectors: 2*pi * (a^-T)
    recip = 2 * np.pi * np.linalg.inv(pbc).T
    # Find smallest non-zero |k| in the cluster BZ
    candidate = float("inf")
    for m1 in range(-2, 3):
        for m2 in range(-2, 3):
            for m3 in range(-2, 3):
                if m1 == 0 and m2 == 0 and m3 == 0:
                    continue
                k = m1 * recip[0] + m2 * recip[1] + m3 * recip[2]
                candidate = min(candidate, np.linalg.norm(k))
    return candidate


OUT_DIR = Path(__file__).resolve().parents[1] / "paper" / "figs"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))

    Ns = [d[3] for d in DATA]
    nu_c = [d[4] / d[3] for d in DATA]
    labels = [d[0] for d in DATA]
    feasible = [d[6] for d in DATA]
    bases = [d[1] for d in DATA]
    dims = [d[2] for d in DATA]

    # Color by basis
    colours = ["#d62728" if b == "cubic" else "#1f77b4" for b in bases]
    edgecolours = ["k" if f else "k" for f in feasible]
    markers = ["o" if f else "X" for f in feasible]

    # --- panel (a): nu_c vs cluster ---
    ax = axes[0]
    xs = np.arange(len(DATA))
    bars = ax.bar(
        xs,
        nu_c,
        color=colours,
        edgecolor="k",
        alpha=0.85,
        width=0.6,
    )
    for x, n, m, f in zip(xs, nu_c, markers, feasible):
        ax.plot(
            x,
            n + 0.04,
            marker=m,
            color="white",
            markersize=14,
            markeredgecolor="k",
            markeredgewidth=1.2,
            linestyle="",
            zorder=10,
        )
    ax.axhline(1.0, ls="--", color="grey", alpha=0.7)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=0, fontsize=8)
    ax.set_ylabel(r"$\nu_c = N_{\rm hex,contract}/N_{\rm sites}$")
    ax.set_ylim(0.0, 1.32)
    ax.set_title(r"(a) Per-site contractible hexagons (bulk $=1$)")
    ax.grid(alpha=0.3, axis="y")
    ax.text(
        0.5,
        0.18,
        "every reasonable cluster preserves\nthe full bulk hexagon density",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.9),
    )

    # --- panel (b): k_min with and without twist averaging ---
    ax = axes[1]
    k_no_twist = [k_min(b, d, n_phi=1) for b, d in zip(bases, dims)]
    k_twist2 = [k_min(b, d, n_phi=2) for b, d in zip(bases, dims)]
    width = 0.4
    bar1 = ax.bar(
        xs - width / 2,
        k_no_twist,
        width,
        color=[c for c in colours],
        alpha=0.55,
        edgecolor="k",
        label=r"$N_\varphi=1$ (no twist)",
    )
    bar2 = ax.bar(
        xs + width / 2,
        k_twist2,
        width,
        color=colours,
        edgecolor="k",
        label=r"$N_\varphi=2$ (8-corner average)",
    )
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=0, fontsize=8)
    ax.set_ylabel(r"min non-zero $|\mathbf{k}|$ on cluster grid (in $a^{-1}$)")
    ax.set_title(r"(b) Brillouin-zone resolution")
    ax.legend(loc="upper right", framealpha=0.95)
    ax.grid(alpha=0.3, axis="y")

    # --- panel (c): omega_min comparison to T_peak_bulk ---
    ax = axes[2]
    omega_min_no = [PHOTON_SPEED * k for k in k_no_twist]
    omega_min_t2 = [PHOTON_SPEED * k for k in k_twist2]
    bar3 = ax.bar(
        xs - width / 2,
        omega_min_no,
        width,
        color=colours,
        alpha=0.55,
        edgecolor="k",
        label=r"$\omega_{\rm min}$, $N_\varphi=1$",
    )
    bar4 = ax.bar(
        xs + width / 2,
        omega_min_t2,
        width,
        color=colours,
        edgecolor="k",
        label=r"$\omega_{\rm min}$, $N_\varphi=2$",
    )
    ax.axhline(
        T_PEAK_BULK,
        ls="--",
        color="green",
        lw=1.5,
        label=r"$T_{\rm peak}^{\rm bulk}\approx 0.42\,g_{\rm hex}$",
    )
    ax.axhline(
        G_HEX,
        ls=":",
        color="black",
        lw=1.5,
        label=r"$g_{\rm hex} = 12|J_\pm|^3/J_{zz}^2$",
    )
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=0, fontsize=8)
    ax.set_ylabel(r"$\omega_{\rm min} \sim c\,k_{\rm min}$ (in $J_{zz}$)")
    ax.set_title(r"(c) Lowest sampled photon mode vs. bulk Schottky")
    ax.set_yscale("log")
    ax.legend(loc="upper right", framealpha=0.95, fontsize=8)
    ax.grid(alpha=0.3, axis="y", which="both")

    # Legend explaining colours
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    handles = [
        Patch(facecolor="#d62728", edgecolor="k", label="cubic conventional cluster"),
        Patch(facecolor="#1f77b4", edgecolor="k", label="FCC primitive cluster"),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="grey",
            markeredgecolor="k",
            markersize=11,
            label="ED feasible",
        ),
        Line2D(
            [0],
            [0],
            marker="X",
            color="w",
            markerfacecolor="grey",
            markeredgecolor="k",
            markersize=11,
            label="Hilbert space too large",
        ),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.04),
        ncol=4,
        frameon=False,
    )

    fig.suptitle(
        r"Cluster geometry vs. photon-scale resolvability "
        r"($J_\pm = -0.1\,J_{zz}$, $g_{\rm hex}=0.012\,J_{zz}$)",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0.04, 1, 0.95])
    out_pdf = OUT_DIR / "fig_cluster_geometry.pdf"
    out_png = OUT_DIR / "fig_cluster_geometry.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    print(f"wrote {out_pdf}")
    print(f"wrote {out_png}")

    # also print the BZ-resolution table to stdout for the writeup
    print()
    print(
        f"{'cluster':<24s} {'L_max[a]':>10s} "
        f"{'kmin(N=1)':>12s} {'kmin(N=2)':>12s} "
        f"{'omin/g_hex(N=2)':>16s}"
    )
    for d, knt, k2 in zip(DATA, k_no_twist, k_twist2):
        L_max = max(np.linalg.norm(v) for v in cluster_pbc_vectors(d[1], d[2]))
        print(
            f"{d[0]:<24s} {L_max:>10.3f} {knt:>12.3f} {k2:>12.3f} "
            f"{PHOTON_SPEED * k2 / G_HEX:>16.2f}"
        )


if __name__ == "__main__":
    main()
