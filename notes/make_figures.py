#!/usr/bin/env python3
"""Generate pedagogical figures for finite_size_loop_projection_notes.tex.

All physics geometry (positions, ice states, perturbation theory, projector,
specific heat) is taken from the standalone recompute module so that the
figures are guaranteed consistent with the text and its JSON output.  The
figures are intended to be understandable by a reader with no background in
quantum spin ice, so each panel is heavily annotated.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Circle, Rectangle, FancyBboxPatch
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d.art3d import Line3DCollection

import recompute_finite_size_artifact as R

HERE = Path(__file__).resolve().parent
FIGS = HERE / "figs"
FIGS.mkdir(exist_ok=True)

# ---- house style -----------------------------------------------------------
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "mathtext.fontset": "cm",
    "font.family": "serif",
    "axes.linewidth": 0.9,
    "savefig.bbox": "tight",
    "savefig.dpi": 200,
})

C_UP = "#c0392b"      # raised spin / "in"
C_DN = "#2471a3"      # lowered spin / "out"
C_ICE = "#27ae60"     # ice / physical
C_ART = "#e67e22"     # artifact
C_GREY = "#7f8c8d"
C_HEX = "#8e44ad"


class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        return np.min(zs)


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(FIGS / f"{name}.{ext}")
    plt.close(fig)
    print("wrote", name)


# ===========================================================================
# Fig 1: the ice rule and its defects
# ===========================================================================
def fig_ice_rule():
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.6), subplot_kw={"projection": "3d"})

    # tetra vertices
    tet = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], float) * 0.5
    center = tet.mean(0)
    edges = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]

    def draw(ax, spins, title, cost):
        for a, b in edges:
            ax.plot(*zip(tet[a], tet[b]), color=C_GREY, lw=1.2, alpha=0.6)
        for p, s in zip(tet, spins):
            d = (p - center)
            d = d / np.linalg.norm(d)
            col = C_UP if s > 0 else C_DN
            # arrow points IN (toward center) if "in", OUT if "out"
            tip = p + (-0.55 * d if s > 0 else 0.55 * d)
            base = p - (-0.55 * d if s > 0 else 0.55 * d) * 0.0 + p * 0
            ax.add_artist(Arrow3D([p[0], tip[0]], [p[1], tip[1]], [p[2], tip[2]],
                                  mutation_scale=13, lw=2.2, arrowstyle="-|>", color=col))
            ax.scatter(*p, color=col, s=45, depthshade=False, zorder=5)
        ax.set_title(title, pad=2)
        ax.text2D(0.5, -0.04, cost, transform=ax.transAxes, ha="center",
                  fontsize=10.5, color="black")
        ax.set_box_aspect((1, 1, 1))
        ax.set_axis_off()
        ax.view_init(elev=16, azim=32)
        lim = 0.95
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)

    draw(axes[0], [1, 1, -1, -1], "two-in / two-out", r"$Q_t=0$   (ice, no cost)")
    draw(axes[1], [1, 1, 1, -1], "three-in / one-out", r"$|Q_t|=1$   costs $J_{zz}/2$")
    draw(axes[2], [-1, -1, -1, 1], "one-in / three-out", r"$|Q_t|=1$   costs $J_{zz}/2$")

    fig.suptitle("The ice rule: every tetrahedron wants two spins in and two out",
                 y=1.0, fontsize=12.5)
    # legend
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], color=C_UP, lw=2.4, label="spin pointing in"),
               Line2D([0], [0], color=C_DN, lw=2.4, label="spin pointing out")]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, -0.02))
    fig.subplots_adjust(bottom=0.12)
    save(fig, "fig_ice_rule")


# ===========================================================================
# Fig 2: why the shortest ice loop is a hexagon (diamond vs a forbidden square)
# ===========================================================================
def fig_diamond_hexagon():
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))

    # ----- left: a hexagon exists on the (bipartite, triangle-free) diamond
    ax = axes[0]
    ang = np.deg2rad(np.arange(6) * 60 + 90)
    hexpts = np.column_stack([np.cos(ang), np.sin(ang)])
    for i in range(6):
        ax.plot(*zip(hexpts[i], hexpts[(i + 1) % 6]), color=C_ICE, lw=2.6, zorder=1)
    for i, p in enumerate(hexpts):
        col = "#2c3e50" if i % 2 == 0 else "white"
        ax.scatter(*p, s=170, color=col, edgecolor="#2c3e50", lw=1.6, zorder=3)
    # midpoints = pyrochlore sites
    for i in range(6):
        m = 0.5 * (hexpts[i] + hexpts[(i + 1) % 6])
        ax.scatter(*m, s=90, marker="s", color=C_HEX, zorder=4)
    ax.text(0, 0, "flippable\nhexagon", ha="center", va="center", fontsize=11)
    ax.set_title("Allowed: 6-site ring", fontsize=12)
    ax.text(0.5, -0.16,
            r"diamond lattice is bipartite $\Rightarrow$ shortest loop has 6 links",
            transform=ax.transAxes, ha="center", fontsize=10)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1.6, 1.5)

    # ----- right: a 4-site (square) loop would need a triangle-free 4-cycle -> forbidden
    ax = axes[1]
    sq = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], float) * 0.85
    for i in range(4):
        ax.plot(*zip(sq[i], sq[(i + 1) % 4]), color=C_ART, lw=2.6, ls="--")
    cols = ["#2c3e50", "white", "#2c3e50", "white"]
    for p, c in zip(sq, cols):
        ax.scatter(*p, s=170, color=c, edgecolor="#2c3e50", lw=1.6, zorder=3)
    ax.text(0, 0.15, "no such\n4-site ring", ha="center", va="center",
            fontsize=11, color=C_ART)
    ax.text(0, -0.35, r"$\times$", ha="center", va="center", fontsize=34, color=C_ART)
    ax.set_title("Forbidden in the bulk: 4-site ring", fontsize=12)
    ax.text(0.5, -0.16,
            "a 4-cycle needs two same-colour sites adjacent\n(a diamond 4-loop) — it does not exist",
            transform=ax.transAxes, ha="center", fontsize=10)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1.6, 1.5)

    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#2c3e50",
               markeredgecolor="#2c3e50", markersize=11, label="tetrahedron (diamond vertex)"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor=C_HEX,
               markersize=10, label="spin (pyrochlore site = diamond link)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, -0.03))
    fig.suptitle("Why the smallest physical ring exchange has six spins", y=1.0, fontsize=12.5)
    fig.subplots_adjust(bottom=0.16)
    save(fig, "fig_diamond_hexagon")


# ===========================================================================
# Fig 3: the hexagon ring exchange (the physical photon move)
# ===========================================================================
def fig_ring_exchange():
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8))
    ang = np.deg2rad(np.arange(6) * 60 + 90)
    hexpts = np.column_stack([np.cos(ang), np.sin(ang)])

    def draw(ax, spins, title, sub):
        for i in range(6):
            ax.plot(*zip(hexpts[i], hexpts[(i + 1) % 6]), color=C_GREY, lw=1.6, zorder=1)
        for p, s in zip(hexpts, spins):
            col = C_UP if s > 0 else C_DN
            d = p / np.linalg.norm(p)
            perp = np.array([-d[1], d[0]])
            # small vertical arrow to indicate up/down
            a0 = p - 0.16 * np.array([0, 1]) * s
            a1 = p + 0.16 * np.array([0, 1]) * s
            ax.annotate("", xy=a1, xytext=a0,
                        arrowprops=dict(arrowstyle="-|>", color=col, lw=2.4))
            ax.scatter(*p, s=60, color=col, zorder=4)
        ax.set_title(title, fontsize=11.5)
        ax.text(0, 0, sub, ha="center", va="center", fontsize=11)
        ax.set_aspect("equal"); ax.axis("off")
        ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1.5, 1.5)

    up = [1, -1, 1, -1, 1, -1]
    dn = [-1, 1, -1, 1, -1, 1]
    draw(axes[0], up, "start (ice)", r"$|\!\uparrow\downarrow\uparrow\downarrow\uparrow\downarrow\rangle$")
    draw(axes[1], up, "flip 3 alternating pairs", "costs\n" + r"$\sim J_{zz}$ virtually")
    # middle arrow overlay
    axes[1].annotate("", xy=(1.35, 0), xytext=(-1.35, 0),
                     arrowprops=dict(arrowstyle="-|>", color=C_HEX, lw=2.2),
                     annotation_clip=False)
    draw(axes[2], dn, "end (ice, flipped)", r"$|\!\downarrow\uparrow\downarrow\uparrow\downarrow\uparrow\rangle$")

    fig.suptitle(r"The physical ring exchange: amplitude $g_{\rm hex}=12|J_\pm|^3/J_{zz}^2$"
                 r"  (third order in $J_\pm$)", y=1.02, fontsize=12.5)
    save(fig, "fig_ring_exchange")


# ===========================================================================
# Fig 4: the torus and winding (a plain cartoon)
# ===========================================================================
def fig_torus_winding():
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))

    for ax in axes:
        ax.add_patch(Rectangle((0, 0), 4, 4, fill=False, ec="#2c3e50", lw=1.8))
        ax.set_xlim(-0.6, 4.6); ax.set_ylim(-0.6, 4.6)
        ax.set_aspect("equal"); ax.axis("off")
        # identify opposite edges
        for (x, y, s) in [(-0.35, 2, r"$\equiv$"), (4.35, 2, r"$\equiv$")]:
            ax.text(x, y, s, ha="center", va="center", fontsize=13, color=C_GREY)
        ax.annotate("", xy=(4.0, 4.35), xytext=(0.0, 4.35),
                    arrowprops=dict(arrowstyle="<->", color=C_GREY, lw=1.2))
        ax.text(2, 4.55, "glued", ha="center", fontsize=9, color=C_GREY)

    # left: contractible loop (closes inside the box)
    ax = axes[0]
    th = np.linspace(0, 2 * np.pi, 200)
    ax.plot(2 + 0.9 * np.cos(th), 2 + 0.9 * np.sin(th), color=C_ICE, lw=2.8)
    ax.scatter([2], [2], color=C_ICE, s=30)
    ax.set_title("Contractible loop", fontsize=12, color=C_ICE)
    ax.text(2, 0.15, r"winding $\mathbf{w}=0$" + "\ncloses in real space\n= physical hexagon",
            ha="center", fontsize=10)

    # right: winding loop (leaves right edge, re-enters left)
    ax = axes[1]
    ax.plot([1.2, 3.9], [1.5, 2.2], color=C_ART, lw=2.8)
    ax.plot([0.1, 1.2], [2.4, 1.5], color=C_ART, lw=2.8)
    ax.plot([3.9, 4.0], [2.2, 2.25], color=C_ART, lw=2.8)
    # dashed continuation across the identified boundary
    ax.annotate("", xy=(0.1, 2.35), xytext=(4.0, 2.28),
                arrowprops=dict(arrowstyle="-|>", color=C_ART, lw=2.0, ls=(0, (4, 3))),
                annotation_clip=False)
    ax.scatter([1.2], [1.5], color=C_ART, s=30)
    ax.set_title("Winding loop", fontsize=12, color=C_ART)
    ax.text(2, 0.15, r"winding $\mathbf{w}\neq 0$" + "\ncloses only through the wall\n= finite-size artifact",
            ha="center", fontsize=10)

    fig.suptitle("A periodic cluster is a torus: some loops close only by wrapping around it",
                 y=1.0, fontsize=12.5)
    fig.subplots_adjust(bottom=0.14)
    save(fig, "fig_torus_winding")


# ===========================================================================
# Fig 5: the REAL 3D loops on the 16-site cubic cluster + the transported dipole
# ===========================================================================
def _unwrap(cl, path):
    coords = [cl.positions[path[0]].copy()]
    cur = coords[0].copy()
    seq = list(path) + [path[0]]
    for a, b in zip(seq, seq[1:]):
        for v, n in cl.adj[a]:
            if v == b:
                cur = cur + (cl.positions[b] - cl.positions[a] - np.array(n) @ cl.Lvecs)
                coords.append(cur.copy())
                break
    return np.array(coords)


def fig_real_loops():
    cl = R.build_cluster("cubic", (1, 1, 1))
    # winding 4-loop (single-axis winding) and a contractible hexagon
    loop4 = next(p for p, w in cl.loops4 if tuple(sorted(abs(x) for x in w)) == (0, 0, 1))
    w4 = next(w for p, w in cl.loops4 if p == loop4)
    hexp = next(p for p, w in cl.hexes if tuple(w) == (0, 0, 0))

    fig = plt.figure(figsize=(11, 4.8))

    def draw_loop(ax, path, wind, title, artifact):
        pts = _unwrap(cl, path)  # closed, last==first (lifted)
        col = C_ART if artifact else C_ICE
        # bonds; a bond wraps the torus iff its stored image vector n is nonzero
        for k in range(len(path)):
            a, b = path[k], path[(k + 1) % len(path)]
            nvec = next(n for v, n in cl.adj[a] if v == b)
            # a contractible loop can be lifted to close inside one cell (draw solid);
            # only the artifact loop is forced to cross the torus wall
            wrap = artifact and any(x != 0 for x in nvec)
            seg = pts[k:k + 2]
            ax.plot(*seg.T, color=col, lw=2.6, ls="--" if wrap else "-",
                    alpha=0.95 if wrap else 1.0)
        # raised / lowered spins alternate around the ring
        for k, s in enumerate(path):
            c = C_UP if k % 2 == 0 else C_DN
            ax.scatter(*pts[k], color=c, s=70, depthshade=False, zorder=6)
        # faint full cluster
        ax.scatter(*cl.positions.T, color=C_GREY, s=8, alpha=0.35)
        ax.set_title(title, fontsize=11.5, color=col)
        ax.set_box_aspect((1, 1, 1)); ax.set_axis_off()
        ax.view_init(elev=18, azim=-60)

    ax1 = fig.add_subplot(121, projection="3d")
    draw_loop(ax1, hexp, (0, 0, 0), "Contractible hexagon (physics)", False)
    ax1.text2D(0.5, -0.02,
               r"raised$-$lowered balance: $\boldsymbol{\delta}=\boldsymbol{\rho}+\mathbf{N}\mathbf{L}=0$",
               transform=ax1.transAxes, ha="center", fontsize=10, color=C_ICE)

    ax2 = fig.add_subplot(122, projection="3d")
    draw_loop(ax2, loop4, w4, "Winding four-loop (artifact)", True)
    ax2.text2D(0.5, -0.02,
               r"offset by half a cell: $\boldsymbol{\delta}=(0,0,-\frac{1}{2})\neq 0$",
               transform=ax2.transAxes, ha="center", fontsize=10, color=C_ART)

    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=C_UP, markersize=9, label="raised spin"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=C_DN, markersize=9, label="lowered spin"),
        Line2D([0], [0], color="k", lw=2, ls="--", label="bond wrapping the torus"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Both loops flip alternating spins and preserve the ice rule — "
                 "only their torus transport differs", y=1.0, fontsize=12.5)
    fig.subplots_adjust(bottom=0.1)
    save(fig, "fig_real_loops")


# ===========================================================================
# Fig 6: the perturbation-theory ladder (energy denominators)
# ===========================================================================
def fig_pt_ladder():
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4), sharey=True)

    def ladder(ax, energies, labels, order, amp, color):
        n = len(energies)
        xs = np.arange(n)
        for x, e, lab in zip(xs, energies, labels):
            ax.hlines(e, x - 0.32, x + 0.32, color=color, lw=3)
            ax.text(x, e + 0.06, lab, ha="center", fontsize=9.5)
        for x in range(n - 1):
            ax.annotate("", xy=(x + 0.68, energies[x + 1]), xytext=(x + 0.32, energies[x]),
                        arrowprops=dict(arrowstyle="-|>", color=C_GREY, lw=1.6))
            # denominator label on rising/falling steps
            mid_e = 0.5 * (energies[x] + energies[x + 1])
            if energies[x + 1] > 0:
                ax.text(x + 0.5, mid_e + 0.12, r"$J_\pm$", ha="center", fontsize=9, color=C_GREY)
        ax.set_title(f"{order}\n{amp}", fontsize=11)
        ax.set_xlim(-0.6, n - 0.4)
        ax.set_xticks([])
        ax.spines[["top", "right", "bottom"]].set_visible(False)

    # 4-loop: ice(0) -> defect(Jzz) -> ice(0), 2 hops
    ladder(axes[0], [0, 1, 0], ["ice", r"2 defects, $+J_{zz}$", "ice"],
           "Four-loop  (2nd order)", r"$g_4=\dfrac{4J_\pm^2}{J_{zz}}$", C_ART)
    # hexagon: ice -> Jzz -> Jzz -> ice, 3 hops
    ladder(axes[1], [0, 1, 1, 0],
           ["ice", r"$+J_{zz}$", r"$+J_{zz}$", "ice"],
           "Hexagon  (3rd order)", r"$g_{\rm hex}=\dfrac{12|J_\pm|^3}{J_{zz}^2}$", C_HEX)
    axes[0].set_ylabel("virtual energy above the ice manifold")
    axes[0].set_ylim(-0.35, 1.75)
    fig.suptitle("Each transverse exchange costs one power of $J_\\pm$ and climbs $J_{zz}$;\n"
                 "one extra rung makes the hexagon a weaker, higher-order process",
                 y=1.14, fontsize=11.5)
    fig.subplots_adjust(top=0.74)
    save(fig, "fig_pt_ladder")


# ===========================================================================
# Fig 7: scale separation g4/ghex = Jzz/(3|Jpm|)
# ===========================================================================
def fig_scale_sep():
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    jpm = np.linspace(0.01, 0.3, 300)
    Jzz = 1.0
    g4 = 4 * jpm ** 2 / Jzz
    ghex = 12 * jpm ** 3 / Jzz ** 2
    ax.plot(jpm, g4, color=C_ART, lw=2.4, label=r"artifact $g_4=4J_\pm^2/J_{zz}$")
    ax.plot(jpm, ghex, color=C_HEX, lw=2.4, label=r"physics $g_{\rm hex}=12|J_\pm|^3/J_{zz}^2$")
    ax.fill_between(jpm, ghex, g4, color=C_ART, alpha=0.10)
    ax.set_yscale("log")
    ax.set_xlabel(r"$|J_\pm|/J_{zz}$")
    ax.set_ylabel(r"energy scale $/\,J_{zz}$")
    ax.set_title(r"The artifact dominates as $J_\pm\to0$:  $g_4/g_{\rm hex}=J_{zz}/(3|J_\pm|)$")
    ax.legend(frameon=False, loc="lower right")
    # annotate the gap at a representative point
    x0 = 0.05
    ax.annotate("", xy=(x0, 4 * x0 ** 2), xytext=(x0, 12 * x0 ** 3),
                arrowprops=dict(arrowstyle="<->", color="k", lw=1.2))
    ax.text(x0 + 0.006, np.sqrt(4 * x0 ** 2 * 12 * x0 ** 3),
            r"$\times\,J_{zz}/3|J_\pm|$" + f"\n= {1/(3*x0):.0f}x at " + r"$|J_\pm|{=}0.05$",
            fontsize=9.5, va="center")
    ax.grid(True, which="both", alpha=0.2)
    save(fig, "fig_scale_sep")


# ===========================================================================
# Fig 8: twist averaging keeps half; zero-transport removes all
# ===========================================================================
def fig_twist_fail():
    data = json.loads((HERE / "recomputed_finite_size_artifact.json").read_text())
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    channels = [
        ("winding 4-loop", "4_loop_wrapping", "H2"),
        ("wrapping hexagon", "hexagon_wrapping", "H3"),
        ("physical hexagon", "hexagon_contractible", "H3"),
    ]
    case = data["cases"][0]  # cubic 16
    cs = case["channel_survival"]
    labels, corner, delta0 = [], [], []
    for name, key, order in channels:
        rec = cs[order][key]
        labels.append(name)
        corner.append(rec["corner_mean"])
        delta0.append(rec["delta0_mean"])
    x = np.arange(len(labels))
    bw = 0.34
    b1 = ax.bar(x - bw / 2, corner, bw, color=C_GREY, label="boundary-twist average")
    b2 = ax.bar(x + bw / 2, delta0, bw, color=C_ICE, label=r"zero-transport ($\boldsymbol{\delta}=0$)")
    ax.bar_label(b1, fmt="%.1f", padding=2, fontsize=9)
    ax.bar_label(b2, fmt="%.1f", padding=2, fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("fraction of coupling kept")
    ax.set_ylim(0, 1.18)
    ax.axhline(1.0, color="k", lw=0.6, ls=":")
    for xi in x[:2]:
        ax.annotate("half survives!", xy=(xi - bw / 2, 0.5), xytext=(xi - bw / 2, 0.78),
                    ha="center", fontsize=8.5, color=C_ART,
                    arrowprops=dict(arrowstyle="-|>", color=C_ART, lw=1.2))
    ax.set_title("Ordinary twist averaging leaves half of every artifact;\n"
                 r"projecting onto $\boldsymbol{\delta}=0$ removes them and keeps the physics")
    ax.legend(frameon=False, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "fig_twist_fail")


# ===========================================================================
# Fig 9: the specific-heat peak is displaced (bare vs clean)
# ===========================================================================
def fig_ct_twopeak():
    cl = R.build_cluster("cubic", (1, 1, 1))
    pt = R.sw_order23(cl, verbose=False)
    T = np.geomspace(2e-4, 0.08, 700)
    jpm = -0.05
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for mode, color, lab in (("all", C_ART, "bare (with winding four-loops)"),
                             ("delta0", C_ICE, r"clean ($\boldsymbol{\delta}=0$ projection)")):
        H = R.assemble(cl, pt, jpm, mode)
        E = np.linalg.eigvalsh(H)
        C = R.specific_heat(E, T)
        ax.plot(T, C, color=color, lw=2.4, label=lab)
        Tpk = R.refined_peak(T, C)
        ax.axvline(Tpk, color=color, lw=1.0, ls=":")
    g4 = 4 * jpm ** 2
    ghex = 12 * abs(jpm) ** 3
    for val, txt, col in ((g4, r"$g_4$", C_ART), (ghex, r"$g_{\rm hex}$", C_HEX)):
        ax.axvline(val, color=col, lw=1.2, ls="--", alpha=0.7)
        ax.text(val, ax.get_ylim()[1] * 0.02, txt, color=col, ha="center", fontsize=11)
    ax.set_xscale("log")
    ax.set_xlabel(r"temperature $T/J_{zz}$")
    ax.set_ylabel(r"specific heat $C(T)$")
    ax.set_title(r"16-site cubic cluster, $J_\pm=-0.05\,J_{zz}$:"
                 "\nthe low-$T$ peak moves from the artifact scale to the photon scale")
    ax.legend(frameon=False, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "fig_ct_twopeak")


if __name__ == "__main__":
    fig_ice_rule()
    fig_diamond_hexagon()
    fig_ring_exchange()
    fig_torus_winding()
    fig_real_loops()
    fig_pt_ladder()
    fig_scale_sep()
    fig_twist_fail()
    fig_ct_twopeak()
    print("all figures written to", FIGS)
