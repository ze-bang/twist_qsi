"""Figure: contractible vs non-contractible (wrapping) six-site ice loop,
drawn ON the cubic 1x1x1 periodic cell.

Sites are placed at their home positions inside the unit cube [0,1)^3.
Each loop bond is the real minimum-image NN vector. Bulk bonds (wrap n=0)
are drawn as solid segments inside the cell; boundary-crossing bonds
(n != 0) are drawn as dashed stubs that leave the cell through one face
(ending on the periodic image, shown as an open 'ghost' marker), which on
the torus re-enters through the opposite face. A contractible loop's
boundary crossings cancel and it closes inside the cell; a wrapping loop
walks off to a neighbouring cell and only closes via the PBC
identification.
"""
from __future__ import annotations

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401,E402

import cluster_geometry_audit as cg

L = np.array([1.0, 1.0, 1.0])


def order_cycle(sites, adj):
    sites = list(sites)
    nbr = {u: {v for v, _ in adj[u]} for u in sites}
    path = [sites[0]]
    used = {sites[0]}
    while len(path) < len(sites):
        for v in nbr[path[-1]]:
            if v in sites and v not in used:
                path.append(v)
                used.add(v)
                break
        else:
            break
    return path


def wrap01(r):
    return np.asarray(r) - np.floor(np.asarray(r))


def draw_cube(ax, origin=(0, 0, 0), color="0.7", lw=1.0, ls="-"):
    o = np.asarray(origin, float)
    r = [0, 1]
    for s in r:
        for t in r:
            ax.plot([o[0], o[0]+1], [o[1]+s, o[1]+s], [o[2]+t, o[2]+t], color=color, lw=lw, ls=ls)
            ax.plot([o[0]+s, o[0]+s], [o[1], o[1]+1], [o[2]+t, o[2]+t], color=color, lw=lw, ls=ls)
            ax.plot([o[0]+s, o[0]+s], [o[1]+t, o[1]+t], [o[2], o[2]+1], color=color, lw=lw, ls=ls)


def draw_loop(ax, path, vertices, bond_wrap, color, wrap_vec):
    """Draw the UNWRAPPED lift of the loop anchored at the start's home
    position, with the home cell (and, for a wrapping loop, the neighbour
    cell it lands in) drawn for reference."""
    draw_cube(ax, (0, 0, 0))
    s0 = wrap01(vertices[path[0]])
    pts = [s0]
    flags = []
    for i in range(len(path)):
        u, v = path[i], path[(i + 1) % len(path)]
        n = np.array(bond_wrap[(u, v)], float)
        bond = (np.asarray(vertices[v]) - np.asarray(vertices[u])) - n * L
        pts.append(pts[-1] + bond)
        flags.append(tuple(int(x) for x in n) != (0, 0, 0))
    pts = np.array(pts)

    end_disp = -np.array(wrap_vec, float) * L  # = sum of real bonds = pts[-1]-pts[0]
    if np.linalg.norm(end_disp) > 1e-6:
        draw_cube(ax, np.round(end_disp), color="tab:orange", lw=1.0, ls=":")

    for i in range(len(pts) - 1):
        c = "crimson" if flags[i] else color
        ax.plot(*zip(pts[i], pts[i + 1]), color=c, lw=3.5,
                ls=("--" if flags[i] else "-"), solid_capstyle="round", zorder=4)

    ax.scatter(pts[1:-1, 0], pts[1:-1, 1], pts[1:-1, 2], c=color, s=110,
               edgecolors="k", depthshade=False, zorder=6)
    ax.scatter(*pts[0], c="limegreen", s=300, edgecolors="k",
               depthshade=False, zorder=7, label="start")
    if np.linalg.norm(end_disp) > 1e-6:
        ax.scatter(*pts[-1], facecolors="white", edgecolors="red", marker="X",
                   s=260, lw=2.5, depthshade=False, zorder=7,
                   label="end = image of start")
        ax.plot(*zip(pts[0], pts[-1]), color="gray", lw=1.5, ls=":", zorder=3)
    else:
        ax.scatter(*pts[-1], c="limegreen", s=300, edgecolors="k",
                   depthshade=False, zorder=7)

    lo = pts.min(0).min() - 0.2
    hi = pts.max(0).max() + 0.2
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_zlim(lo, hi)
    ax.set_xlabel("x/a"); ax.set_ylabel("y/a"); ax.set_zlabel("z/a")
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=18, azim=-60)
    ax.legend(loc="upper left", fontsize=8)


def main():
    vertices, edges, _t, bond_wrap, adj = cg.build_graph(1, 1, 1)
    tets = cg._enumerate_4_cliques(adj, len(vertices))
    site_tets = cg.site_to_tetrahedra(tets)
    pbc = [np.array([1.0, 0, 0]), np.array([0, 1.0, 0]), np.array([0, 0, 1.0])]
    hexes = [(s, w) for s, w in cg.enumerate_simple_cycles(adj, 6)
             if cg.is_ice_preserving(s, site_tets, tets)
             and cg.is_111_planar_with_wrap(s, vertices, bond_wrap, pbc)]
    contr = next((s, w) for s, w in hexes if tuple(w) == (0, 0, 0))
    wrap = next((s, w) for s, w in hexes if tuple(w) != (0, 0, 0))

    fig = plt.figure(figsize=(12.5, 6))
    for k, (sites, w, color) in enumerate([
            (contr[0], contr[1], "tab:blue"),
            (wrap[0], wrap[1], "tab:orange")]):
        ax = fig.add_subplot(1, 2, k + 1, projection="3d")
        path = order_cycle(sites, adj)
        draw_loop(ax, path, vertices, bond_wrap, color, w)
        kind = ("Contractible  $w=(0,0,0)$\nlift closes inside one cell"
                if tuple(w) == (0, 0, 0)
                else f"Wrapping  $w={tuple(int(x) for x in w)}$\n"
                     "lift ends in the neighbour cell (dotted box)")
        ax.set_title(kind, fontsize=11)

    fig.suptitle("Six-site ice loops, unwrapped onto the $1\\times1\\times1$ pyrochlore PBC cell "
                 "(green = start; dashed red = boundary-crossing bond)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig("contractible_vs_wrapping_hexagon.png", dpi=150, bbox_inches="tight")
    fig.savefig("../paper/figs/fig_contractible_vs_wrapping.pdf", bbox_inches="tight")
    print("wrote PNG and ../paper/figs/fig_contractible_vs_wrapping.pdf")


if __name__ == "__main__":
    main()
