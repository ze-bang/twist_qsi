#!/usr/bin/env python3
"""Labelled reference for Fig. 1i: every site and tetrahedron numbered.

Diagnostic only -- writes campaign/outputs/fig1_labelled.pdf/.png and touches
nothing the manuscript uses.  Use it to name a site or tetrahedron exactly
when describing what a panel should show.
"""
from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notes"))
sys.path.insert(0, str(ROOT / "campaign"))
import recompute_finite_size_artifact as geometry  # noqa: E402
import make_figures as F  # noqa: E402

OUT = ROOT / "campaign" / "outputs"

plt.rcParams.update({
    "font.family": "serif", "mathtext.fontset": "cm", "text.usetex": False,
    "font.size": 9, "figure.facecolor": "white", "savefig.dpi": 200,
})


def panel(ax, cluster, path, view, colour, title):
    basis = F._projection_basis(view)
    P = cluster.positions @ basis.T
    centre = P.mean(axis=0)
    scale = max(np.ptp(P[:, 0]), np.ptp(P[:, 1]))
    P = (P - centre) / scale

    def to_plot(p):
        return (np.asarray(p) @ basis.T - centre) / scale

    for (a, b), wrap in zip(cluster.bonds, cluster.bond_wrap):
        if np.any(wrap):
            continue
        ax.plot(*P[[a, b]].T, color="#c8ced6", lw=0.7, zorder=1)

    # tetrahedra: frame + index at the centroid
    for t, sites in enumerate(cluster.tets):
        corners = np.array([
            to_plot(F._nearest_image(cluster, cluster.positions[s],
                                     cluster.positions[sites[0]]))
            for s in sites])
        for i, j in combinations(range(4), 2):
            ax.plot(*np.array([corners[i], corners[j]]).T,
                    color="#9aa1a9", lw=0.5, alpha=0.45, zorder=2)
        seat = corners.mean(axis=0)
        ax.text(*seat, f"t{t}", color="#1a5e3a", fontsize=9, ha="center",
                va="center", zorder=8, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.14", fc="white",
                          ec="#1a5e3a", lw=0.6, alpha=0.95))

    # the loop, with each edge numbered and its wrap printed
    for k, (a, b) in enumerate(zip(path, path[1:] + path[:1])):
        wrap = tuple(int(x) for x in F._edge_wrap(cluster, a, b))
        wraps = any(wrap)
        ax.plot(*P[[a, b]].T, color=colour, lw=2.4 if k % 2 == 0 else 1.4,
                ls="--" if wraps else "-", alpha=0.9, zorder=4)
        mid = 0.5 * (P[a] + P[b])
        tag = f"e{k}" + (f"\n{wrap}" if wraps else "")
        ax.text(*mid, tag, color=colour, fontsize=7.5, ha="center", va="center",
                zorder=9, bbox=dict(boxstyle="round,pad=0.10", fc="white",
                                    ec="none", alpha=0.9))

    # every site, numbered
    ax.scatter(P[:, 0], P[:, 1], s=110, facecolor="white", edgecolor="#16181d",
               linewidth=0.9, zorder=6)
    for s, p in enumerate(P):
        ax.text(*p, str(s), fontsize=8.0, ha="center", va="center",
                color="#16181d", zorder=7)

    ax.set_title(title, fontsize=10)
    ax.set_aspect("equal")
    ax.set_axis_off()


def main() -> None:
    cluster = geometry.build_cluster("cubic", (1, 1, 1))
    views = F.FIG1_VIEWS
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 7.0))
    panel(axes[0], cluster, F.HEXAGON_PATH, views["hexagon"], "#2f5d96",
          f"hexagon  path={F.HEXAGON_PATH}   thick edges = exchanges (e0,e2,e4)")
    panel(axes[1], cluster, F.AXIAL_PATH, views["axial"], "#a95a45",
          f"axial  path={F.AXIAL_PATH}   thick edges = exchanges (e0,e2)")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig1_labelled.{ext}", bbox_inches="tight")
    plt.close(fig)

    print(f"wrote {OUT/'fig1_labelled.pdf'}\n")
    print("site -> its two tetrahedra")
    for s in range(cluster.n_sites):
        print(f"  {s:2d}: {[t for t, x in enumerate(cluster.tets) if s in x]}")
    print("\ntetrahedron -> its four sites")
    for t, x in enumerate(cluster.tets):
        print(f"  t{t}: {x}")
    print("\nexchange bonds and the tetrahedra they charge")
    for name, path in (("hexagon", F.HEXAGON_PATH), ("axial", F.AXIAL_PATH)):
        for k in range(0, len(path), 2):
            a, b = path[k], path[(k + 1) % len(path)]
            ta = [t for t, x in enumerate(cluster.tets) if a in x]
            tb = [t for t, x in enumerate(cluster.tets) if b in x]
            shared = set(ta) & set(tb)
            print(f"  {name:8s} e{k} = ({a},{b}) : shared t{sorted(shared)[0]},"
                  f" charges t{[t for t in ta if t not in shared][0]} (via {a})"
                  f" and t{[t for t in tb if t not in shared][0]} (via {b})")


if __name__ == "__main__":
    main()
