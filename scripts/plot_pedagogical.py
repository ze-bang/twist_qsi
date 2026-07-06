"""
Pedagogical schematic figures for the twist-averaging paper.

Generates three figures, all saved to twist_qsi_demo/paper/figs/:

1. fig_loops_pictogram.{pdf,png}
     3D view of the 16-site (1x1x1) pyrochlore cluster. Highlights:
       (a) one contractible bulk hexagon  w = (0,0,0)
       (b) one wrapping 4-cycle           w != (0,0,0)
     Bonds that cross the cluster boundary are drawn dashed and their
     wrap vectors are annotated.

2. fig_twist_topology.{pdf,png}
     Schematic of the cluster as a flat 3-torus T^3 = S^1 x S^1 x S^1.
     Shows:
       - the three non-contractible generators gamma_x, gamma_y, gamma_z
       - the three Aharonov-Bohm fluxes phi_x, phi_y, phi_z threading them
       - one contractible loop (no flux pickup) and one
         winding-(1,0,0) loop picking up phase exp(i phi_x)

3. fig_phasor_average.{pdf,png}
     Phasor diagram. For each of the 8 twist corners phi in {0,pi}^3,
     plot the unit phasor exp(i w.phi) in the complex plane for two
     winding choices: w=(0,0,0) (all phasors line up at +1, sum = 8)
     and w=(1,0,0) (phasors split 4/+1 vs 4/-1, sum = 0). This is
     character orthogonality made visual.
"""
from __future__ import annotations

from pathlib import Path
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d.proj3d import proj_transform

DEMO_HAM = Path(__file__).resolve().parents[1] / "output" / "demo" / \
    "phi_0.000pi_0.000pi_0.000pi" / "ham"
FIGS = Path(__file__).resolve().parents[1] / "paper" / "figs"

L = np.array([1.0, 1.0, 1.0])  # cubic cluster size


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------
def load_positions() -> np.ndarray:
    pos = np.zeros((16, 3))
    for line in (DEMO_HAM / "positions.dat").read_text().splitlines():
        line = line.strip()
        if (not line) or line.startswith("#"):
            continue
        toks = line.split()
        i = int(toks[0])
        pos[i] = [float(toks[3]), float(toks[4]), float(toks[5])]
    return pos


def load_nn_list() -> list[tuple[int, int]]:
    bonds: set[tuple[int, int]] = set()
    for line in (DEMO_HAM / "pyrochlore_super_1x1x1_pbc_nn_list.dat").read_text().splitlines():
        line = line.strip()
        if (not line) or line.startswith("#"):
            continue
        toks = list(map(int, line.split()))
        i = toks[0]
        for j in toks[2:]:
            a, b = sorted((i, j))
            bonds.add((a, b))
    return sorted(bonds)


def wrap_vector(ri: np.ndarray, rj: np.ndarray) -> np.ndarray:
    return np.round((rj - ri) / L).astype(int)


# ---------------------------------------------------------------------------
# Loop search
# ---------------------------------------------------------------------------
def find_short_cycles(adj: dict[int, list[int]],
                      pos: np.ndarray,
                      length: int) -> list[tuple[list[int], np.ndarray]]:
    """Return all simple cycles of given length, each as
    (site_sequence, total_wrap_vector). Sequences are returned with their
    canonical rotation (smallest start, smaller-second-element direction)
    so duplicates are filtered."""
    cycles: list[tuple[tuple[int, ...], np.ndarray]] = []
    seen: set[tuple[int, ...]] = set()

    def canon(seq: list[int]) -> tuple[int, ...]:
        n = len(seq)
        rotations = [tuple(seq[k:] + seq[:k]) for k in range(n)]
        rotations += [tuple(reversed(r)) for r in rotations]
        return min(rotations)

    def dfs(start: int, path: list[int], wrap: np.ndarray, depth: int):
        last = path[-1]
        for nb in adj[last]:
            if nb == start and depth == length:
                full = canon(path)
                if full not in seen:
                    seen.add(full)
                    closing = wrap_vector(pos[last], pos[start])
                    cycles.append((list(path), wrap + closing))
                continue
            if nb in path:
                continue
            if depth >= length:
                continue
            n_ij = wrap_vector(pos[last], pos[nb])
            dfs(start, path + [nb], wrap + n_ij, depth + 1)

    for s in range(len(pos)):
        dfs(s, [s], np.zeros(3, dtype=int), 1)
    return cycles


def cycle_wrap_segments(seq: list[int], pos: np.ndarray) -> list[np.ndarray]:
    """Per-edge wrap vectors n_ij for cycle (closing edge included)."""
    out = []
    n = len(seq)
    for k in range(n):
        i = seq[k]
        j = seq[(k + 1) % n]
        out.append(wrap_vector(pos[i], pos[j]))
    return out


def hexagon_normal_111(pos: np.ndarray, seq: list[int]) -> np.ndarray | None:
    """If a 6-site loop lies (within tolerance) in a (111)-family plane,
    return the integer normal direction; else None."""
    coords = pos[seq]
    centroid = coords.mean(axis=0)
    centered = coords - centroid
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    n = vh[-1]
    n = n / np.linalg.norm(n)
    flatness = np.max(np.abs(centered @ n))
    if flatness > 1e-3:
        return None
    candidates = np.array(list(product([-1, 0, 1], repeat=3)))
    candidates = candidates[np.linalg.norm(candidates, axis=1) > 0]
    best = None
    best_dot = -1.0
    for c in candidates:
        if abs(int(np.sum(np.abs(c))) - 3) > 0:
            continue
        cn = c / np.linalg.norm(c)
        d = abs(float(n @ cn))
        if d > best_dot:
            best_dot = d
            best = c
    if best_dot < 0.99:
        return None
    return best


# ---------------------------------------------------------------------------
# 3D arrow helper
# ---------------------------------------------------------------------------
class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, _ = proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        return min(zs3d)


# ---------------------------------------------------------------------------
# FIGURE 1: loops on the 16-site cluster
# ---------------------------------------------------------------------------
def fig_loops_pictogram():
    pos = load_positions()
    bonds = load_nn_list()
    sublat = np.zeros(16, dtype=int)
    for line in (DEMO_HAM / "positions.dat").read_text().splitlines():
        line = line.strip()
        if (not line) or line.startswith("#"):
            continue
        toks = line.split()
        sublat[int(toks[0])] = int(toks[2])
    adj: dict[int, list[int]] = {i: [] for i in range(len(pos))}
    for a, b in bonds:
        adj[a].append(b)
        adj[b].append(a)

    cycles4 = find_short_cycles(adj, pos, 4)
    cycles6 = find_short_cycles(adj, pos, 6)

    bulk_hex = None
    for seq, wrap in cycles6:
        if not np.array_equal(wrap, [0, 0, 0]):
            continue
        normal = hexagon_normal_111(pos, seq)
        if normal is None:
            continue
        bulk_hex = (seq, wrap, normal)
        break

    spurious4 = None
    best_score = None
    for seq, wrap in cycles4:
        if np.array_equal(wrap, [0, 0, 0]):
            continue
        per_edge = cycle_wrap_segments(seq, pos)
        n_wrap_bonds = sum(int(np.any(n != 0)) for n in per_edge)
        sublats = set(int(sublat[s]) for s in seq)
        if len(sublats) != 4:
            continue
        wabs = int(np.sum(np.abs(wrap)))
        score = (n_wrap_bonds, wabs, -seq[0])
        if best_score is None or score < best_score:
            best_score = score
            spurious4 = (seq, wrap)

    if bulk_hex is None:
        raise RuntimeError("Could not find a contractible (111) hexagon")
    if spurious4 is None:
        raise RuntimeError("Could not find a wrapping 4-cycle")

    sublat_color = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    def draw_lattice(ax):
        for i, j in bonds:
            n_ij = wrap_vector(pos[i], pos[j])
            if np.all(n_ij == 0):
                ax.plot(*zip(pos[i], pos[j]), color='gray', lw=0.5, alpha=0.45)
            else:
                ri = pos[i]
                rj_unwrapped = pos[j] - n_ij * L
                mid_a = ri + 0.5 * (rj_unwrapped - ri)
                ax.plot([ri[0], mid_a[0]], [ri[1], mid_a[1]], [ri[2], mid_a[2]],
                        color='gray', lw=0.4, alpha=0.30, ls=(0, (1, 1)))
                rj = pos[j]
                ri_unwrapped = pos[i] + n_ij * L
                mid_b = rj + 0.5 * (ri_unwrapped - rj)
                ax.plot([rj[0], mid_b[0]], [rj[1], mid_b[1]], [rj[2], mid_b[2]],
                        color='gray', lw=0.4, alpha=0.30, ls=(0, (1, 1)))
        for i in range(16):
            ax.scatter(*pos[i], color=sublat_color[sublat[i]],
                       s=22, depthshade=False, edgecolor='k',
                       linewidth=0.4, alpha=0.85)
        for x in [0, 1]:
            for y in [0, 1]:
                ax.plot([x, x], [y, y], [0, 1],
                        color='k', lw=0.5, alpha=0.25, ls='--')
            for z in [0, 1]:
                ax.plot([x, x], [0, 1], [z, z],
                        color='k', lw=0.5, alpha=0.25, ls='--')
        for y in [0, 1]:
            for z in [0, 1]:
                ax.plot([0, 1], [y, y], [z, z],
                        color='k', lw=0.5, alpha=0.25, ls='--')

    def loop_segments(seq):
        """Return a list of (p_start, p_end, n_wrap) segments showing the
        loop. Wrapping bonds are drawn as a 'go to wall' + 'come from wall'
        pair so they don't visually fly across the cell."""
        segs = []
        n_total = len(seq)
        for k in range(n_total):
            i = seq[k]
            j = seq[(k + 1) % n_total]
            n_ij = wrap_vector(pos[i], pos[j])
            if np.all(n_ij == 0):
                segs.append((pos[i].copy(), pos[j].copy(), n_ij, False))
            else:
                rj_unwrapped = pos[j] - n_ij * L
                mid_to = pos[i] + 0.55 * (rj_unwrapped - pos[i])
                segs.append((pos[i].copy(), mid_to, n_ij, True))
                ri_unwrapped = pos[i] + n_ij * L
                mid_from = pos[j] + 0.55 * (ri_unwrapped - pos[j])
                segs.append((pos[j].copy(), mid_from, n_ij, True))
        return segs

    fig = plt.figure(figsize=(11.5, 5.4))
    axA = fig.add_subplot(1, 2, 1, projection='3d')
    axB = fig.add_subplot(1, 2, 2, projection='3d')

    draw_lattice(axA)
    seq6, w6, normal6 = bulk_hex
    for k in range(len(seq6)):
        i = seq6[k]
        j = seq6[(k + 1) % len(seq6)]
        axA.plot(*zip(pos[i], pos[j]), color='#0046b8', lw=3.0, alpha=0.95)
    coords6 = pos[seq6]
    axA.scatter(coords6[:, 0], coords6[:, 1], coords6[:, 2],
                color='#0046b8', s=80, edgecolor='k', linewidth=0.6,
                depthshade=False, zorder=10)
    centroid6 = coords6.mean(axis=0)
    n6 = normal6 / np.linalg.norm(normal6)
    n_disp = 0.18 * n6
    arr = Arrow3D([centroid6[0], centroid6[0] + n_disp[0]],
                  [centroid6[1], centroid6[1] + n_disp[1]],
                  [centroid6[2], centroid6[2] + n_disp[2]],
                  mutation_scale=14, lw=1.5, arrowstyle="-|>",
                  color='#0046b8', alpha=0.9)
    axA.add_artist(arr)
    axA.text(centroid6[0] + 1.35 * n_disp[0],
             centroid6[1] + 1.35 * n_disp[1],
             centroid6[2] + 1.35 * n_disp[2],
             f'$\\hat n=({int(normal6[0]):+d},{int(normal6[1]):+d},{int(normal6[2]):+d})/\\sqrt{{3}}$',
             color='#0046b8', fontsize=8.5)
    axA.set_title(r'(a) Contractible hexagon  $\mathbf{w}=\mathbf{0}$' +
                  '\n' +
                  r'$g_{\rm hex}=12|J_\pm|^3/J_{zz}^2$',
                  fontsize=10)

    draw_lattice(axB)
    seq4, w4 = spurious4
    wrap_dirs_seen: set[tuple[int, int, int]] = set()
    for (p_start, p_end, n_ij, is_wrap) in loop_segments(seq4):
        col = '#c0392b'
        ls = '--' if is_wrap else '-'
        axB.plot([p_start[0], p_end[0]],
                 [p_start[1], p_end[1]],
                 [p_start[2], p_end[2]],
                 color=col, lw=3.0, ls=ls, alpha=0.95)
        if is_wrap:
            key = tuple(int(c) for c in n_ij)
            if key not in wrap_dirs_seen:
                wrap_dirs_seen.add(key)
                lbl_pos = p_end + 0.05 * np.array([0.0, 0.0, 1.0])
                axB.text(lbl_pos[0], lbl_pos[1], lbl_pos[2],
                         f'$\\mathbf{{n}}_{{ij}}=({n_ij[0]:+d},{n_ij[1]:+d},{n_ij[2]:+d})$',
                         color='#c0392b', fontsize=8)
    coords4 = pos[seq4]
    axB.scatter(coords4[:, 0], coords4[:, 1], coords4[:, 2],
                color='#c0392b', s=80, edgecolor='k', linewidth=0.6,
                depthshade=False, zorder=10)
    axB.set_title(r'(b) Wrapping 4-cycle  $\mathbf{w}=' +
                  f'({int(w4[0]):+d},{int(w4[1]):+d},{int(w4[2]):+d})$' +
                  '\n' + r'$g_4^{\rm spur}\sim 4|J_\pm|^2/J_{zz}$',
                  fontsize=10)

    for ax in (axA, axB):
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_zlim(-0.05, 1.05)
        ax.set_box_aspect((1, 1, 1))
        ax.set_xticks([0, 0.5, 1])
        ax.set_yticks([0, 0.5, 1])
        ax.set_zticks([0, 0.5, 1])
        ax.set_xlabel('$x$', fontsize=9, labelpad=-6)
        ax.set_ylabel('$y$', fontsize=9, labelpad=-6)
        ax.set_zlabel('$z$', fontsize=9, labelpad=-6)
        ax.tick_params(labelsize=7)
        ax.view_init(elev=22, azim=32)

    fig.suptitle(
        r'Two emergent ring-exchange loops on the $1{\times}1{\times}1$ '
        r'pyrochlore cluster',
        fontsize=11, y=0.99)
    fig.text(0.5, 0.02,
             r'Solid bonds: bulk ($\mathbf{n}_{ij}=\mathbf{0}$).  '
             r'Dashed grey bonds: boundary-crossing.  '
             r'Sublattices $\mu$ = 0/1/2/3 = blue/orange/green/red.',
             ha='center', fontsize=8.5)
    fig.tight_layout(rect=[0, 0.04, 1, 0.95])

    out_pdf = FIGS / "fig_loops_pictogram.pdf"
    out_png = FIGS / "fig_loops_pictogram.png"
    fig.savefig(out_pdf, bbox_inches='tight')
    fig.savefig(out_png, dpi=160, bbox_inches='tight')
    plt.close(fig)
    print(f"  wrote {out_pdf}")
    print(f"  wrote {out_png}")
    print(f"  bulk hex     sites = {seq6}, normal = {tuple(normal6)}")
    print(f"  spurious 4   sites = {seq4}, w = {tuple(w4)}")


# ---------------------------------------------------------------------------
# FIGURE 2: cluster torus + holonomy
# ---------------------------------------------------------------------------
def fig_twist_topology():
    fig = plt.figure(figsize=(11.5, 4.6))
    axA = fig.add_subplot(1, 2, 1, projection='3d')
    axB = fig.add_subplot(1, 2, 2)

    for x in [0, 1]:
        for y in [0, 1]:
            axA.plot([x, x], [y, y], [0, 1], color='k', lw=0.6, alpha=0.55)
        for z in [0, 1]:
            axA.plot([x, x], [0, 1], [z, z], color='k', lw=0.6, alpha=0.55)
    for y in [0, 1]:
        for z in [0, 1]:
            axA.plot([0, 1], [y, y], [z, z], color='k', lw=0.6, alpha=0.55)

    cyc_x = np.linspace(0, 1, 80)
    axA.plot(cyc_x, np.full_like(cyc_x, 0.18), np.full_like(cyc_x, 0.18),
             color='#1f77b4', lw=2.3)
    axA.plot(np.full_like(cyc_x, 0.18), cyc_x, np.full_like(cyc_x, 0.18),
             color='#2ca02c', lw=2.3)
    axA.plot(np.full_like(cyc_x, 0.18), np.full_like(cyc_x, 0.18), cyc_x,
             color='#d62728', lw=2.3)
    axA.text(1.05, 0.18, 0.18, r'$\gamma_x\,(\varphi_x)$',
             color='#1f77b4', fontsize=10)
    axA.text(0.18, 1.05, 0.18, r'$\gamma_y\,(\varphi_y)$',
             color='#2ca02c', fontsize=10)
    axA.text(0.18, 0.18, 1.05, r'$\gamma_z\,(\varphi_z)$',
             color='#d62728', fontsize=10)

    theta = np.linspace(0, 2 * np.pi, 80)
    cx, cy, cz, R = 0.55, 0.55, 0.55, 0.16
    hx = cx + R * np.cos(theta) * np.sqrt(2 / 3)
    hy = cy + R * np.cos(theta) * (-1 / np.sqrt(6)) + R * np.sin(theta) * (1 / np.sqrt(2))
    hz = cz + R * np.cos(theta) * (-1 / np.sqrt(6)) + R * np.sin(theta) * (-1 / np.sqrt(2))
    axA.plot(hx, hy, hz, color='#0046b8', lw=2.0)
    axA.text(cx + 0.02, cy + 0.20, cz + 0.10,
             r'$\mathcal{L}_{\rm hex}\!:\,\mathbf{w}=\mathbf{0}$',
             color='#0046b8', fontsize=9)

    line_x = np.linspace(0.0, 1.0, 80)
    axA.plot(line_x, np.full_like(line_x, 0.78),
             np.full_like(line_x, 0.78), color='#c0392b', lw=2.0)
    axA.text(0.45, 0.85, 0.78,
             r'$\mathcal{L}_{4}\!:\,\mathbf{w}=(1,0,0)$',
             color='#c0392b', fontsize=9)

    axA.text(0.5, -0.18, 0.0,
             r'$T^3 = \mathbb{R}^3 / (L_x\hat e_x \oplus L_y\hat e_y \oplus L_z\hat e_z)$',
             ha='center', fontsize=9)
    axA.set_title(r'(a) Cluster as a flat 3-torus '
                  r'$T^3$ with $H_1=\mathbb{Z}^3$', fontsize=10)
    axA.set_xlim(-0.05, 1.15)
    axA.set_ylim(-0.05, 1.15)
    axA.set_zlim(-0.05, 1.15)
    axA.set_box_aspect((1, 1, 1))
    axA.set_xlabel('$x$', fontsize=9, labelpad=-6)
    axA.set_ylabel('$y$', fontsize=9, labelpad=-6)
    axA.set_zlabel('$z$', fontsize=9, labelpad=-6)
    axA.tick_params(labelsize=7)
    axA.view_init(elev=18, azim=28)

    axB.set_xlim(-0.1, 1.6)
    axB.set_ylim(-0.1, 1.1)
    axB.set_aspect('equal')
    axB.add_patch(plt.Rectangle((0, 0), 1, 1, fill=False,
                                edgecolor='k', lw=1.0))
    axB.text(0.5, -0.07, r'$x \to x + L_x$', ha='center', fontsize=9)
    axB.text(-0.06, 0.5, r'$y \to y + L_y$', va='center', rotation=90, fontsize=9)

    flux_circ = plt.Circle((1.25, 0.5), 0.16, fill=False, edgecolor='#1f77b4', lw=1.4)
    axB.add_patch(flux_circ)
    axB.annotate('', xy=(1.25 + 0.16 * np.cos(np.pi / 3),
                         0.5 + 0.16 * np.sin(np.pi / 3)),
                 xytext=(1.25, 0.5),
                 arrowprops=dict(arrowstyle='->', color='#1f77b4', lw=1.4))
    axB.text(1.25, 0.5, r'$\Phi=\varphi_x$', ha='center', va='center',
             color='#1f77b4', fontsize=9.5)

    theta2 = np.linspace(0, 2 * np.pi, 60)
    rx, ry = 0.12, 0.10
    cxh, cyh = 0.30, 0.32
    axB.plot(cxh + rx * np.cos(theta2), cyh + ry * np.sin(theta2),
             color='#0046b8', lw=2.0)
    axB.annotate('', xy=(cxh + rx, cyh + 0.005),
                 xytext=(cxh + rx, cyh - 0.005),
                 arrowprops=dict(arrowstyle='->', color='#0046b8', lw=1.6))
    axB.text(cxh, cyh - 0.18, r'$W_{\mathcal{L}_{\rm hex}}=1$',
             ha='center', fontsize=9.5, color='#0046b8')
    axB.text(cxh, cyh - 0.25, r'$\mathbf{w}=\mathbf{0}$',
             ha='center', fontsize=8.5, color='#0046b8')

    seg_y = 0.72
    axB.annotate('', xy=(0.95, seg_y), xytext=(0.05, seg_y),
                 arrowprops=dict(arrowstyle='->', color='#c0392b', lw=2.0))
    axB.plot([0.95, 1.0], [seg_y, seg_y], color='#c0392b',
             lw=2.0, ls='--')
    axB.plot([0.0, 0.05], [seg_y, seg_y], color='#c0392b',
             lw=2.0, ls='--')
    axB.text(0.5, seg_y + 0.06, r'$W_{\mathcal{L}_4}=e^{-i\varphi_x}$',
             ha='center', color='#c0392b', fontsize=9.5)
    axB.text(0.5, seg_y + 0.13, r'$\mathbf{w}=(1,0,0)$',
             ha='center', color='#c0392b', fontsize=8.5)

    axB.text(0.5, 1.07,
             r'Wilson loop: $W_{\mathcal{L}}(\boldsymbol{\varphi})=e^{-i\,\mathbf{w}_\mathcal{L}\cdot\boldsymbol{\varphi}}$',
             ha='center', fontsize=10)

    axB.text(0.5, -0.18,
             r'Twist averaging: $\overline{W_\mathcal{L}}=\delta_{\mathbf{w},\mathbf{0}}$',
             ha='center', fontsize=10)

    axB.set_xticks([])
    axB.set_yticks([])
    for s in axB.spines.values():
        s.set_visible(False)
    axB.set_title(r'(b) Twist as Aharonov$-$Bohm flux through $\gamma_x$',
                  fontsize=10)

    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    out_pdf = FIGS / "fig_twist_topology.pdf"
    out_png = FIGS / "fig_twist_topology.png"
    fig.savefig(out_pdf, bbox_inches='tight')
    fig.savefig(out_png, dpi=160, bbox_inches='tight')
    plt.close(fig)
    print(f"  wrote {out_pdf}")
    print(f"  wrote {out_png}")


# ---------------------------------------------------------------------------
# FIGURE 3: phasor cancellation under twist averaging
# ---------------------------------------------------------------------------
def fig_phasor_average():
    pi = np.pi
    corners = np.array(list(product([0.0, pi], repeat=3)))
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.2))

    titles = [
        r'(a) $\mathbf{w}=(0,0,0)$:  $e^{i\,\mathbf{w}\cdot\boldsymbol{\varphi}}=1$',
        r'(b) $\mathbf{w}=(1,0,0)$:  $e^{i\varphi_x}\in\{+1,-1\}$',
        r'(c) $\mathbf{w}=(1,1,0)$:  $e^{i(\varphi_x+\varphi_y)}\in\{\pm 1\}$',
    ]
    ws = [np.array([0, 0, 0]),
          np.array([1, 0, 0]),
          np.array([1, 1, 0])]

    for ax, title, w in zip(axes, titles, ws):
        ax.set_aspect('equal')
        circle = plt.Circle((0, 0), 1.0, fill=False,
                            edgecolor='gray', lw=0.8, ls='--')
        ax.add_patch(circle)
        ax.axhline(0, color='gray', lw=0.4)
        ax.axvline(0, color='gray', lw=0.4)

        phases = np.exp(1j * (corners @ w))
        unique = {}
        for ph in phases:
            key = (round(ph.real, 6), round(ph.imag, 6))
            unique[key] = unique.get(key, 0) + 1

        for (xr, yi), mult in unique.items():
            offsets = np.linspace(-0.04, 0.04, mult) if mult > 1 else [0.0]
            for off in offsets:
                ang = np.arctan2(yi, xr) + 1e-9
                px = xr + off * np.sin(ang)
                py = yi - off * np.cos(ang)
                ax.annotate('', xy=(px, py), xytext=(0, 0),
                            arrowprops=dict(arrowstyle='->',
                                            color='#1f77b4', lw=1.4, alpha=0.7))
            label_x = xr * 1.18 + 0.05
            label_y = yi * 1.18 + 0.05
            ax.text(label_x, label_y, f'$\\times {mult}$',
                    color='#1f77b4', fontsize=10)

        total = phases.sum()
        ax.annotate('', xy=(total.real, total.imag), xytext=(0, 0),
                    arrowprops=dict(arrowstyle='->',
                                    color='#c0392b', lw=2.4))
        ax.scatter([total.real], [total.imag], color='#c0392b',
                   s=40, zorder=5)
        sum_str = (r'$\sum_{\boldsymbol{\varphi}}\,'
                   r'e^{i\,\mathbf{w}\cdot\boldsymbol{\varphi}}'
                   r' = ' + str(int(round(total.real))) + r'$')
        ax.text(0.05, -1.18, sum_str,
                ha='left', fontsize=10.5, color='#c0392b',
                transform=ax.transData)

        ax.set_xlim(-1.6, 1.6)
        ax.set_ylim(-1.6, 1.6)
        ax.set_xticks([-1, 0, 1])
        ax.set_yticks([-1, 0, 1])
        ax.set_title(title, fontsize=10.5)
        ax.set_xlabel(r'$\mathrm{Re}\,e^{i\,\mathbf{w}\cdot\boldsymbol{\varphi}}$',
                      fontsize=9)
        ax.set_ylabel(r'$\mathrm{Im}\,e^{i\,\mathbf{w}\cdot\boldsymbol{\varphi}}$',
                      fontsize=9)
        ax.tick_params(labelsize=8)

    fig.suptitle(
        r'Phasor sum over the 8 corners $\boldsymbol{\varphi}\in\{0,\pi\}^3$:  '
        r'$\frac{1}{8}\sum_{\boldsymbol{\varphi}} e^{i\,\mathbf{w}\cdot\boldsymbol{\varphi}}'
        r'=\delta_{\mathbf{w}\,\mathrm{mod}\,2,\,\mathbf{0}}$',
        fontsize=11, y=1.02)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    out_pdf = FIGS / "fig_phasor_average.pdf"
    out_png = FIGS / "fig_phasor_average.png"
    fig.savefig(out_pdf, bbox_inches='tight')
    fig.savefig(out_png, dpi=160, bbox_inches='tight')
    plt.close(fig)
    print(f"  wrote {out_pdf}")
    print(f"  wrote {out_png}")


def main():
    FIGS.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "text.usetex": False,
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "axes.linewidth": 0.6,
    })
    print("[1/3] fig_loops_pictogram")
    fig_loops_pictogram()
    print("[2/3] fig_twist_topology")
    fig_twist_topology()
    print("[3/3] fig_phasor_average")
    fig_phasor_average()


if __name__ == "__main__":
    main()
