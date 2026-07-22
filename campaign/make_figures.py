#!/usr/bin/env python3
"""Generate manuscript figures only from active campaign products."""

from __future__ import annotations

import json
import sys
from itertools import product
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notes"))
sys.path.insert(0, str(ROOT / "src"))
import recompute_finite_size_artifact as geometry  # noqa: E402
from qsi_campaign.protocol import (  # noqa: E402
    full_hilbert_counterterm_spectrum,
)
from qsi_campaign.thermodynamics import thermal_observables  # noqa: E402

OUTPUT = ROOT / "campaign" / "outputs"
FIGURES = ROOT / "paper" / "figs"
FIGURES.mkdir(parents=True, exist_ok=True)

# --- theme ---------------------------------------------------------------
# Palette validated with the data-viz six checks (adjacent pairlist, the
# correct gate for line charts): worst normal-vision dE 27.3, worst protan
# dE 20.9 for the two data identities.  Guide lines are lighter steps of the
# series hue whose peak they explain, so the annotation carries meaning
# instead of spending a new identity slot.
INK = "#16181d"
INK_MUTED = "#5c6169"
INK_FAINT = "#9aa1a9"
RULE = "#c8ced6"
GRID = "#e6e8ec"

# gnuplot's default pm3d palette (rgbformulae 7,5,15), the convention for
# spectral-weight maps.  Zero maps to black, so in-panel labels invert.
DSSF_CMAP = "gnuplot"
DSSF_LABEL = "#ffffff"

BARE = "#c05621"       # periodic ED
CLEAN = "#2b6cb0"      # winding-free
FCC = "#1a7f4f"        # FCC-32
DIAGONAL = "#9c4f7c"   # third loop class in the geometry panels
QMC = INK

G4 = "#d47b42"         # artifact scale, tinted toward periodic ED
G6 = "#4a8ace"         # ring-exchange scale, tinted toward winding-free

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "mathtext.fontset": "cm",
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{amsmath}",
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "legend.fontsize": 6.8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "text.color": INK,
        "axes.labelcolor": INK,
        "axes.edgecolor": RULE,
        "axes.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": RULE,
        "ytick.color": RULE,
        "xtick.labelcolor": INK,
        "ytick.labelcolor": INK,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.minor.width": 0.5,
        "ytick.minor.width": 0.5,
        "xtick.major.size": 2.6,
        "ytick.major.size": 2.6,
        "xtick.minor.size": 1.4,
        "ytick.minor.size": 1.4,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "legend.frameon": False,
        "legend.labelcolor": INK,
        "legend.handlelength": 1.6,
        "legend.handletextpad": 0.6,
        "legend.columnspacing": 1.2,
        "grid.color": GRID,
        "grid.linewidth": 0.5,
        "lines.solid_capstyle": "round",
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.dpi": 400,
    }
)


def save(figure: plt.Figure, name: str) -> None:
    # PDF only: the manuscripts include the vector version, and nothing
    # references a raster copy.
    figure.savefig(FIGURES / f"{name}.pdf", bbox_inches="tight")
    plt.close(figure)


def _draw_exchange_arrow(ax, start, stop, color: str) -> None:
    start = np.asarray(start, dtype=float)
    stop = np.asarray(stop, dtype=float)
    midpoint = 0.5 * (start + stop)
    direction = stop - start
    tail = midpoint - 0.18 * direction
    head = midpoint + 0.18 * direction
    ax.annotate(
        "",
        xy=head,
        xytext=tail,
        arrowprops={"arrowstyle": "-|>", "color": color, "lw": 1.15},
        zorder=5,
    )


def _nearest_image(cluster, position, anchor):
    """The periodic image of `position` closest to `anchor`."""
    best = None
    for shift in product((-1, 0, 1), repeat=3):
        candidate = position - np.asarray(shift, dtype=float) @ cluster.Lvecs
        distance = np.linalg.norm(candidate - anchor)
        if best is None or distance < best[0]:
            best = (distance, candidate)
    return best[1]


def _tetrahedron_center(cluster, tetrahedron: int, anchor_site: int) -> np.ndarray:
    """Centre of a tetrahedron, lifted to sit beside `anchor_site`."""
    anchor = cluster.positions[anchor_site]
    corners = [
        _nearest_image(cluster, cluster.positions[site], anchor)
        for site in cluster.tets[tetrahedron]
    ]
    return np.mean(corners, axis=0)


def _charge_trajectory(cluster, path: tuple[int, ...]) -> list[list[dict]]:
    """Tetrahedral charges after each exchange of a ring-exchange sequence.

    Every exchange flips both ends of one bond.  The two spins share a
    tetrahedron whose charge is therefore unchanged, so the move acts on the
    two *outer* tetrahedra: it creates a spinon-antispinon pair, or moves an
    existing one, or annihilates the pair.  Returns one entry per exchange,
    each listing the charged tetrahedra that survive it.
    """
    def charges(state: int) -> np.ndarray:
        return np.array(
            [sum(1 if (state >> s) & 1 else -1 for s in tet) for tet in cluster.tets]
        )

    def flippable(state: int) -> bool:
        return all(
            ((state >> path[k]) & 1) != ((state >> path[(k + 1) % len(path)]) & 1)
            for k in range(len(path))
        )

    state = next(int(s) for s in cluster.ice_states if flippable(int(s)))
    if np.any(charges(state)):
        raise RuntimeError("the seed configuration is not an ice state")

    steps: list[list[dict]] = []
    for k in range(0, len(path), 2):
        left, right = path[k], path[(k + 1) % len(path)]
        state ^= (1 << left) | (1 << right)
        occupied = []
        for tet in np.flatnonzero(charges(state)):
            on_bond = [s for s in cluster.tets[tet] if s in (left, right)]
            anchor = on_bond[0] if on_bond else next(
                s for s in cluster.tets[tet] if s in path
            )
            occupied.append(
                {
                    "tetrahedron": int(tet),
                    "charge": int(charges(state)[tet]),
                    "position": _tetrahedron_center(cluster, int(tet), anchor),
                }
            )
        steps.append(occupied)
    if steps[-1]:
        raise RuntimeError(f"sequence {path} does not return to the ice manifold")
    return steps


def _draw_charge_trajectory(ax, cluster, path, basis, center, scale) -> None:
    """Mark the spinon-antispinon pair that the ring exchange must annihilate.

    The first exchange splits a pair onto the two tetrahedra flanking the bond;
    intermediate exchanges walk one charge along the loop; the last one has to
    bring them back together.  We draw the configuration just before that final
    exchange, so the panel poses the question the loop class answers: on the
    contractible hexagon the two charges are neighbours outright, whereas on a
    winding four-loop they are neighbours only across a boundary-crossing bond,
    so the pair meets an image of itself rather than annihilating on the
    covering lattice.
    """
    def to_plot(point):
        return (np.asarray(point) @ basis.T - center) / scale

    steps = _charge_trajectory(cluster, path)
    if len(steps) < 2 or not steps[-2]:
        return
    for entry in steps[-2]:
        point = to_plot(entry["position"])
        ax.plot(
            *point,
            marker="o",
            ms=7.0,
            mfc="white",
            mec=INK,
            mew=0.8,
            zorder=7,
        )
        ax.text(
            *point,
            "$+$" if entry["charge"] > 0 else r"$-$",
            color=INK,
            fontsize=6.4,
            ha="center",
            va="center",
            zorder=8,
        )
    # if a charge walked to get here, show the leg it travelled
    if len(steps) > 2:
        previous = {e["tetrahedron"] for e in steps[-3]}
        current = {e["tetrahedron"] for e in steps[-2]}
        arrived = [e for e in steps[-2] if e["tetrahedron"] not in previous]
        departed = [e for e in steps[-3] if e["tetrahedron"] not in current]
        if arrived and departed:
            start_point = to_plot(departed[0]["position"])
            stop_point = to_plot(arrived[0]["position"])
            offset = stop_point - start_point
            ax.annotate(
                "",
                xy=start_point + 0.72 * offset,
                xytext=start_point + 0.28 * offset,
                arrowprops={
                    "arrowstyle": "-|>",
                    "color": INK_MUTED,
                    "lw": 0.8,
                    "linestyle": (0, (2.2, 1.6)),
                },
                zorder=6,
            )


def _edge_wrap(cluster, left: int, right: int) -> tuple[int, int, int]:
    for neighbor, wrap in cluster.adj[left]:
        if neighbor == right:
            return wrap
    raise ValueError(f"sites {left}, {right} are not adjacent")


def _transport_walk(cluster, path: tuple[int, ...]) -> np.ndarray:
    """Running partial sums 2*sum_l Delta p_l along a loop.

    Row n is the transport accumulated after n moves, so row 0 is the origin
    and the final row is q_gamma.  Only alternating edges carry a move: the
    intervening edges are the return legs of the same exchange.
    """
    total = np.zeros(3, dtype=float)
    points = [total.copy()]
    for edge_index, (left, right) in enumerate(zip(path, path[1:] + path[:1])):
        if edge_index % 2:
            continue
        image = np.asarray(_edge_wrap(cluster, left, right), dtype=float)
        total = total + 2.0 * (
            cluster.positions[right]
            - cluster.positions[left]
            - image @ cluster.Lvecs
        )
        points.append(total.copy())
    return np.array(points)


def _projection_basis(view: np.ndarray) -> np.ndarray:
    normal = np.asarray(view, dtype=float)
    normal /= np.linalg.norm(normal)
    reference = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(reference, normal)) > 0.9:
        reference = np.array([0.0, 1.0, 0.0])
    horizontal = np.cross(reference, normal)
    horizontal /= np.linalg.norm(horizontal)
    vertical = np.cross(normal, horizontal)
    return np.vstack((horizontal, vertical))


def _draw_cluster_loop(
    ax,
    cluster,
    path: tuple[int, ...],
    winding: tuple[int, int, int],
    color: str,
    view: np.ndarray,
    title: str,
    annotate_charges: bool = False,
) -> None:
    basis = _projection_basis(view)
    projected = cluster.positions @ basis.T
    center = projected.mean(axis=0)
    scale = max(np.ptp(projected[:, 0]), np.ptp(projected[:, 1]))
    projected = (projected - center) / scale

    for (left, right), wrap in zip(cluster.bonds, cluster.bond_wrap):
        if np.any(wrap):
            continue
        edge = projected[[left, right]]
        ax.plot(*edge.T, color=RULE, lw=0.48, zorder=1)
    ax.scatter(
        projected[:, 0],
        projected[:, 1],
        s=7,
        facecolor=INK_FAINT,
        edgecolor="white",
        linewidth=0.25,
        zorder=2,
    )

    loop = projected[list(path)]
    transported_dipole = _transport_walk(cluster, path)[-1]
    for edge_index, (left, right) in enumerate(zip(path, path[1:] + path[:1])):
        edge = projected[[left, right]]
        image = np.asarray(_edge_wrap(cluster, left, right), dtype=float)
        wraps = np.any(image)
        ax.plot(
            *edge.T,
            color=color,
            lw=1.45,
            ls="--" if wraps else "-",
            solid_capstyle="round",
            zorder=4,
        )
        if edge_index % 2 == 0:
            _draw_exchange_arrow(ax, edge[0], edge[1], color)
    if annotate_charges:
        _draw_charge_trajectory(ax, cluster, path, basis, center, scale)

    faces = [color if index % 2 == 0 else "white" for index in range(len(path))]
    ax.scatter(
        loop[:, 0],
        loop[:, 1],
        s=18,
        facecolors=faces,
        edgecolors=color,
        linewidths=0.75,
        zorder=5,
    )

    dipole_integer = np.rint(transported_dipole).astype(int)
    if (winding == (0, 0, 0)) != bool(np.all(dipole_integer == 0)):
        raise RuntimeError(f"loop winding and transported dipole disagree for {path}")
    # the transported-dipole arrow is dropped: the charge markers now carry the
    # geometry, and the printed triple carries the exact label
    ax.text(
        0.0,
        -0.72,
        rf"$\mathbf{{q}}_\gamma=({dipole_integer[0]},{dipole_integer[1]},{dipole_integer[2]})$",
        color=INK,
        fontsize=5.2,
        ha="center",
        va="center",
    )
    ax.set_title(title, loc="left", pad=0.5, fontsize=6.2)
    ax.set_xlim(-0.74, 0.74)
    ax.set_ylim(-0.80, 0.66)
    ax.set_aspect("equal")
    ax.set_axis_off()


def _geometry_panel_data():
    cluster = geometry.build_cluster("cubic", (1, 1, 1))
    axial = next(
        (path, winding)
        for path, winding in cluster.loops4
        if path == (1, 3, 6, 4)
    )
    diagonal = next(
        (path, winding)
        for path, winding in cluster.loops4
        if path == (4, 7, 8, 11)
    )
    hexagon = next(
        (path, winding)
        for path, winding in cluster.hexes
        if path == (1, 2, 8, 9, 6, 4)
    )
    hex_points = cluster.positions[list(hexagon[0])]
    _, _, right_vectors = np.linalg.svd(hex_points - hex_points.mean(axis=0))
    panels = (
        (*hexagon, CLEAN, right_vectors[-1], r"(a) hexagon"),
        (*axial, BARE, np.array([1.25, -1.45, 1.0]), r"(b) axial"),
        (*diagonal, DIAGONAL, np.array([1.25, -1.45, 1.0]), r"(c) diagonal"),
    )
    return cluster, panels


def geometry_figure() -> None:
    cluster, panels = _geometry_panel_data()

    # Single-column variant: too narrow for the per-move labels of Fig. 1.
    figure, axes = plt.subplots(1, 3, figsize=(3.35, 1.48))
    for ax, (path, winding, color, view, title) in zip(axes, panels):
        _draw_cluster_loop(ax, cluster, path, winding, color, view, title)
    figure.tight_layout(pad=0.05, w_pad=0.03)
    save(figure, "fig_geometry")


def _draw_thermodynamic_benchmark(ax, exact, exact_report: dict) -> None:
    temperature = exact["temperature"]
    jpm = float(exact_report["jpm"])
    g4 = 4.0 * jpm**2
    g6 = 12.0 * abs(jpm) ** 3
    ax.plot(
        temperature,
        exact["bare_full_heat_capacity_per_site"],
        color=BARE,
        lw=1.25,
    )
    ax.plot(
        temperature,
        exact["M4_full_heat_capacity_per_site"],
        color=CLEAN,
        lw=1.45,
    )
    entropy_axis = ax.twinx()
    entropy_axis.plot(
        temperature,
        exact["bare_full_entropy_per_site"],
        color=BARE,
        lw=1.1,
        ls="--",
    )
    entropy_axis.plot(
        temperature,
        exact["M4_full_entropy_per_site"],
        color=CLEAN,
        lw=1.3,
        ls="--",
    )
    for value, name, tone in ((g6, r"$g_6$", G6), (g4, r"$g_4$", G4)):
        ax.axvline(value, color=tone, lw=0.75, ls=(0, (1.2, 1.8)), zorder=1)
        ax.text(value, 0.305, name, color=tone, ha="center", va="bottom")
    ax.set_xscale("log")
    ax.set_xlim(6.0e-4, 2.0)
    ax.set_ylim(0.0, 0.32)
    entropy_axis.set_ylim(0.0, 0.72)
    ax.set_xlabel(r"$T/J_{zz}$")
    ax.set_ylabel(r"$C/N$")
    entropy_axis.set_ylabel(r"$S/N$")
    ax.spines["top"].set_visible(False)
    entropy_axis.spines["top"].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=0.45, zorder=0)
    method_handles = (
        Line2D([], [], color=BARE, lw=1.5, label="periodic ED"),
        Line2D([], [], color=CLEAN, lw=1.5, label="winding-free"),
    )
    ax.legend(
        handles=method_handles,
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=2,
        columnspacing=1.6,
    )


def _draw_dssf_panels(axes, titles):
    data_path = OUTPUT / "dssf_cubic16_p0p046000.npz"
    if not data_path.exists():
        return None
    data = np.load(data_path)
    frequency = np.asarray(data["frequency"])
    resolved_spectra = (
        np.asarray(data["periodic_spectrum"]),
        np.asarray(data["winding_free_spectrum"]),
    )
    spectra = tuple(
        np.column_stack((spectrum[:, 0], spectrum[:, 1:].mean(axis=1)))
        for spectrum in resolved_spectra
    )
    labels = (r"$\Gamma$", r"$X$")
    frequency_edges = np.concatenate(
        (
            [frequency[0] - 0.5 * (frequency[1] - frequency[0])],
            0.5 * (frequency[:-1] + frequency[1:]),
            [frequency[-1] + 0.5 * (frequency[-1] - frequency[-2])],
        )
    )
    momentum_edges = np.arange(len(labels) + 1, dtype=float) - 0.5
    maximum = max(float(np.max(spectrum)) for spectrum in spectra)
    image = None
    for ax, spectrum, title in zip(axes, spectra, titles):
        image = ax.pcolormesh(
            momentum_edges,
            frequency_edges,
            spectrum,
            cmap=DSSF_CMAP,
            vmin=0.0,
            vmax=maximum,
            shading="flat",
            rasterized=True,
        )
        ax.set_xticks(np.arange(len(labels)), labels)
        ax.set_xlim(-0.5, len(labels) - 0.5)
        ax.set_ylim(0.0, 0.09)
        ax.set_xlabel(r"$\mathbf{q}$")
        ax.set_title(title, loc="left")
        for side in ("top", "right"):
            ax.spines[side].set_visible(True)
            ax.spines[side].set_color(RULE)
    axes[0].set_ylabel(r"$\omega/J_{zz}$")
    return image


def dssf_figure() -> None:
    figure, axes = plt.subplots(1, 2, figsize=(3.35, 2.05), sharey=True)
    image = _draw_dssf_panels(
        axes,
        ("(a) periodic ED", "(b) winding-free"),
    )
    if image is None:
        plt.close(figure)
        return
    colorbar = figure.colorbar(
        image,
        ax=axes,
        orientation="vertical",
        fraction=0.06,
        pad=0.04,
        aspect=22,
    )
    colorbar.set_label(r"$S^{zz}(\mathbf{q},\omega)$", labelpad=6)
    colorbar.outline.set_edgecolor(RULE)
    colorbar.outline.set_linewidth(0.6)
    figure.subplots_adjust(left=0.13, right=0.84, top=0.90, bottom=0.20, wspace=0.12)
    save(figure, "fig_dssf")


def summary_figure(exact, exact_report: dict) -> None:
    figure = plt.figure(figsize=(7.0, 3.80))
    outer = figure.add_gridspec(
        1,
        2,
        width_ratios=(1.45, 0.88),
        left=0.055,
        right=0.98,
        top=0.95,
        bottom=0.15,
        wspace=0.26,
    )

    left_grid = outer[0, 0].subgridspec(
        2,
        1,
        height_ratios=(1.0, 1.26),
        hspace=0.26,
    )
    cluster, panels = _geometry_panel_data()
    geometry_grid = left_grid[0, 0].subgridspec(1, 3, wspace=0.03)
    geometry_axes = [figure.add_subplot(geometry_grid[0, index]) for index in range(3)]
    summary_titles = (
        r"i.a) hexagon",
        r"i.b) axial",
        r"i.c) diagonal",
    )
    for ax, (path, winding, color, view, _), title in zip(
        geometry_axes, panels, summary_titles
    ):
        _draw_cluster_loop(
            ax, cluster, path, winding, color, view, title, annotate_charges=True
        )

    thermodynamics_axis = figure.add_subplot(left_grid[1, 0])
    _draw_thermodynamic_benchmark(thermodynamics_axis, exact, exact_report)
    thermodynamics_axis.text(
        0.015,
        0.94,
        "ii)",
        transform=thermodynamics_axis.transAxes,
        ha="left",
        va="top",
    )

    dssf_grid = outer[0, 1].subgridspec(
        3,
        1,
        height_ratios=(0.10, 1.0, 1.0),
        hspace=0.17,
    )
    colorbar_axis = figure.add_subplot(dssf_grid[0, 0])
    dssf_axes = (
        figure.add_subplot(dssf_grid[1, 0]),
        figure.add_subplot(dssf_grid[2, 0]),
    )
    dssf_axes[1].sharey(dssf_axes[0])
    dssf_axes[1].tick_params(labelleft=False)
    image = _draw_dssf_panels(
        dssf_axes,
        ("iii.a) periodic ED", "iii.b) winding-free"),
    )
    if image is None:
        plt.close(figure)
        return
    for ax, title, position in zip(
        dssf_axes,
        ("iii.a) periodic ED", "iii.b) winding-free"),
        ((0.03, 0.90), (0.03, 0.90)),
    ):
        ax.set_title("", loc="left")
        ax.text(
            *position,
            title,
            color=DSSF_LABEL,
            fontsize=7.2,
            ha="left",
            va="top",
            transform=ax.transAxes,
            zorder=5,
        )
        ax.set_ylabel("")
        ax.yaxis.tick_right()
        ax.tick_params(axis="y", labelleft=False, labelright=True, pad=2)
        ax.yaxis.set_label_position("right")
    dssf_axes[0].set_yticks((0.02, 0.04, 0.06, 0.08))
    dssf_axes[1].set_yticks((0.00, 0.02, 0.04, 0.06))
    dssf_axes[0].tick_params(labelbottom=False)
    dssf_axes[0].set_xlabel("")
    # matplotlib places the label clear of the right-hand tick labels itself,
    # which hand-set figure coordinates cannot do reliably across renderers
    for ax in dssf_axes:
        ax.yaxis.set_label_position("right")
        ax.set_ylabel(r"$\omega/J_{zz}$", labelpad=3)

    colorbar = figure.colorbar(image, cax=colorbar_axis, orientation="horizontal")
    colorbar.set_label(r"$S^{zz}(\mathbf{q},\omega)$")
    colorbar.outline.set_edgecolor(RULE)
    colorbar.outline.set_linewidth(0.6)
    colorbar.ax.xaxis.set_label_position("top")
    colorbar.ax.tick_params(pad=1)
    figure.text(
        1.025,
        0.52,
        r"$\omega/J_{zz}$",
        rotation=270,
        ha="center",
        va="center",
    )
    save(figure, "fig_summary")


def character_convergence_figure(exact_report: dict) -> None:
    convergence = exact_report["convergence"]
    changes = 100.0 * np.array(
        [
            convergence["M2_to_M3_centered_operator_relative_error"],
            convergence["M3_to_M4_centered_operator_relative_error"],
        ]
    )
    positions = np.arange(len(changes))

    figure, ax = plt.subplots(figsize=(3.35, 2.05))
    bars = ax.bar(
        positions,
        changes,
        width=0.56,
        color=(BARE, CLEAN),
        edgecolor="white",
        linewidth=0.5,
        zorder=3,
    )
    ax.set_yscale("log")
    ax.axhline(5.0, color=INK_MUTED, lw=0.8, ls="--", zorder=2)
    ax.text(
        0.98,
        5.6,
        r"$5\%$ convergence gate",
        color=INK_MUTED,
        fontsize=6.8,
        ha="right",
        va="bottom",
        transform=ax.get_yaxis_transform(),
    )
    for bar, value in zip(bars, changes):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            value * 1.25,
            rf"${value:.3g}\%$",
            ha="center",
            va="bottom",
        )
    ax.set_xticks(positions, (r"$M=2\to3$", r"$M=3\to4$"))
    ax.set_ylabel(r"centered operator change $\epsilon$ (\%)")
    ax.set_ylim(5.0e-3, 40.0)
    ax.grid(axis="y", color=GRID, lw=0.45, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    save(figure, "fig_character_convergence")


def low_temperature_heat_capacity_figure() -> None:
    """Heat capacity over the full range, down to the sector-degeneracy floor.

    The decade below the winding-free peak is the diagnostic window: any
    structure there would signal residual inter-sector coupling rather than
    physics, since the sixfold ground manifold is exactly degenerate and a
    degeneracy contributes to the entropy but never to C.
    """
    cache = ROOT / "campaign" / "cache" / "full_ed_cubic16_jpm_0p046.npz"
    exact_path = OUTPUT / "nonperturbative_cubic16_p0p046000.npz"
    if not (cache.exists() and exact_path.exists()):
        print("skipping low-temperature heat capacity: inputs missing")
        return
    full_spectrum = np.load(cache)["E_full"]
    band = np.load(exact_path)["M4_spectrum"]
    spliced = full_hilbert_counterterm_spectrum(full_spectrum, band)

    temperature = np.geomspace(1.0e-5, 2.0, 4000)
    periodic = thermal_observables(full_spectrum, temperature, n_sites=16)
    winding_free = thermal_observables(spliced, temperature, n_sites=16)

    jpm = 0.046
    scales = (
        (12.0 * abs(jpm) ** 3, r"$g_6$", G6),
        (4.0 * jpm**2, r"$g_4$", G4),
    )

    figure, ax = plt.subplots(figsize=(3.35, 2.25))
    ax.plot(
        temperature,
        periodic["heat_capacity_per_site"],
        color=BARE,
        lw=1.35,
        label="periodic ED",
    )
    ax.plot(
        temperature,
        winding_free["heat_capacity_per_site"],
        color=CLEAN,
        lw=1.55,
        label=r"winding-free ($M=4$)",
    )
    for value, name, tone in scales:
        ax.axvline(value, color=tone, lw=0.75, ls=(0, (1.2, 1.8)), zorder=1)
        ax.text(value, 0.205, name, color=tone, fontsize=7, ha="center")
    ax.set_xscale("log")
    ax.set_xlim(1.0e-5, 2.0)
    ax.set_ylim(0.0, 0.33)
    ax.set_xlabel(r"$T/J_{zz}$")
    ax.set_ylabel(r"$C/N$")
    ax.legend(frameon=False, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    save(figure, "fig_low_temperature_heat")


def main() -> None:
    geometry_figure()
    dssf_figure()
    low_temperature_heat_capacity_figure()
    report = json.loads((OUTPUT / "validation_report.json").read_text())


    figure, ax = plt.subplots(figsize=(3.35, 2.45))
    labels = [
        "QMC",
        "cubic\nperiodic",
        "cubic\nwinding-free",
        "FCC\nwinding-free",
    ]
    peaks = [
        report["qmc"]["low_peak"][0],
        report["clusters"]["cubic16"]["bare_low_peak"][0],
        report["clusters"]["cubic16"]["clean_low_peak"][0],
        report["clusters"]["fcc32"]["clean_low_peak_order3_band"][0],
    ]
    colors = [QMC, BARE, CLEAN, FCC]
    ax.bar(np.arange(4), peaks, color=colors, width=0.68)
    ax.axhline(report["qmc"]["low_peak"][0], color=QMC, lw=0.7, ls=":")
    ax.set_yscale("log")
    ax.set_ylabel(r"low-$T$ peak $T/J_{zz}$")
    ax.set_xticks(np.arange(4), labels)
    ax.set_title("Finite-size and projection hierarchy", loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    save(figure, "fig_hierarchy")

    exact_path = OUTPUT / "nonperturbative_cubic16_p0p046000.npz"
    exact_report_path = OUTPUT / "nonperturbative_cubic16_p0p046000.json"
    if exact_path.exists() and exact_report_path.exists():
        exact = np.load(exact_path)
        exact_report = json.loads(exact_report_path.read_text())
        figure, ax = plt.subplots(figsize=(6.4, 2.8))
        _draw_thermodynamic_benchmark(ax, exact, exact_report)
        figure.tight_layout()
        save(figure, "fig_nonperturbative")

        summary_figure(exact, exact_report)
        character_convergence_figure(exact_report)

    print(f"wrote figures to {FIGURES}")


if __name__ == "__main__":
    main()
