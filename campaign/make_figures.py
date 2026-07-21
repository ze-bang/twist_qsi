#!/usr/bin/env python3
"""Generate manuscript figures only from active campaign products."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notes"))
import recompute_finite_size_artifact as geometry  # noqa: E402

OUTPUT = ROOT / "campaign" / "outputs"
FIGURES = ROOT / "paper" / "figs"
FIGURES.mkdir(parents=True, exist_ok=True)

plt.rcParams.update(
    {
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "text.usetex": True,
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "legend.fontsize": 6.8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "axes.linewidth": 0.7,
        "savefig.dpi": 320,
    }
)

BARE = "#d55e00"
CLEAN = "#0072b2"
FCC = "#009e73"
QMC = "#111111"


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
    annotate_moves: bool = False,
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
        ax.plot(*edge.T, color="#c9ced3", lw=0.48, zorder=1)
    ax.scatter(
        projected[:, 0],
        projected[:, 1],
        s=7,
        facecolor="#aeb5bb",
        edgecolor="white",
        linewidth=0.25,
        zorder=2,
    )

    loop = projected[list(path)]
    walk = _transport_walk(cluster, path)
    transported_dipole = walk[-1]
    increments = np.diff(walk, axis=0)
    loop_center = loop.mean(axis=0)
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
            if annotate_moves:
                # Each move contributes 2*Delta p_l; on these clusters that is
                # always a half-integer vector, so print it as (n)/2.
                doubled = np.rint(2.0 * increments[edge_index // 2]).astype(int)
                anchor = 0.5 * (edge[0] + edge[1])
                # Sit beside the arrow, on the side facing away from the loop
                # centre, so parallel moves do not stack on the same spot.
                along = edge[1] - edge[0]
                perpendicular = np.array([-along[1], along[0]])
                norm = np.linalg.norm(perpendicular)
                perpendicular = (
                    perpendicular / norm if norm > 1.0e-9 else np.array([0.0, 1.0])
                )
                if np.dot(perpendicular, anchor - loop_center) < 0.0:
                    perpendicular = -perpendicular
                ax.text(
                    *(anchor + 0.195 * perpendicular),
                    rf"$({doubled[0]},{doubled[1]},{doubled[2]})/2$",
                    color=color,
                    fontsize=5.4,
                    ha="center",
                    va="center",
                    zorder=7,
                    bbox={
                        "boxstyle": "round,pad=0.10",
                        "facecolor": "white",
                        "edgecolor": "none",
                        "alpha": 0.9,
                    },
                )
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
    vector_center = np.array([0.0, -0.59])
    if np.linalg.norm(transported_dipole) > 1.0e-10:
        direction = transported_dipole @ basis.T
        direction = 0.20 * direction / np.linalg.norm(direction)
        ax.annotate(
            "",
            xy=vector_center + 0.5 * direction,
            xytext=vector_center - 0.5 * direction,
            arrowprops={"arrowstyle": "-|>", "color": "#31363b", "lw": 0.9},
            zorder=6,
        )
    else:
        ax.plot(*vector_center, marker="o", ms=1.8, color="#31363b", zorder=6)
    ax.text(
        0.0,
        -0.72,
        rf"$\mathbf{{q}}_\gamma=({dipole_integer[0]},{dipole_integer[1]},{dipole_integer[2]})$",
        color="#31363b",
        fontsize=5.2,
        ha="center",
        va="center",
    )
    ax.set_title(title, loc="left", pad=0.5, fontsize=6.2)
    ax.set_xlim(-0.62, 0.62)
    ax.set_ylim(-0.78, 0.62)
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
        (*diagonal, "#cc79a7", np.array([1.25, -1.45, 1.0]), r"(c) diagonal"),
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
    scale_color = "#737a80"
    ax.axvline(g6, color=scale_color, lw=0.65, ls=":", zorder=1)
    ax.axvline(g4, color=scale_color, lw=0.65, ls=":", zorder=1)
    ax.text(g6, 0.305, r"$g_6$", color=scale_color, ha="center", va="bottom")
    ax.text(g4, 0.305, r"$g_4$", color=scale_color, ha="center", va="bottom")
    ax.set_xscale("log")
    ax.set_xlim(6.0e-4, 2.0)
    ax.set_ylim(0.0, 0.32)
    entropy_axis.set_ylim(0.0, 0.72)
    ax.set_xlabel(r"$T/J_{zz}$")
    ax.set_ylabel(r"$C/N$")
    entropy_axis.set_ylabel(r"$S/N$")
    ax.spines["top"].set_visible(False)
    entropy_axis.spines["top"].set_visible(False)
    ax.grid(axis="y", color="#e5e8ea", lw=0.45, zorder=0)
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
            cmap="magma",
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
        ax.spines[["top", "right"]].set_visible(False)
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
    colorbar.set_label(r"$S^{zz}(\mathbf{q},\omega)$")
    figure.subplots_adjust(left=0.15, right=0.88, top=0.90, bottom=0.20, wspace=0.10)
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
            ax, cluster, path, winding, color, view, title, annotate_moves=True
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
        ((0.03, 0.08), (0.03, 0.90)),
    ):
        ax.set_title("", loc="left")
        ax.text(
            *position,
            title,
            color="white",
            fontsize=7.2,
            ha="left",
            va="bottom" if position[1] < 0.5 else "top",
            transform=ax.transAxes,
            zorder=5,
        )
        ax.set_ylabel("")
        ax.yaxis.tick_right()
        ax.tick_params(axis="y", labelleft=False, labelright=True, pad=2)
    dssf_axes[0].set_yticks((0.02, 0.04, 0.06, 0.08))
    dssf_axes[1].set_yticks((0.00, 0.02, 0.04, 0.06))
    dssf_axes[0].tick_params(labelbottom=False)
    dssf_axes[0].set_xlabel("")
    colorbar = figure.colorbar(image, cax=colorbar_axis, orientation="horizontal")
    colorbar.set_label(r"$S^{zz}(\mathbf{q},\omega)$")
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
    ax.axhline(5.0, color="#555b60", lw=0.8, ls="--", zorder=2)
    ax.text(
        0.98,
        5.13,
        r"$5\%$ convergence gate",
        color="#555b60",
        fontsize=6.8,
        ha="right",
        va="bottom",
        transform=ax.get_yaxis_transform(),
    )
    for bar, value in zip(bars, changes):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            value + 0.18,
            rf"${value:.3g}\%$",
            ha="center",
            va="bottom",
        )
    ax.set_xticks(positions, (r"$M=2\to3$", r"$M=3\to4$"))
    ax.set_ylabel(r"centered operator change $\epsilon$ (\%)")
    ax.set_ylim(0.0, 7.8)
    ax.grid(axis="y", color="#e5e8ea", lw=0.45, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    save(figure, "fig_character_convergence")


def main() -> None:
    geometry_figure()
    dssf_figure()
    report = json.loads((OUTPUT / "validation_report.json").read_text())


    figure, ax = plt.subplots(figsize=(3.35, 2.45))
    labels = ["QMC", "cubic periodic", "cubic winding-free", "FCC winding-free\n(order 3)"]
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
