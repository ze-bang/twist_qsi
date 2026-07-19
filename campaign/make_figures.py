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


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notes"))
import recompute_finite_size_artifact as geometry  # noqa: E402
OUTPUT = ROOT / "campaign" / "outputs"
FIGURES = ROOT / "paper" / "figs"
FIGURES.mkdir(parents=True, exist_ok=True)

plt.rcParams.update(
    {
        "font.family": "serif",
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
    for extension in ("pdf", "png"):
        figure.savefig(FIGURES / f"{name}.{extension}", bbox_inches="tight")
    plt.close(figure)


def _edge_wrap(cluster, left: int, right: int) -> tuple[int, int, int]:
    for neighbor, wrap in cluster.adj[left]:
        if neighbor == right:
            return wrap
    raise ValueError(f"sites {left}, {right} are not adjacent")


def geometry_figure() -> None:
    cluster = geometry.build_cluster("cubic", (1, 1, 1))
    four_axis = next(
        (path, wind)
        for path, wind in cluster.loops4
        if tuple(sorted(abs(value) for value in wind)) == (0, 0, 1)
    )
    four_diagonal = next(
        (path, wind)
        for path, wind in cluster.loops4
        if tuple(sorted(abs(value) for value in wind)) == (0, 1, 1)
    )
    hexagon = next((path, wind) for path, wind in cluster.hexes if wind == (0, 0, 0))
    loops = (
        (*four_axis, "winding four-loop", BARE),
        (*four_diagonal, "winding four-loop", "#cc79a7"),
        (*hexagon, "contractible hexagon", CLEAN),
    )

    figure = plt.figure(figsize=(7.0, 2.35))
    for panel, (path, wind, title, color) in enumerate(loops, start=1):
        ax = figure.add_subplot(1, 3, panel, projection="3d")
        corners = np.asarray(
            [(x, y, z) for x in (0.0, 1.0) for y in (0.0, 1.0) for z in (0.0, 1.0)]
        )
        for left, first in enumerate(corners):
            for right, second in enumerate(corners):
                if right > left and np.count_nonzero(first != second) == 1:
                    ax.plot(*np.vstack((first, second)).T, color="#888888", lw=0.55, alpha=0.8)
        for left, right in cluster.bonds:
            xyz = cluster.positions[[left, right]]
            ax.plot(*xyz.T, color="#c8c8c8", lw=0.45, alpha=0.7)
        ax.scatter(*cluster.positions.T, s=8, color="#555555", depthshade=False)
        for left, right in zip(path, path[1:] + path[:1]):
            xyz = cluster.positions[[left, right]]
            wrap = _edge_wrap(cluster, left, right)
            ax.plot(
                *xyz.T,
                color=color,
                lw=2.1,
                ls="--" if wrap != (0, 0, 0) else "-",
            )
        ax.scatter(*cluster.positions[list(path)].T, s=18, color=color, depthshade=False)
        ax.set_proj_type("ortho")
        ax.view_init(elev=24, azim=-48)
        ax.set_box_aspect((1, 1, 1))
        ax.set_xlim(-0.04, 1.04)
        ax.set_ylim(-0.04, 1.04)
        ax.set_zlim(-0.04, 1.04)
        ax.set_axis_off()
        ax.set_title(f"({chr(96 + panel)}) {title}\n$\\mathbf{{w}}={wind}$", pad=-2)
    figure.text(
        0.5,
        0.01,
        "cubic-16 cluster; dashed colored bonds cross a periodic boundary",
        ha="center",
        fontsize=7,
    )
    figure.tight_layout(rect=(0, 0.05, 1, 1), w_pad=0.2)
    save(figure, "fig_geometry")


def main() -> None:
    geometry_figure()
    curves = np.load(OUTPUT / "validation_curves.npz")
    report = json.loads((OUTPUT / "validation_report.json").read_text())
    temperature = curves["temperature"]

    figure, axes = plt.subplots(1, 3, figsize=(7.0, 2.45))
    ax = axes[0]
    ax.plot(
        temperature,
        curves["bare_full_c"],
        color=BARE,
        lw=1.35,
        label="cubic-16 microscopic",
    )
    ax.plot(
        temperature,
        curves["clean_full_c"],
        color=CLEAN,
        lw=1.6,
        label="cubic-16 projected band",
    )
    ax.plot(
        temperature,
        curves["fcc_clean_band_c"],
        color=FCC,
        lw=1.2,
        ls="--",
        label="FCC-32 order-3 band",
    )
    ax.errorbar(
        curves["qmc_temperature"],
        curves["qmc_heat_capacity"],
        yerr=0.0015,
        fmt="o",
        ms=1.8,
        mew=0,
        elinewidth=0.35,
        color=QMC,
        alpha=0.8,
        label="QMC, Huang et al.",
    )
    ax.set_xscale("log")
    ax.set_xlim(4.5e-4, 2.0e-2)
    ax.set_ylim(bottom=0)
    ax.set_xlabel(r"$T/J_{zz}$")
    ax.set_ylabel(r"$C/N$")
    ax.set_title(r"(a) low-$T$ heat capacity", loc="left")
    ax.legend(frameon=False, loc="upper right")

    ax = axes[1]
    ax.plot(temperature, curves["bare_full_s"], color=BARE, lw=1.2, label="microscopic")
    ax.plot(temperature, curves["clean_full_s"], color=CLEAN, lw=1.45, label="band replaced")
    ax.errorbar(
        curves["qmc_temperature"],
        curves["qmc_entropy"],
        yerr=0.002,
        fmt="o",
        ms=1.5,
        mew=0,
        elinewidth=0.3,
        color=QMC,
        alpha=0.75,
        label="QMC",
    )
    ax.set_xscale("log")
    ax.set_xlim(6.0e-4, 2.0)
    ax.set_ylim(0.0, 0.72)
    ax.set_xlabel(r"$T/J_{zz}$")
    ax.set_ylabel(r"$S/N$")
    ax.set_title("(b) entropy", loc="left")
    ax.legend(frameon=False, loc="lower right")

    ax = axes[2]
    ax.plot(temperature, curves["bare_full_c"], color=BARE, lw=1.35, label="microscopic")
    ax.plot(temperature, curves["clean_full_c"], color=CLEAN, lw=1.55, label="band replaced")
    ax.set_xscale("log")
    ax.set_xlim(1.0e-4, 2.0)
    ax.set_ylim(bottom=0)
    ax.set_xlabel(r"$T/J_{zz}$")
    ax.set_ylabel(r"$C/N$")
    shift = 100.0 * report["all_temperature"]["relative_high_peak_temperature_change"]
    height = 100.0 * report["all_temperature"]["relative_high_peak_height_change"]
    ax.text(
        0.04,
        0.20,
        rf"high-$T$ peak: $\Delta T={shift:.2f}\%$, $\Delta C={height:.2f}\%$",
        transform=ax.transAxes,
        va="bottom",
    )
    ax.set_title("(c) all-temperature stability", loc="left")
    ax.legend(frameon=False, loc="upper right")
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle(r"zero-flux verification at $J_\pm/J_{zz}=0.046$", y=1.01, fontsize=8)
    figure.tight_layout(w_pad=1.0)
    save(figure, "fig_validation")

    figure, ax = plt.subplots(figsize=(3.35, 2.45))
    labels = ["QMC", "cubic bare", "cubic clean", "FCC clean\n(order 3)"]
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
        exact_temperature = exact["temperature"]
        grid_colors = {2: "#56b4e9", 3: "#009e73", 4: CLEAN}

        figure, axes = plt.subplots(1, 3, figsize=(7.0, 2.4))
        ax = axes[0]
        ax.plot(
            exact_temperature,
            exact["bare_full_heat_capacity_per_site"],
            color=BARE,
            lw=1.05,
            label="periodic ED",
        )
        for grid in (2, 3, 4):
            ax.plot(
                exact_temperature,
                exact[f"M{grid}_full_heat_capacity_per_site"],
                color=grid_colors[grid],
                lw=1.45 if grid == 4 else 1.0,
                ls="-" if grid == 4 else "--",
                label=rf"exact $M={grid}$",
            )
        ax.plot(
            exact["qmc_temperature"],
            exact["qmc_heat_capacity_per_site"],
            "o",
            color=QMC,
            ms=1.8,
            label="QMC",
        )
        ax.set_xscale("log")
        ax.set_xlim(4.5e-4, 2.0e-2)
        ax.set_ylim(bottom=0)
        ax.set_xlabel(r"$T/J_{zz}$")
        ax.set_ylabel(r"$C/N$")
        ax.set_title(r"(a) nonperturbative $C/N$", loc="left")
        ax.legend(frameon=False, loc="upper right", ncol=2, columnspacing=0.7)

        ax = axes[1]
        for grid in (2, 4):
            ax.plot(
                exact_temperature,
                exact[f"M{grid}_full_entropy_per_site"],
                color=grid_colors[grid],
                lw=1.45 if grid == 4 else 1.0,
                ls="-" if grid == 4 else "--",
                label=rf"exact $M={grid}$",
            )
        ax.plot(
            exact["qmc_temperature"],
            exact["qmc_entropy_per_site"],
            "o",
            color=QMC,
            ms=1.7,
            label="QMC",
        )
        ax.set_xscale("log")
        ax.set_xlim(6.0e-4, 2.0)
        ax.set_ylim(0.0, 0.72)
        ax.set_xlabel(r"$T/J_{zz}$")
        ax.set_ylabel(r"$S/N$")
        ax.set_title("(b) entropy", loc="left")
        ax.legend(frameon=False, loc="lower right")

        ax = axes[2]
        errors = np.array(
            [
                exact_report["convergence"][
                    "M1_to_M2_centered_operator_relative_error"
                ],
                exact_report["convergence"][
                    "M2_to_M3_centered_operator_relative_error"
                ],
                exact_report["convergence"][
                    "M3_to_M4_centered_operator_relative_error"
                ],
            ]
        )
        ax.bar(
            np.arange(3),
            errors,
            color=(grid_colors[2], grid_colors[3], grid_colors[4]),
            width=0.68,
        )
        ax.axhline(0.05, color=QMC, lw=0.8, ls=":", label="5% gate")
        ax.set_yscale("log")
        ax.set_xticks(np.arange(3), (r"$1\to2$", r"$2\to3$", r"$3\to4$"))
        ax.set_ylabel("centered operator error")
        ax.set_title("(c) character convergence", loc="left")
        ax.legend(frameon=False, loc="upper right")
        for axis in axes:
            axis.spines[["top", "right"]].set_visible(False)
        figure.suptitle(
            r"exact twisted-band projection at $J_\pm/J_{zz}=0.046$",
            y=1.01,
            fontsize=8,
        )
        figure.tight_layout(w_pad=1.0)
        save(figure, "fig_nonperturbative")

        figure, axes = plt.subplots(3, 1, figsize=(3.35, 5.5))
        ax = axes[0]
        ax.plot(
            exact_temperature,
            exact["bare_full_heat_capacity_per_site"],
            color=BARE,
            lw=1.0,
            label="periodic ED",
        )
        for grid in (2, 3, 4):
            ax.plot(
                exact_temperature,
                exact[f"M{grid}_full_heat_capacity_per_site"],
                color=grid_colors[grid],
                lw=1.4 if grid == 4 else 0.9,
                ls="-" if grid == 4 else "--",
                label=rf"exact $M={grid}$",
            )
        ax.plot(
            exact["qmc_temperature"],
            exact["qmc_heat_capacity_per_site"],
            "o",
            color=QMC,
            ms=1.7,
            label="QMC",
        )
        ax.set_xscale("log")
        ax.set_xlim(4.5e-4, 2.0e-2)
        ax.set_ylim(bottom=0)
        ax.set_ylabel(r"$C/N$")
        ax.set_title(r"(a) nonperturbative heat capacity", loc="left")
        ax.legend(frameon=False, ncol=2, loc="upper right", columnspacing=0.7)

        ax = axes[1]
        for grid in (2, 4):
            ax.plot(
                exact_temperature,
                exact[f"M{grid}_full_entropy_per_site"],
                color=grid_colors[grid],
                lw=1.4 if grid == 4 else 0.9,
                ls="-" if grid == 4 else "--",
                label=rf"exact $M={grid}$",
            )
        ax.plot(
            exact["qmc_temperature"],
            exact["qmc_entropy_per_site"],
            "o",
            color=QMC,
            ms=1.6,
            label="QMC",
        )
        ax.set_xscale("log")
        ax.set_xlim(6.0e-4, 2.0)
        ax.set_ylim(0.0, 0.72)
        ax.set_ylabel(r"$S/N$")
        ax.set_title("(b) entropy", loc="left")
        ax.legend(frameon=False, loc="lower right")

        ax = axes[2]
        ax.bar(
            np.arange(3),
            errors,
            color=(grid_colors[2], grid_colors[3], grid_colors[4]),
            width=0.68,
        )
        ax.axhline(0.05, color=QMC, lw=0.8, ls=":", label="5% gate")
        ax.set_yscale("log")
        ax.set_xticks(np.arange(3), (r"$1\to2$", r"$2\to3$", r"$3\to4$"))
        ax.set_ylabel("centered operator error")
        ax.set_title("(c) character convergence", loc="left")
        ax.legend(frameon=False, loc="upper right")
        for axis in axes:
            axis.set_xlabel(r"$T/J_{zz}$" if axis is not axes[2] else "grid step")
            axis.spines[["top", "right"]].set_visible(False)
        figure.tight_layout(h_pad=0.8)
        save(figure, "fig_nonperturbative_column")

    print(f"wrote figures to {FIGURES}")


if __name__ == "__main__":
    main()
