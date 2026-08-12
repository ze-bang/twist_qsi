#!/usr/bin/env python3
"""Make the FCC-32 loop hierarchy and explicit depth-convergence figure.

The coupling scan uses the complete depth-two ice-manifold census.  The
right panel uses exactly the same 16 deterministic source columns and all of
their simple-loop partners at depths one through four.  Each member of that
ladder is evaluated at the ground energy of its own truncated Hamiltonian.
The script refuses to draw the convergence panel until all four depths exist.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TOMO = ROOT / "campaign" / "outputs" / "wp1_tomography"
FIGS = ROOT / "paper" / "figs"

INK = "#16181d"
MUTED = "#686d75"
GRID = "#e1e4e8"
COLORS = {8: "#356399", 10: "#ad5a43", 12: "#927126"}
MARKERS = {8: "o", 10: "s", 12: "^"}
PERIMETERS = (8, 10, 12)
SAMPLING_SYSTEMATIC = 0.08
LABELS = {
    8: r"$K_8/K_6$",
    10: r"$K_{10}/K_8$",
    12: r"$K_{12}/K_{10}$",
}


def perturbative_ratio(length: int, x: np.ndarray | float):
    if length == 8:
        return (10.0 / 3.0) * x
    if length == 10:
        return 3.5 * x
    if length == 12:
        return 3.6 * x
    raise ValueError(length)


def successive_ratios(record: dict) -> dict[int, float]:
    relative = {
        length: float(record[f"k{length}_over_k6"])
        for length in PERIMETERS
    }
    return {
        8: relative[8],
        10: relative[10] / relative[8],
        12: relative[12] / relative[10],
    }


def complete_depth_two_scan() -> list[dict]:
    records = []
    for path in sorted(TOMO.glob("tomo_fcc32_L2_-0p*.json")):
        if "_n" in path.stem:
            continue
        record = json.loads(path.read_text())
        if all(f"k{length}_over_k6" in record for length in PERIMETERS):
            records.append(record)
    return sorted(records, key=lambda record: abs(record["jpm"]))


def physical_target(depth: int) -> dict:
    candidates = []
    for path in TOMO.glob(
            f"target_fcc32_d{depth}_-0p300_n16_z*.json"):
        record = json.loads(path.read_text())
        candidates.append((path, record))
    if not candidates:
        raise FileNotFoundError(f"missing matched physical d={depth} result")

    if depth < 4:
        reference_path = TOMO / f"tomo_fcc32_L{depth}_-0p300_n512.json"
        reference = json.loads(reference_path.read_text())["z_star"]
        path, record = min(
            candidates,
            key=lambda item: abs(item[1]["common_anchor"] - reference))
        if abs(record["common_anchor"] - reference) > 2.0e-6:
            raise RuntimeError(f"d={depth} target is not at its ground energy")
    else:
        matched = [item for item in candidates
                   if item[1].get("anchor_mode") == "depth_matched_ground"]
        if not matched:
            raise RuntimeError("d=4 result is not depth matched")
        path, record = matched[-1]
    record["source_path"] = str(path.relative_to(ROOT))
    return record


def target_statistics(record: dict) -> tuple[dict[int, float], dict[int, list[float]]]:
    """Successive ratios and paired-column bootstrap intervals."""
    path = ROOT / record["source_path"]
    with np.load(path.with_suffix(".npz")) as saved:
        sums = saved["sums"]
        counts = saved["counts"]
    amplitude = sums.sum(axis=0) / counts.sum(axis=0)
    ratios = np.array([
        amplitude[1] / amplitude[0],
        amplitude[2] / amplitude[1],
        amplitude[3] / amplitude[2],
    ])
    rng = np.random.default_rng(20260812)
    bootstrap = np.empty((4000, 3), dtype=float)
    for index in range(len(bootstrap)):
        pick = rng.integers(0, len(sums), size=len(sums))
        sampled = sums[pick].sum(axis=0) / counts[pick].sum(axis=0)
        bootstrap[index] = [
            sampled[1] / sampled[0],
            sampled[2] / sampled[1],
            sampled[3] / sampled[2],
        ]
    raw_intervals = np.percentile(bootstrap, [2.5, 97.5], axis=0).T
    # Across d=1--3, where larger-column calculations exist, the greatest
    # mismatch of these successive ratios is 7.6%.  Inflate the plotted
    # intervals to 8% so the d=4 point does not advertise bootstrap precision
    # while hiding the observed finite-column systematic.
    intervals = np.column_stack((
        np.minimum(raw_intervals[:, 0], ratios * (1 - SAMPLING_SYSTEMATIC)),
        np.maximum(raw_intervals[:, 1], ratios * (1 + SAMPLING_SYSTEMATIC)),
    ))
    return dict(zip(PERIMETERS, ratios)), dict(
        zip(PERIMETERS, intervals.tolist()))


def main() -> int:
    scan = complete_depth_two_scan()
    if not scan:
        raise RuntimeError("missing complete depth-two coupling scan")
    convergence = [physical_target(depth) for depth in range(1, 5)]
    convergence_statistics = [
        target_statistics(record) for record in convergence]

    figure, (left, right) = plt.subplots(
        1, 2, figsize=(7.05, 3.25), gridspec_kw={"width_ratios": (1.55, 1.0)})
    figure.subplots_adjust(left=0.085, right=0.985, bottom=0.17, top=0.91,
                           wspace=0.30)
    for axis in (left, right):
        axis.set_yscale("log")
        axis.grid(True, which="both", color=GRID, linewidth=0.55)
        axis.set_axisbelow(True)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    coupling = np.array([abs(record["jpm"]) for record in scan])
    dense = np.linspace(max(0.015, coupling.min()), 0.305, 300)
    for length in PERIMETERS:
        values = np.array(
            [successive_ratios(record)[length] for record in scan])
        left.plot(coupling, values, marker=MARKERS[length],
                  color=COLORS[length], markersize=3.8, linewidth=1.25,
                  label=LABELS[length] + r" ($d=2$)", zorder=3)
        left.plot(dense, perturbative_ratio(length, dense), "--",
                  color=COLORS[length], linewidth=0.9, alpha=0.65, zorder=2)

        value = convergence_statistics[-1][0][length]
        interval = convergence_statistics[-1][1][length]
        left.errorbar(
            [0.30], [value],
            yerr=[[value - interval[0]], [interval[1] - value]],
            marker=MARKERS[length], markersize=6.0, markerfacecolor="white",
            markeredgecolor=COLORS[length], markeredgewidth=1.2,
            color=COLORS[length], linewidth=0.9, capsize=2.0, zorder=5)

    left.set_xlim(0.018, 0.312)
    left.set_ylim(8.0e-3, 1.45)
    left.set_xlabel(r"$|J_\pm|/J_{zz}$")
    left.set_ylabel("successive simple-loop amplitude ratio")
    left.set_title("(a) coupling dependence", loc="left")
    handles = [
        Line2D([], [], color=COLORS[length], marker=MARKERS[length],
               linewidth=1.2, label=LABELS[length])
        for length in PERIMETERS
    ]
    handles.extend([
        Line2D([], [], color=MUTED, linestyle="--", linewidth=1.0,
               label="leading perturbation theory"),
        Line2D([], [], color=MUTED, marker="o", markerfacecolor="white",
               linestyle="none", label=r"matched $d=4$ point"),
    ])
    left.legend(handles=handles, frameon=False, fontsize=6.4, ncol=2,
                loc="lower right")

    depths = np.arange(1, 5)
    for length in PERIMETERS:
        values = np.array(
            [statistics[0][length] for statistics in convergence_statistics])
        intervals = np.array(
            [statistics[1][length] for statistics in convergence_statistics])
        errors = np.vstack((values - intervals[:, 0], intervals[:, 1] - values))
        right.errorbar(
            depths, values, yerr=errors, marker=MARKERS[length],
            color=COLORS[length], markersize=4.4, linewidth=1.25,
            capsize=2.0, zorder=3)
        right.axhline(
            perturbative_ratio(length, 0.30), color=COLORS[length],
            linestyle="--", linewidth=0.8, alpha=0.55, zorder=1)
    right.set_xlim(0.8, 4.2)
    right.set_xticks(depths)
    right.set_ylim(8.0e-3, 1.45)
    right.set_xlabel(r"virtual depth $d$")
    right.set_ylabel(r"successive ratio at $|J_\pm|/J_{zz}=0.30$")
    right.set_title("(b) truncation convergence", loc="left")
    right.text(0.97, 0.97, "dashed: leading order", transform=right.transAxes,
               ha="right", va="top", color=MUTED, fontsize=6.5)

    FIGS.mkdir(parents=True, exist_ok=True)
    destination = FIGS / "fig_loop_ratios.pdf"
    figure.savefig(destination)
    figure.savefig(destination.with_suffix(".png"), dpi=220)
    plt.close(figure)

    print(f"wrote {destination}")
    for depth, record in zip(depths, convergence):
        values = convergence_statistics[depth - 1][0]
        ratios = (
            f"K8/K6={values[8]:.8g}, K10/K8={values[10]:.8g}, "
            f"K12/K10={values[12]:.8g}")
        print(f"d={depth}: z={record['common_anchor']:.10g}; {ratios}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
