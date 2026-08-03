#!/usr/bin/env python3
"""Extract visual-comparison curves from Fig. 2(a) of Smith et al. PRL 135, 086702.

The axis calibration is read off the plotted tick marks rather than assumed.
In ``Figure2.png`` of arXiv:2501.08327v5 the panel-(a) frame runs from
``x=110.5`` to ``x=1031.5`` and from ``y=9`` to ``y=700.5``.  The long x ticks
sit at ``x=389.5`` and ``x=726.5``; identifying them with 0.1 K and 1 K puts
the frame edges at 0.0149 K and 8.04 K and places the minor ticks at 0.05, 0.5,
2 and 3 K, which is the axis as drawn.  The y ticks at ``y=147.5, 285.5, 424.5,
561.5`` are 2.0, 1.5, 1.0 and 0.5, so ``C=0`` falls on the frame at ``y=699.5``.

With this calibration the extracted experimental trace peaks at 0.0237 K and
0.0640 K, reproducing the ``T_2 ~ 0.025`` K and ``T_1 ~ 0.065`` K quoted in the
abstract.  An earlier revision of this file used ``x=329 -> 0.1`` K,
``x=614 -> 1`` K and ``y=591 -> 0``, ``y=474 -> 0.5``, which moved both peaks
to 0.030 K and 0.095 K; curves extracted with it must not be used.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image


X_AT_0P1 = 389.5
X_AT_1 = 726.5
Y_AT_0 = 699.5
PIXELS_PER_0P5 = 138.0

FRAME = {"x_min": 113, "x_max": 1029, "y_min": 12, "y_max": 697}
# Panel (a) also contains an inset and a legend block; neither is data.
INSET = {"x": (598, 1035), "y": (0, 405)}
LEGEND = {"x": (225, 330), "y": (405, 585)}

SERIES = {
    "experiment": ((0.0, 69.0, 134.0), 60.0),
    "published_NLC_A": ((87.0, 157.0, 28.0), 70.0),
    "published_NLC_B": ((205.0, 60.0, 30.0), 90.0),
}


def temperature(x: float) -> float:
    return 10.0 ** (-1.0 + (x - X_AT_0P1) / (X_AT_1 - X_AT_0P1))


def heat_capacity(y: float) -> float:
    return 0.5 * (Y_AT_0 - y) / PIXELS_PER_0P5


def colour_mask(rgb: np.ndarray, colour, tolerance: float) -> np.ndarray:
    mask = np.linalg.norm(rgb - np.asarray(colour, dtype=float), axis=2) < tolerance
    keep = np.zeros(mask.shape, dtype=bool)
    keep[FRAME["y_min"] : FRAME["y_max"], FRAME["x_min"] : FRAME["x_max"]] = True
    keep[INSET["y"][0] : INSET["y"][1], INSET["x"][0] : INSET["x"][1]] = False
    keep[LEGEND["y"][0] : LEGEND["y"][1], LEGEND["x"][0] : LEGEND["x"][1]] = False
    return mask & keep


def column_centres(mask: np.ndarray) -> dict[int, float]:
    """One marker centre per pixel column.

    The experimental points carry error bars in both directions, so a plain
    median over the column is pulled off the marker.  Marker rows are far
    denser than error-bar rows, so keep the densest run when a column is tall.
    """
    centres = {}
    height = mask.shape[0]
    for x in range(FRAME["x_min"], FRAME["x_max"]):
        rows = np.flatnonzero(mask[:, x])
        if not len(rows):
            continue
        if len(rows) > 12:
            histogram = np.bincount(rows, minlength=height).astype(float)
            smoothed = np.convolve(histogram, np.ones(11), "same")
            rows = rows[np.abs(rows - int(np.argmax(smoothed))) <= 8]
        centres[x] = float(np.median(rows))
    return centres


def longest_contiguous(centres: dict[int, float], max_gap: int = 25) -> dict[int, float]:
    """Drop stray pixels outside the plotted range of a curve."""
    columns = sorted(centres)
    if not columns:
        return {}
    runs, current = [], [columns[0]]
    for previous, column in zip(columns, columns[1:]):
        if column - previous <= max_gap:
            current.append(column)
        else:
            runs.append(current)
            current = [column]
    runs.append(current)
    best = max(runs, key=len)
    return {column: centres[column] for column in best}


def binned(centres: dict[int, float], width: int) -> list[tuple[float, float]]:
    columns = sorted(centres)
    if not columns:
        return []
    points = []
    for low in range(columns[0], columns[-1] + 1, width):
        group = [column for column in columns if low <= column < low + width]
        if group:
            points.append(
                (float(np.mean(group)), float(np.median([centres[c] for c in group])))
            )
    return points


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path, help="Figure2.png from arXiv:2501.08327v5")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("ce2hf2o7_smith2025_digitized.csv"),
    )
    args = parser.parse_args()
    rgb = np.asarray(Image.open(args.image).convert("RGB"), dtype=float)

    traces = {
        name: binned(longest_contiguous(column_centres(colour_mask(rgb, colour, tol))), 4)
        for name, (colour, tol) in SERIES.items()
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        handle.write(
            "# Digitized from Fig. 2(a), Smith et al., Phys. Rev. Lett. 135, "
            "086702 (2025), arXiv:2501.08327v5.\n"
            "# Visual-comparison data only; no experimental uncertainties were "
            "inferred from the raster image.\n"
            "# Source Figure2.png SHA256: "
            "cc2f9ea660b8c26320557734d337c7513be6073be95143750412e0d6c87646a9.\n"
            f"# Calibration from plotted ticks: x={X_AT_0P1} -> 0.1 K, "
            f"x={X_AT_1} -> 1 K; y={Y_AT_0} -> 0, {PIXELS_PER_0P5} px per "
            "0.5 J mol_Ce^-1 K^-1.\n"
        )
        writer = csv.writer(handle)
        writer.writerow(["series", "temperature_K", "Cmag_J_molCe_K"])
        for series, trace in traces.items():
            for x, y in trace:
                writer.writerow(
                    [series, f"{temperature(x):.9g}", f"{heat_capacity(y):.9g}"]
                )
    for series, trace in traces.items():
        print(
            f"{series}: {len(trace)} points, "
            f"T = {temperature(trace[0][0]):.4g} to {temperature(trace[-1][0]):.4g} K"
        )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
