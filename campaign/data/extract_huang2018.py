#!/usr/bin/env python3
"""Extract C/N and S/N from Huang et al., PRL 120, 167202, Fig. 1(b)."""

from __future__ import annotations

import argparse
import csv
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np


SVG = "{http://www.w3.org/2000/svg}"
HEAT_PATHS = (84, 85, 86)
ENTROPY_PATHS = (629, 630, 631)

# Vector coordinates of the 10^-3 major x tick and one logarithmic decade.
X_AT_T_1E3 = 393.006966
X_PER_DECADE = 468.510787

# Both vertical axes have zero at this vector coordinate.
Y_AT_ZERO = 227.059072
Y_PER_HEAT_ONE = 3180.05
Y_PER_ENTROPY_ONE = 1141.68


def path_points(svg: Path, indices: tuple[int, ...]) -> np.ndarray:
    paths = list(ET.parse(svg).getroot().iter(f"{SVG}path"))
    points: list[tuple[float, float]] = []
    for index in indices:
        command = paths[index].attrib["d"]
        points.extend(
            (float(x), float(y))
            for x, y in re.findall(
                r"[ML]\s*([-+0-9.eE]+)\s+([-+0-9.eE]+)", command
            )
        )
    return np.asarray(points, dtype=float)


def merge_duplicate_x(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x_values, inverse = np.unique(points[:, 0], return_inverse=True)
    y_values = np.zeros_like(x_values)
    counts = np.zeros_like(x_values)
    np.add.at(y_values, inverse, points[:, 1])
    np.add.at(counts, inverse, 1.0)
    return x_values, y_values / counts


def digitize(svg: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    heat_x, heat_y = merge_duplicate_x(path_points(svg, HEAT_PATHS))
    entropy_x, entropy_y = merge_duplicate_x(path_points(svg, ENTROPY_PATHS))
    temperature = 10.0 ** (-3.0 + (heat_x - X_AT_T_1E3) / X_PER_DECADE)
    heat = (heat_y - Y_AT_ZERO) / Y_PER_HEAT_ONE
    entropy_native = (entropy_y - Y_AT_ZERO) / Y_PER_ENTROPY_ONE
    entropy = np.interp(heat_x, entropy_x, entropy_native)
    return temperature, heat, entropy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("svg", type=Path, help="SVG converted from arXiv source fig1.pdf")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).with_name("huang_2018_qmc_jpm_0p046.csv"),
    )
    args = parser.parse_args()
    temperature, heat, entropy = digitize(args.svg)
    with args.out.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (
                "temperature_over_Jzz",
                "heat_capacity_per_site",
                "entropy_per_site",
                "delta_log10_temperature",
                "delta_heat_capacity_per_site",
                "delta_entropy_per_site",
            )
        )
        for t_value, c_value, s_value in zip(temperature, heat, entropy):
            writer.writerow(
                (
                    f"{t_value:.10g}",
                    f"{c_value:.10g}",
                    f"{s_value:.10g}",
                    "0.002",
                    "0.0015",
                    "0.002",
                )
            )
    print(f"wrote {len(temperature)} points to {args.out}")


if __name__ == "__main__":
    main()
