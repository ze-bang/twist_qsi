#!/usr/bin/env python3
"""Extract the thermodynamic-limit QMC C/N curve from Kato-Onoda Fig. 2(b)."""

from __future__ import annotations

import argparse
import csv
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np


SVG = "{http://www.w3.org/2000/svg}"
PATH_INDICES = range(1774, 1780)
X_AT_T_1E3 = 124.64
X_PER_DECADE = 109.34
Y_AT_C_ZERO = 34.08
Y_PER_C_ONE = 814.64


def path_points(svg: Path) -> np.ndarray:
    paths = list(ET.parse(svg).getroot().iter(f"{SVG}path"))
    points = []
    for index in PATH_INDICES:
        command = paths[index].attrib["d"]
        points.extend(
            (float(x), float(y))
            for x, y in re.findall(r"[ML]\s*([-+0-9.eE]+)\s+([-+0-9.eE]+)", command)
        )
    data = np.asarray(points)
    # The first two points are the legend sample, not the curve.
    return data[2:]


def digitize(svg: Path, bins: int = 96) -> tuple[np.ndarray, np.ndarray]:
    points = path_points(svg)
    log_temperature = -3.0 + (points[:, 0] - X_AT_T_1E3) / X_PER_DECADE
    heat = (points[:, 1] - Y_AT_C_ZERO) / Y_PER_C_ONE
    edges = np.linspace(log_temperature.min(), log_temperature.max(), bins + 1)
    centers, values = [], []
    for lower, upper in zip(edges[:-1], edges[1:]):
        mask = (log_temperature >= lower) & (log_temperature < upper)
        if np.any(mask):
            centers.append(np.median(log_temperature[mask]))
            values.append(np.median(heat[mask]))
    order = np.argsort(centers)
    return 10.0 ** np.asarray(centers)[order], np.asarray(values)[order]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("svg", type=Path, help="SVG converted from Kato-Onoda fig2_new.pdf")
    parser.add_argument("--out", type=Path, default=Path(__file__).with_name("kato_onoda_2015_qmc_jpm_p1over22.csv"))
    args = parser.parse_args()
    temperature, heat = digitize(args.svg)
    with args.out.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("temperature_over_Jzz", "heat_capacity_per_site", "delta_log10_temperature", "delta_heat_capacity_per_site"))
        for t_value, c_value in zip(temperature, heat):
            writer.writerow((f"{t_value:.9g}", f"{c_value:.9g}", "0.015", "0.003"))
    print(f"wrote {len(temperature)} points to {args.out}")


if __name__ == "__main__":
    main()
