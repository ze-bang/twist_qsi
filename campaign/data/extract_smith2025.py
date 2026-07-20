#!/usr/bin/env python3
"""Extract visual-comparison curves from Smith et al. PRL 135, 086702."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image


X_AT_0P1 = 329.0
X_AT_1 = 614.0
Y_AT_0 = 591.0
Y_AT_0P5 = 474.0


def temperature(x: float) -> float:
    return 10.0 ** (-1.0 + (x - X_AT_0P1) / (X_AT_1 - X_AT_0P1))


def heat_capacity(y: float) -> float:
    return 0.5 * (Y_AT_0 - y) / (Y_AT_0 - Y_AT_0P5)


def grouped_trace(mask, x_min, x_max, width, y_selector):
    pixels = []
    for x in range(x_min, x_max + 1):
        yy = y_selector(x, np.flatnonzero(mask[:, x]))
        if len(yy):
            pixels.append((x, float(np.median(yy))))
    result = []
    for lo in range(x_min, x_max + 1, width):
        group = [(x, y) for x, y in pixels if lo <= x < lo + width]
        if group:
            result.append(
                (
                    float(np.mean([x for x, _ in group])),
                    float(np.median([y for _, y in group])),
                )
            )
    return result


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

    blue = np.linalg.norm(rgb - np.array([0.0, 69.0, 134.0]), axis=2) < 48.0
    experimental = grouped_trace(
        blue,
        120,
        720,
        5,
        lambda x, yy: yy[
            (yy > (35 if x < 450 else 180)) & (yy < (285 if x < 450 else 592))
        ],
    )
    green = np.linalg.norm(rgb - np.array([87.0, 157.0, 28.0]), axis=2) < 60.0
    nlc_a = grouped_trace(
        green, 300, 620, 4, lambda _x, yy: yy[(yy > 50) & (yy < 360)]
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        handle.write(
            "# Digitized from Fig. 2(a), Smith et al., Phys. Rev. Lett. 135, "
            "086702 (2025), arXiv:2501.08327v5.\n"
            "# Visual-comparison data only; no experimental uncertainties were "
            "inferred from the raster image.\n"
            "# Source Figure2.png SHA256: "
            "cc2f9ea660b8c26320557734d337c7513be6073be95143750412e0d6c87646a9.\n"
            "# Calibration: x=329 -> 0.1 K, x=614 -> 1 K; y=591 -> 0, "
            "y=474 -> 0.5 J mol_Ce^-1 K^-1.\n"
        )
        writer = csv.writer(handle)
        writer.writerow(["series", "temperature_K", "Cmag_J_molCe_K"])
        for series, trace in (("experiment", experimental), ("published_NLC_A", nlc_a)):
            for x, y in trace:
                writer.writerow(
                    [series, f"{temperature(x):.9g}", f"{heat_capacity(y):.9g}"]
                )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
