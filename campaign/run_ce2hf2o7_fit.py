#!/usr/bin/env python3
"""Rank converged FCC-32 winding-free XYZ curves against Ce2Hf2O7 data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from qsi_campaign.material_fit import load_digitized_series, rank_model_curves


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=ROOT / "campaign" / "material_fit_config.json"
    )
    parser.add_argument("--allow-unconverged", action="store_true")
    parser.add_argument("--top", type=int, default=10)
    args = parser.parse_args()

    config = json.loads(args.config.read_text())
    data_path = ROOT / config["data"]["path"]
    temperatures, heat = load_digitized_series(data_path)
    model_paths = sorted(ROOT.glob(config["model_curves"]["glob"]))
    if not model_paths:
        raise FileNotFoundError(
            "no FCC-32 XYZ production curves found; expected files matching "
            f"{config['model_curves']['glob']!r}. See campaign/README.md for the "
            "required NPZ contract."
        )

    window = tuple(float(value) for value in config["fit"]["temperature_window_K"])
    ranked = rank_model_curves(
        model_paths,
        temperatures,
        heat,
        window,
        require_converged=not args.allow_unconverged,
    )
    output = {
        "status": "exploratory_digitized_fit",
        "qualification": (
            "The source raster has no tabulated experimental uncertainties. "
            "Replace it with author data before quoting confidence intervals."
        ),
        "data": str(data_path),
        "temperature_window_K": list(window),
        "ranked_parameter_sets": ranked[: args.top],
    }
    output_path = ROOT / config["fit"]["output"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
