"""Rerun the 32-site FCC scheme comparison at PT order 4 and regenerate the
hero figures (16-site panels reloaded from the saved exact npz data)."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import operator_vs_observable as ovo

FIGS = ovo.FIGS


def load16(jpm):
    d = np.load(FIGS / f"avg_schemes_16_jpm{jpm}.npz")
    return {k: d[k] for k in
            ("C_corners", "C_bare", "C_obs", "C_op", "C_d0", "C_ct",
             "g4", "ghex")}


def main():
    with open(FIGS / "avg_schemes_summary.json") as f:
        summary = json.load(f)
    d32a = ovo.run32(-0.05, summary, order=4)
    d32b = ovo.run32(-0.10, summary, order=4)
    d16a, d16b = load16(-0.1), load16(-0.05)
    ovo.hero_figure(d16a, d32b, -0.10, -0.10, "fig_avg_schemes.pdf")
    ovo.hero_figure(d16b, d32a, -0.05, -0.05, "fig_avg_schemes_jpm005.pdf")
    with open(FIGS / "avg_schemes_summary.json", "w") as f:
        json.dump(summary, f, indent=1)
    print("DONE")


if __name__ == "__main__":
    main()
