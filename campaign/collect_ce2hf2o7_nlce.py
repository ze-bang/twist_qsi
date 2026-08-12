#!/usr/bin/env python3
"""Copy the converged order-6 Ce2Hf2O7 NLCE curves into the paper repository.

The NLCE runs live outside this tree (ce2hf2o7_nlce/), so the curves the
figure draws are extracted once into campaign/data/ and the campaign becomes
self-contained.  Only the temperatures where the bare order-6 sum agrees with
order 5 to better than ``TOLERANCE`` are kept; everything below that is left
as NaN so the figure cannot silently draw an unconverged tail.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
NLCE = Path("/lustre09/project/6003507/zhouzb79/ce2hf2o7_nlce")
# Only set A is carried to order 6 for the figure.  The order-6 runs use the
# abelian symmetry lane (point_group=off); the auto max-clique search hangs for
# hours on the 19-site order-6 clusters, while the abelian folds give an
# identical spectrum in ~20 min per cluster.
TAGS = ("A",)
ORDER = 6
DIRECTORY = {"A": "o6off_setA"}
TOLERANCE = 0.02
GAS_CONSTANT = 8.31446261815324  # J K^-1 mol^-1; NLC_sum reports C in units of R

# Columns of nlc_resummation_comparison.txt.
TEMPERATURE = 0
BARE_LAST = 7   # highest order computed (6)
BARE_PREV = 8   # one order below (5)


def main() -> None:
    curves = {}
    reference = None
    for tag in TAGS:
        path = (NLCE / DIRECTORY[tag] / f"nlc_results_order_{ORDER}"
                / "nlc_resummation_comparison.txt")
        if not path.exists():
            sys.exit(f"missing {path}; the order-6 NLCE run has not landed")
        data = np.loadtxt(path)
        temperatures = data[:, TEMPERATURE]
        order6, order5 = data[:, BARE_LAST], data[:, BARE_PREV]
        converged = (
            np.abs(order6 - order5) / np.maximum(np.abs(order6), 1.0e-12)
            < TOLERANCE
        )
        # NLC_sum reports C in units of R; the figure axis is J K^-1 mol_Ce^-1.
        heat = np.where(converged, order6 * GAS_CONSTANT, np.nan)
        if reference is None:
            reference = temperatures
        elif not np.allclose(reference, temperatures):
            sys.exit(f"set {tag} is on a different temperature grid")
        curves[tag] = heat
        lowest = temperatures[converged].min() if converged.any() else float("nan")
        print(f"set {tag}: {converged.sum()}/{len(temperatures)} converged, "
              f"down to T = {lowest:.4f} K")

    output = ROOT / "campaign" / "data" / "ce2hf2o7_nlce_order6.csv"
    columns = np.column_stack([reference] + [curves[tag] for tag in TAGS])
    header = (
        "Order-6 pyrochlore NLCE, dipole-octupole XYZ, bare sum.\n"
        "Blank where the order-6 and order-5 bare sums differ by more than "
        f"{TOLERANCE:.0%}.\n"
        "temperature_K," + ",".join(f"C_{tag}" for tag in TAGS)
    )
    np.savetxt(output, columns, delimiter=",", header=header, comments="# ",
               fmt="%.8e")
    # np.savetxt writes the column names behind a comment marker; rewrite the
    # last header line bare so genfromtxt(names=True) picks it up.
    text = output.read_text().splitlines()
    text[2] = text[2].removeprefix("# ")
    output.write_text("\n".join(text) + "\n")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
