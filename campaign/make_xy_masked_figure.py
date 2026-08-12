#!/usr/bin/env python3
"""The (Jx, Jy) phase diagram again, winding-free: the mask instead of raw ED.

Same plane, same grid, same reflection as the naive scan -- but every
quantity is read off the masked Feshbach map, i.e. the exact fold of the
spinon space into the ice block with all transport-sector-crossing entries
deleted.  No counterterm is involved: a hexagon flux lives inside the ice
manifold, so the masked map's own ground multiplet is enough.

Panels, paired with the naive figure where a pairing exists:
  (a) ice weight of the winding-free ground multiplet, Z = [1-dSigma/dz]^-1
      -- against panel (a) of the naive figure;
  (b) gap above the protected ground multiplet, within the ice-descended
      band -- NOT the same object as the naive panel (b), which is the gap
      of the full spectrum; labelled accordingly rather than pretended to
      be comparable;
  (c) band completeness, roots surviving out of 90.  This takes the slot of
      the naive ice-violation panel, which cannot be reproduced mask-only:
      <sum_t Q_t^2> needs the full-space wavefunction, which is exactly
      what the masked map does not carry and the counterterm exists to
      supply.  Completeness is the better diagnostic here anyway -- where
      it falls below 90 the ice-descended band is dissolving into the
      continuum, and where it hits zero there is no band at all;
  (d) the hexagon flux: the 0-flux / pi-flux label, the panel the naive
      diagram gets wrong everywhere on the Jpm < 0 side.

Outputs: outputs/xy_phase_plane/fig_xy_masked.{pdf,png}
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import ListedColormap, LogNorm  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_xy_phase_figure import INK, overlay  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SCAN = ROOT / "campaign" / "outputs" / "xy_phase_plane"

PANELS = ("masked_ice_weight", "masked_gap", "masked_n_converged",
          "masked_flux", "masked_flippable")

# Phi = <sum O_h>/<sum N_h> is +-1 exactly for a flux eigenstate, so any
# tolerance far from 1 does; anything landing strictly between would be new
# physics and is reported rather than rounded away.
FLUX_TOL = 1e-6
FLIP_TOL = 1e-6


def load(n_grid: int):
    """Upper-triangle solves, mirrored: Jx <-> Jy is exact for all of these."""
    axis = np.linspace(-1.0, 1.0, n_grid)
    planes = {name: np.full((n_grid, n_grid), np.nan) for name in PANELS}
    files = sorted(SCAN.glob("masked_flux2_*.json"))
    if not files:
        raise SystemExit(f"no masked_flux2_*.json under {SCAN}")
    count = 0
    for path in files:
        for point in json.loads(path.read_text())["points"]:
            column = int(np.argmin(np.abs(axis - point["jx"])))
            row = int(np.argmin(np.abs(axis - point["jy"])))
            for name in PANELS:
                value = point.get(name, np.nan)
                for place in ((row, column), (column, row)):
                    planes[name][place] = value
            count += 1
    # A NaN flux is almost always the physical "no band survives here", not
    # missing data, so count the two separately rather than reporting a
    # thousand blank cells as if the scan had holes.
    covered = int(np.isfinite(planes["masked_n_converged"]).sum())
    print(f"{count} solved points from {len(files)} files; "
          f"{covered} of {n_grid * n_grid} cells covered, "
          f"{n_grid * n_grid - covered} never solved")
    return axis, planes


def main() -> None:
    n_grid = int(sys.argv[1]) if len(sys.argv) > 1 else 41
    axis, planes = load(n_grid)

    flux = planes["masked_flux"]
    flippable = planes["masked_flippable"]
    alive = planes["masked_n_converged"] > 0
    resonating = alive & (flippable > FLIP_TOL)
    finite = np.isfinite(flux) & resonating
    print(f"of {int(alive.sum())} cells with a band, "
          f"{int(resonating.sum())} have a flippable hexagon and "
          f"{int((alive & ~resonating).sum())} have none at all")
    stray = finite & (np.abs(np.abs(flux) - 1.0) > FLUX_TOL)
    print(f"cells with |Phi| not equal to 1: {int(stray.sum())}"
          + (f" -> {np.unique(np.round(flux[stray], 6))}" if stray.any()
             else " (the flux is a clean two-valued label)"))
    print(f"{int(finite.sum())} cells with a surviving band, "
          f"{int((~alive).sum())} with none")
    # Classify with a tolerance, and never with np.round: rounding 1e-31 to
    # six decimals prints "-0." and hides a whole population of genuinely
    # zero-amplitude cells among the pi-flux ones.  The three live classes
    # are physically distinct and get counted as such.
    resonant = np.abs(flux) > FLUX_TOL
    print(f"live cells: {int((finite & resonant & (flux < 0)).sum())} pi-flux"
          f", {int((finite & resonant & (flux > 0)).sum())} 0-flux"
          f", {int((finite & ~resonant).sum())} with no ring resonance"
          f" (|flux| < {FLUX_TOL:g})")
    magnitudes = np.unique(np.round(np.abs(flux[finite & resonant]), 9))
    print(f"distinct |flux| among resonant cells: {magnitudes}")

    step = axis[1] - axis[0]
    edges = np.concatenate([axis - step / 2, [axis[-1] + step / 2]])
    figure, axes = plt.subplots(1, 4, figsize=(18.2, 4.6),
                                constrained_layout=True)

    mesh0 = axes[0].pcolormesh(edges, edges, planes["masked_ice_weight"],
                               cmap="Blues", vmin=0.0, vmax=1.0,
                               shading="flat")
    figure.colorbar(mesh0, ax=axes[0], shrink=0.86).set_label(
        r"$Z$ (winding-free)", fontsize=10)
    axes[0].set_title("(a) ice weight, winding-free", fontsize=10.5,
                      loc="left", color=INK)
    axes[0].set_ylabel(r"$J_y/J_z$", fontsize=10.5)

    gap = np.where(planes["masked_gap"] > 0, planes["masked_gap"], np.nan)
    positive = gap[np.isfinite(gap)]
    # A log norm needs 0 < vmin < vmax, and neither is guaranteed here: most
    # of this plane has no band, so the gap can come back all-NaN or single
    # valued.  Fall back to linear rather than handing LogNorm a NaN.
    if positive.size and positive.max() > positive.min() > 0:
        norm = LogNorm(vmin=positive.min(), vmax=positive.max())
    else:
        norm = None
    mesh1 = axes[1].pcolormesh(edges, edges, gap, cmap="magma_r",
                               shading="flat", norm=norm)
    figure.colorbar(mesh1, ax=axes[1], shrink=0.86).set_label(
        "gap above the ground multiplet", fontsize=10)
    axes[1].set_title("(b) band gap (masked roots)", fontsize=10.5,
                      loc="left", color=INK)

    mesh2 = axes[2].pcolormesh(edges, edges, planes["masked_n_converged"],
                               cmap="viridis", vmin=0, vmax=90,
                               shading="flat")
    figure.colorbar(mesh2, ax=axes[2], shrink=0.86).set_label(
        "roots below the continuum (of 90)", fontsize=10)
    axes[2].set_title("(c) band completeness", fontsize=10.5, loc="left",
                      color=INK)

    # Categorical, not a diverging colormap.  The flux only ever takes three
    # values, so a continuous scale would put 261 exactly-zero cells at the
    # same white as a genuine midpoint and invite reading a gradient that
    # does not exist.
    label = np.full(flux.shape, np.nan)
    label[alive & (flippable <= FLIP_TOL)] = 1.0        # nothing flippable
    label[resonating & (np.abs(flux) <= FLUX_TOL)] = 2.0  # flippable, no res.
    label[resonating & (flux < -FLUX_TOL)] = 0.0
    label[resonating & (flux > FLUX_TOL)] = 3.0
    classes = ListedColormap(["#2c5f9e", "#cfcabb", "#efece2", "#b03a3a"])
    mesh3 = axes[3].pcolormesh(edges, edges, label, cmap=classes, vmin=-0.5,
                               vmax=3.5, shading="flat")
    bar = figure.colorbar(mesh3, ax=axes[3], shrink=0.86,
                          ticks=[0, 1, 2, 3])
    bar.ax.set_yticklabels([r"$\pi$-flux ($\Phi=-1$)", "no flippable hexagon",
                            r"flippable, $\Phi=0$", r"0-flux ($\Phi=+1$)"],
                           fontsize=8)
    axes[3].set_title(r"(d) flux sector", fontsize=10.5, loc="left",
                      color=INK)

    # Cells with no surviving band are NaN in every panel.  Grey them
    # explicitly so "the band has dissolved" cannot be misread as "the scan
    # did not reach here" -- on this plane it is most of the square.
    for handle in axes:
        handle.set_facecolor("#d9d7d0")
        overlay(handle, axis)
    axes[3].text(-0.95, -0.93, "grey: no ice-descended band", fontsize=8,
                 color=INK)

    for suffix in ("pdf", "png"):
        figure.savefig(SCAN / f"fig_xy_masked.{suffix}", dpi=300)
    print(f"wrote {SCAN}/fig_xy_masked.pdf and .png")


if __name__ == "__main__":
    main()
