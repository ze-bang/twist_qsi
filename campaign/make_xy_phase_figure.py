#!/usr/bin/env python3
"""Ground-state map of cubic-16 XYZ over the (Jx, Jy) square at Jz = 1.

Three panels, all from the exact 65,536-state spectrum:
  (a) ice weight of the ground state, <psi|P_ice|psi> -- how much of the
      true ground state lives in the two-in--two-out manifold, i.e. whether
      the ice description applies at all;
  (b) the ground-state gap on a log scale -- an ordered phase on a finite
      cluster announces itself as a quasi-degenerate multiplet, so the gap
      collapsing by orders of magnitude is the boundary diagnostic;
  (c) the mean ice violation <sum_t Q_t^2>, the complementary view: zero in
      the ice manifold, large where monopoles proliferate.

Annotated: the XXZ line Jx = Jy (the campaign's Jpmpm = 0 axis), the pure
pair-flip line Jx = -Jy (Jpm = 0), and the 56 cells of the (Jpm, Jpmpm)
campaign grid, which occupy only a narrow diagonal band of this square.

Because Jx <-> Jy is the exact symmetry Jpmpm -> -Jpmpm, every panel must be
symmetric about the main diagonal; the script measures that asymmetry and
prints it as a self-check rather than assuming it.

Outputs: outputs/xy_phase_plane/fig_xy_phase.{pdf,png}
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SCAN = ROOT / "campaign" / "outputs" / "xy_phase_plane"

CAMPAIGN_JPM = [-0.05, -0.09, -0.125, -0.16, -0.20, -0.24, -0.27, -0.30]
CAMPAIGN_PP = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]

INK = "#0b0b0b"
SECOND = "#52514e"
GRIDCOL = "#e1e0d9"
ACCENT = "#d03b3b"


def load():
    files = sorted(SCAN.glob("xyscan_col*_n*.json"))
    if not files:
        raise SystemExit(f"no scan columns under {SCAN}")
    # a column may have been scanned twice (pilot + array); keep the newest
    by_column = {}
    for path in files:
        record = json.loads(path.read_text())
        by_column[int(record["column"])] = record
    n_grid = max(r["n_grid"] for r in by_column.values())
    axis = np.linspace(-1.0, 1.0, n_grid)
    shape = (n_grid, n_grid)
    weight = np.full(shape, np.nan)
    gap = np.full(shape, np.nan)
    violation = np.full(shape, np.nan)
    for column, record in by_column.items():
        for point in record["points"]:
            row = int(np.argmin(np.abs(axis - point["jy"])))
            weight[row, column] = point["ice_weight"]
            gap[row, column] = max(point["gap"], 1e-12)
            violation[row, column] = point["ice_violation"]
    missing = int(np.isnan(weight).sum())
    return axis, weight, gap, violation, len(by_column), missing


def symmetry_check(name, field):
    """Jx <-> Jy is exact; report the measured breach rather than assume it."""
    finite = np.isfinite(field) & np.isfinite(field.T)
    if not finite.any():
        return np.nan
    deviation = float(np.max(np.abs(field - field.T)[finite]))
    print(f"  {name:16s} max |M - M^T| = {deviation:.3e}")
    return deviation


def overlay(axis_handle, axis_values):
    limit = axis_values[-1]
    axis_handle.plot([-limit, limit], [-limit, limit], "-", lw=1.1,
                     color=INK, alpha=0.5)
    axis_handle.plot([-limit, limit], [limit, -limit], "--", lw=1.1,
                     color=INK, alpha=0.35)
    xs, ys = [], []
    for jpm in CAMPAIGN_JPM:
        for jpmpm in CAMPAIGN_PP:
            xs.append(2.0 * (jpmpm - jpm))
            ys.append(-2.0 * (jpm + jpmpm))
    inside = [(x, y) for x, y in zip(xs, ys)
              if abs(x) <= limit and abs(y) <= limit]
    if inside:
        axis_handle.plot([p[0] for p in inside], [p[1] for p in inside],
                         "o", ms=2.4, mfc="none", mec=ACCENT, mew=0.8)
    axis_handle.set_xlim(-limit, limit)
    axis_handle.set_ylim(-limit, limit)
    axis_handle.set_xlabel(r"$J_x/J_z$", fontsize=10.5)
    axis_handle.set_aspect("equal")
    axis_handle.tick_params(labelsize=9, colors=SECOND)
    for spine in axis_handle.spines.values():
        spine.set_color("#c3c2b7")


def main() -> None:
    axis, weight, gap, violation, columns, missing = load()
    print(f"{columns} columns loaded, {missing} grid points missing")
    print("symmetry check (Jx <-> Jy must be exact):")
    for name, field in (("ice weight", weight), ("gap", gap),
                        ("ice violation", violation)):
        symmetry_check(name, field)

    step = axis[1] - axis[0]
    edges = np.concatenate([axis - step / 2, [axis[-1] + step / 2]])

    figure, axes = plt.subplots(1, 3, figsize=(13.4, 4.6),
                                constrained_layout=True)

    mesh0 = axes[0].pcolormesh(edges, edges, weight, cmap="Blues",
                               vmin=0.0, vmax=1.0, shading="flat")
    figure.colorbar(mesh0, ax=axes[0], shrink=0.86).set_label(
        r"$\langle P_{\rm ice}\rangle$", fontsize=10)
    axes[0].set_title("(a) ice weight of the ground state", fontsize=10.5,
                      loc="left", color=INK)
    axes[0].set_ylabel(r"$J_y/J_z$", fontsize=10.5)

    finite_gap = gap[np.isfinite(gap)]
    mesh1 = axes[1].pcolormesh(
        edges, edges, gap, cmap="magma_r", shading="flat",
        norm=LogNorm(vmin=max(finite_gap.min(), 1e-8), vmax=finite_gap.max()))
    figure.colorbar(mesh1, ax=axes[1], shrink=0.86).set_label(
        r"gap $E_1-E_0$", fontsize=10)
    axes[1].set_title("(b) ground-state gap", fontsize=10.5, loc="left",
                      color=INK)

    mesh2 = axes[2].pcolormesh(edges, edges, violation, cmap="rocket_r"
                               if "rocket_r" in plt.colormaps() else "viridis",
                               shading="flat")
    figure.colorbar(mesh2, ax=axes[2], shrink=0.86).set_label(
        r"$\langle \sum_t Q_t^2\rangle$", fontsize=10)
    axes[2].set_title(r"(c) mean ice violation", fontsize=10.5, loc="left",
                      color=INK)

    for handle in axes:
        overlay(handle, axis)

    for suffix in ("pdf", "png"):
        figure.savefig(SCAN / f"fig_xy_phase.{suffix}", dpi=300)
    print(f"wrote {SCAN}/fig_xy_phase.pdf and .png")


if __name__ == "__main__":
    main()
