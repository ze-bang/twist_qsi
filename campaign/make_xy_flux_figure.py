#!/usr/bin/env python3
"""Hexagon flux over the (Jx, Jy) square, and why the raw cluster misreads it.

The 0-flux / pi-flux label of a quantum spin ice is the sign of the
contractible hexagon amplitude <O_hex + h.c.> in the ground state.  This
script maps it over the plane and then makes the case that the raw map is
not the physical answer:

  (a) raw <O_hex>, diverging about zero -- the naive flux label;
  (b) raw <O_4>, the wrap-around four-loop amplitude, i.e. the finite-size
      artefact the mask exists to remove;
  (c) their ratio, log10 |<O_4>/<O_hex>|, positive wherever the artefact
      loop carries more amplitude than the physical one;
  (d) the XXZ cut, raw against the cached winding-free (H + C) values.

Panel (d) is the point.  g_4 = 4 Jpm^2/Jzz is EVEN in Jpm while
g_6 = 12|Jpm|^3/Jzz^2 is ODD, so on the Jpm > 0 side artefact and physics
push the same way and the raw sign is accidentally right, while on the
Jpm < 0 side -- every cerium pyrochlore -- they fight, the artefact wins on
this cluster, and raw ED reports 0-flux where the winding-free state is
firmly pi-flux.  Reading a flux phase diagram off raw ED would therefore
invert the label across the whole region of interest.

Outputs: outputs/xy_phase_plane/fig_xy_flux.{pdf,png}
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_xy_phase_figure import ACCENT, INK, SECOND, overlay  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SCAN = ROOT / "campaign" / "outputs" / "xy_phase_plane"
COUNTERTERM = ROOT / "campaign" / "outputs" / "exact_counterterm"


def load_flux(n_grid: int):
    """Assemble the upper-triangle solves into full planes by reflection.

    Jx <-> Jy leaves the hexagon operator invariant (the pi/2 rotation that
    sends Jpmpm -> -Jpmpm gives three S+ and three S- a net (i)^3(-i)^3 = 1),
    so the triangle determines the square.
    """
    axis = np.linspace(-1.0, 1.0, n_grid)
    hexagon = np.full((n_grid, n_grid), np.nan)
    winding = np.full((n_grid, n_grid), np.nan)
    files = sorted(SCAN.glob("flux_points_*.json"))
    if not files:
        raise SystemExit(f"no flux_points_*.json under {SCAN}")
    count = 0
    for path in files:
        for point in json.loads(path.read_text())["points"]:
            column = int(np.argmin(np.abs(axis - point["jx"])))
            row = int(np.argmin(np.abs(axis - point["jy"])))
            for place in ((row, column), (column, row)):
                hexagon[place] = point["hexagon_flux"]
                winding[place] = point["winding4_amplitude"]
            count += 1
    print(f"{count} solved points from {len(files)} files; "
          f"{int(np.isnan(hexagon).sum())} cells blank")
    return axis, hexagon, winding


def counterterm_curve():
    """Cached winding-free flux on the XXZ line, from the H + C campaign."""
    jpm, flux = [], []
    for path in sorted(COUNTERTERM.glob("exact_ct_*.json")):
        record = json.loads(path.read_text())
        if "hexagon_flux" not in record:
            continue
        jpm.append(float(record["jpm"]))
        flux.append(float(record["hexagon_flux"]))
    order = np.argsort(jpm)
    return np.asarray(jpm)[order], np.asarray(flux)[order]


def main() -> None:
    n_grid = int(sys.argv[1]) if len(sys.argv) > 1 else 41
    axis, hexagon, winding = load_flux(n_grid)
    finite = np.isfinite(hexagon)
    print(f"raw hexagon flux: min {np.nanmin(hexagon):+.4f}, "
          f"max {np.nanmax(hexagon):+.4f}, "
          f"negative in {int((hexagon[finite] < 0).sum())} of "
          f"{int(finite.sum())} cells")
    # No symmetry self-check is printed here on purpose: load_flux writes
    # both (i, j) and (j, i) from one solve, so any such check would return
    # zero by construction and mean nothing.  The reflection is tested for
    # real, by solving both sides, in check_flux_symmetry.py.

    step = axis[1] - axis[0]
    edges = np.concatenate([axis - step / 2, [axis[-1] + step / 2]])
    figure, axes = plt.subplots(1, 4, figsize=(18.2, 4.6),
                                constrained_layout=True)

    span = float(np.nanmax(np.abs(hexagon)))
    mesh0 = axes[0].pcolormesh(edges, edges, hexagon, cmap="RdBu_r",
                               vmin=-span, vmax=span, shading="flat")
    figure.colorbar(mesh0, ax=axes[0], shrink=0.86).set_label(
        r"$\langle O_{\rm hex}+{\rm h.c.}\rangle$ (raw)", fontsize=10)
    axes[0].set_title("(a) raw hexagon flux", fontsize=10.5, loc="left",
                      color=INK)
    axes[0].set_ylabel(r"$J_y/J_z$", fontsize=10.5)
    negative = int((hexagon[np.isfinite(hexagon)] < 0).sum())
    axes[0].text(-0.95, -0.93, f"negative in {negative} of "
                 f"{int(np.isfinite(hexagon).sum())} cells",
                 fontsize=8, color=INK)

    mesh1 = axes[1].pcolormesh(edges, edges, winding, cmap="viridis",
                               shading="flat")
    figure.colorbar(mesh1, ax=axes[1], shrink=0.86).set_label(
        r"$\langle O_4+{\rm h.c.}\rangle$ (raw)", fontsize=10)
    axes[1].set_title("(b) wrap-around artefact", fontsize=10.5, loc="left",
                      color=INK)

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.log10(np.abs(winding / hexagon))
    reach = float(np.nanmax(np.abs(ratio[np.isfinite(ratio)])))
    mesh2 = axes[2].pcolormesh(edges, edges, ratio, cmap="PuOr_r",
                               vmin=-reach, vmax=reach, shading="flat")
    figure.colorbar(mesh2, ax=axes[2], shrink=0.86).set_label(
        r"$\log_{10}|\langle O_4\rangle / \langle O_{\rm hex}\rangle|$",
        fontsize=10)
    axes[2].set_title("(c) artefact over physics", fontsize=10.5, loc="left",
                      color=INK)

    for handle in axes[:3]:
        overlay(handle, axis)

    diagonal = np.array([hexagon[k, k] for k in range(n_grid)])
    jpm_axis = -axis / 2.0            # Jx = Jy = t  =>  Jpm = -t/2
    # At Jx = Jy = 0 the Hamiltonian is pure Ising and the ground state is
    # the whole 90-fold ice manifold, so the solver returns an arbitrary
    # member and its flux means nothing.  Break the curve there rather than
    # draw the resulting dip as if it were physics.
    origin = int(np.argmin(np.abs(jpm_axis)))
    diagonal[origin] = np.nan
    axes[3].axhline(0.0, lw=0.9, color=INK, alpha=0.45)
    axes[3].plot(jpm_axis, diagonal, "-", lw=1.6, color=INK,
                 label=r"raw ED (this scan)")
    ct_jpm, ct_flux = counterterm_curve()
    # Two branches, not one curve: the sign change at Jpm = 0 is a change of
    # flux sector, and joining across it would draw a crossing that no
    # computed point supports.
    for side, label in ((ct_jpm < 0, r"winding-free $H+C$"), (ct_jpm > 0, None)):
        axes[3].plot(ct_jpm[side], ct_flux[side], "o-", ms=3.4, lw=1.3,
                     color=ACCENT, label=label)
    axes[3].set_xlim(-0.42, 0.12)
    axes[3].set_xlabel(r"$J_\pm/J_z$ (on the XXZ line)", fontsize=10.5)
    axes[3].set_ylabel(r"$\langle O_{\rm hex}+{\rm h.c.}\rangle$",
                       fontsize=10.5)
    axes[3].set_title("(d) the raw label is inverted", fontsize=10.5,
                      loc="left", color=INK)
    axes[3].legend(fontsize=8.5, frameon=False)
    axes[3].tick_params(labelsize=9, colors=SECOND)
    axes[3].text(-0.34, 0.055, "0-flux", fontsize=8.5, color=SECOND)
    axes[3].text(-0.34, -0.075, r"$\pi$-flux", fontsize=8.5, color=SECOND)
    for spine in axes[3].spines.values():
        spine.set_color("#c3c2b7")

    for suffix in ("pdf", "png"):
        figure.savefig(SCAN / f"fig_xy_flux.{suffix}", dpi=300)
    print(f"wrote {SCAN}/fig_xy_flux.pdf and .png")


if __name__ == "__main__":
    main()
