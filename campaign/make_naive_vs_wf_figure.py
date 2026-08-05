#!/usr/bin/env python3
"""Naive ED against winding-free ED, on the XXZ line.

The (Jx, Jy) plane scan is naive ED throughout -- full spectrum, no mask,
no counterterm -- so "what does naive ED say" and "what does the plane say"
are the same question there.  The contrast that matters is against the
winding-free construction, and on the XXZ line that already exists: the
counterterm campaign cached H + C ground multiplets at fourteen couplings.

Four panels, all versus Jpm at Jpmpm = 0:
  (a) ice weight -- how much of the ground state is two-in--two-out;
  (b) hexagon flux -- the 0-flux / pi-flux label, where naive ED inverts;
  (c) the wrap-around amplitude -- the artefact itself, and its removal;
  (d) |<O_4>/<O_hex>|, artefact over physics, log scale: above one, the
      cluster's own wrap-around carries more weight than the ring exchange
      the physics is made of.

Comparability, stated rather than assumed: the H + C numbers are averages
over the ground multiplet (degeneracy 2 at most couplings), the naive ones
come from a non-degenerate ground state, and the two Hamiltonians differ by
C, so energies are NOT comparable and are not plotted.  Ice weight, the two
loop amplitudes and their ratio are the same operators in both.

Outputs: outputs/xy_phase_plane/fig_naive_vs_wf.{pdf,png}
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
from make_xy_phase_figure import ACCENT, INK, SECOND  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SCAN = ROOT / "campaign" / "outputs" / "xy_phase_plane"
COUNTERTERM = ROOT / "campaign" / "outputs" / "exact_counterterm"

FIELDS = ("ice_weight", "hexagon_flux", "winding4_amplitude")


def naive_diagonal():
    """Jx = Jy points from the flux scan: naive ED on the XXZ line."""
    rows = {}
    for path in sorted(SCAN.glob("flux_points_*.json")):
        for point in json.loads(path.read_text())["points"]:
            if abs(point["jx"] - point["jy"]) > 1e-12:
                continue
            rows[float(point["jpm"])] = {
                "ice_weight": point["ice_weight"],
                "hexagon_flux": point["hexagon_flux"],
                "winding4_amplitude": point["winding4_amplitude"],
            }
    jpm = np.array(sorted(rows))
    return jpm, {name: np.array([rows[value][name] for value in jpm])
                 for name in FIELDS}


def winding_free():
    """Cached H + C ground multiplets from the counterterm campaign."""
    rows = {}
    for path in sorted(COUNTERTERM.glob("exact_ct_*.json")):
        record = json.loads(path.read_text())
        if not all(name in record for name in FIELDS):
            continue
        rows[float(record["jpm"])] = record
    jpm = np.array(sorted(rows))
    return jpm, {name: np.array([rows[value][name] for value in jpm])
                 for name in FIELDS}


def style(handle, ylabel, title):
    handle.set_xlabel(r"$J_\pm/J_z$   (XXZ line, $J_{\pm\pm}=0$)",
                      fontsize=10)
    handle.set_ylabel(ylabel, fontsize=10)
    handle.set_title(title, fontsize=10.5, loc="left", color=INK)
    handle.set_xlim(-0.40, 0.09)
    handle.tick_params(labelsize=9, colors=SECOND)
    for spine in handle.spines.values():
        spine.set_color("#c3c2b7")


def main() -> None:
    naive_jpm, naive = naive_diagonal()
    wf_jpm, wf = winding_free()
    print(f"naive: {naive_jpm.size} couplings on the diagonal; "
          f"winding-free: {wf_jpm.size} cached")

    # The pure-Ising point has a 90-fold degenerate ground state, so every
    # eigenvector quantity there is whichever member the solver returned.
    naive = {name: np.where(np.abs(naive_jpm) < 1e-12, np.nan, values)
             for name, values in naive.items()}

    figure, axes = plt.subplots(1, 4, figsize=(17.6, 4.1),
                                constrained_layout=True)

    # Every winding-free series is drawn as two branches.  Jpm = 0 is a
    # change of flux sector, and the cached couplings straddle it with a
    # wide gap, so a joining segment would draw a crossing that no computed
    # point supports.
    sides = (wf_jpm < 0, wf_jpm > 0)

    axes[0].plot(naive_jpm, naive["ice_weight"], "-", lw=1.6, color=INK,
                 label="naive ED")
    for side, label in zip(sides, ("winding-free $H+C$", None)):
        axes[0].plot(wf_jpm[side], wf["ice_weight"][side], "o-", ms=3.4,
                     lw=1.3, color=ACCENT, label=label)
    axes[0].legend(fontsize=8.5, frameon=False)
    style(axes[0], r"$\langle P_{\rm ice}\rangle$", "(a) ice weight")

    axes[1].axhline(0.0, lw=0.9, color=INK, alpha=0.45)
    axes[1].plot(naive_jpm, naive["hexagon_flux"], "-", lw=1.6, color=INK)
    for side in sides:
        axes[1].plot(wf_jpm[side], wf["hexagon_flux"][side], "o-", ms=3.4,
                     lw=1.3, color=ACCENT)
    axes[1].text(-0.37, 0.05, "0-flux", fontsize=8.5, color=SECOND)
    axes[1].text(-0.37, -0.09, r"$\pi$-flux", fontsize=8.5, color=SECOND)
    style(axes[1], r"$\langle O_{\rm hex}+{\rm h.c.}\rangle$",
          "(b) flux label: naive inverts")

    axes[2].plot(naive_jpm, naive["winding4_amplitude"], "-", lw=1.6,
                 color=INK)
    for side in sides:
        axes[2].plot(wf_jpm[side], wf["winding4_amplitude"][side], "o-",
                     ms=3.4, lw=1.3, color=ACCENT)
    style(axes[2], r"$\langle O_4+{\rm h.c.}\rangle$",
          "(c) the wrap-around artefact")

    axes[3].axhline(1.0, lw=0.9, color=INK, alpha=0.45)
    axes[3].semilogy(naive_jpm,
                     np.abs(naive["winding4_amplitude"]
                            / naive["hexagon_flux"]), "-", lw=1.6, color=INK)
    wf_ratio = np.abs(wf["winding4_amplitude"] / wf["hexagon_flux"])
    for side in sides:
        axes[3].semilogy(wf_jpm[side], wf_ratio[side], "o-", ms=3.4, lw=1.3,
                         color=ACCENT)
    axes[3].text(-0.37, 1.35, "artefact dominates", fontsize=8.5,
                 color=SECOND)
    style(axes[3], r"$|\langle O_4\rangle/\langle O_{\rm hex}\rangle|$",
          "(d) artefact over physics")

    for suffix in ("pdf", "png"):
        figure.savefig(SCAN / f"fig_naive_vs_wf.{suffix}", dpi=300)
    print(f"wrote {SCAN}/fig_naive_vs_wf.pdf and .png")

    print("\nXXZ line, naive vs winding-free at the cached couplings:")
    print(f"{'Jpm':>7} {'Zice naive':>11} {'Zice wf':>9} "
          f"{'flux naive':>11} {'flux wf':>9} {'O4 naive':>9} {'O4 wf':>8}")
    for value in wf_jpm:
        near = int(np.argmin(np.abs(naive_jpm - value)))
        if abs(naive_jpm[near] - value) > 0.026:
            continue
        print(f"{value:+7.3f} {naive['ice_weight'][near]:11.4f} "
              f"{wf['ice_weight'][wf_jpm == value][0]:9.4f} "
              f"{naive['hexagon_flux'][near]:+11.4f} "
              f"{wf['hexagon_flux'][wf_jpm == value][0]:+9.4f} "
              f"{naive['winding4_amplitude'][near]:9.4f} "
              f"{wf['winding4_amplitude'][wf_jpm == value][0]:8.4f}")


if __name__ == "__main__":
    main()
