#!/usr/bin/env python3
"""Figure: DSSF weight versus (Jpm, omega) at Gamma and X, both channels.

Eight panels from the ``run_dssf_sweep.py`` cache.  Top row: longitudinal
S^zz from the ice-block fold at the anchor, naive (mask off) against
winding-free (mask on) -- the winding-free Gamma panel is exactly empty, the
transport selection rule.  Bottom row: transverse S^xx from the exact
S_z=+1 Lanczos, where the variants differ only through the reference ground
state.

Two renderings of the same data: ``fig_dssf_sweep`` with omega on a log axis
(lines deposited with a 0.05 dex Gaussian in log10 omega) and
``fig_dssf_sweep_linear`` on a linear axis (fixed Gaussian eta of 0.015 Jzz
for the gauge row, 0.03 for the spinon row).  Vertical extent is broadening,
not lifetime, in both.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "campaign" / "outputs" / "dssf_sweep"
COUPLINGS = [-0.05, -0.10, -0.15, -0.18, -0.20, -0.22, -0.24, -0.26,
             -0.28, -0.30]
SIGMA_DEX = 0.05          # log-axis broadening
ETA_GAUGE = 0.015         # linear-axis broadening, S^zz row
ETA_SPINON = 0.03         # linear-axis broadening, S^xx row
OMEGA_MAX = 3.0
INK, MUTED = "#333333", "#8a8a8a"

ORDER = [("z", 0, "raw"), ("z", 0, "wf"), ("z", 1, "raw"), ("z", 1, "wf"),
         ("x", 0, "raw"), ("x", 0, "wf"), ("x", 1, "raw"), ("x", 1, "wf")]
TITLES = [
    r"$S^{zz}$ at $\Gamma$ — naive",
    r"$S^{zz}$ at $\Gamma$ — winding-free",
    r"$S^{zz}$ at $X$ — naive",
    r"$S^{zz}$ at $X$ — winding-free",
    r"$S^{xx}$ at $\Gamma$ — naive",
    r"$S^{xx}$ at $\Gamma$ — winding-free",
    r"$S^{xx}$ at $X$ — naive",
    r"$S^{xx}$ at $X$ — winding-free",
]


def load_lines():
    """(channel, q, variant) -> list over couplings of (poles, weights)."""
    lines = {key: [] for key in ORDER}
    for jpm in COUPLINGS:
        tag = f"jpm{jpm:+.6f}".replace(".", "p")
        with np.load(DATA / f"dssf_{tag}.npz") as record:
            for qi in (0, 1):
                lines[("z", qi, "raw")].append(
                    (record["z_exc_raw"], record["z_w_raw"][qi]))
                lines[("z", qi, "wf")].append(
                    (record["z_exc_wf"], record["z_w_wf"][qi]))
                for variant in ("raw", "wf"):
                    lines[("x", qi, variant)].append(
                        (record[f"x_{variant}_poles_q{qi}"],
                         record[f"x_{variant}_weights_q{qi}"]))
    return lines


def deposit_log(poles, weights, log_grid):
    poles = np.asarray(poles, float)
    weights = np.asarray(weights, float)
    keep = weights > 1e-14
    poles, weights = np.clip(poles[keep], 1e-4, None), weights[keep]
    if not len(poles):
        return np.zeros_like(log_grid)
    distance = (log_grid[:, None] - np.log10(poles)[None, :]) / SIGMA_DEX
    kernel = np.exp(-0.5 * distance**2) / (SIGMA_DEX * np.sqrt(2 * np.pi))
    return kernel @ weights


def deposit_linear(poles, weights, grid, eta):
    poles = np.asarray(poles, float)
    weights = np.asarray(weights, float)
    keep = weights > 1e-14
    poles, weights = np.clip(poles[keep], 0.0, None), weights[keep]
    if not len(poles):
        return np.zeros_like(grid)
    distance = (grid[:, None] - poles[None, :]) / eta
    kernel = np.exp(-0.5 * distance**2) / (eta * np.sqrt(2 * np.pi))
    return kernel @ weights


def render(lines, mode):
    if mode == "log":
        grid = np.geomspace(3e-4, OMEGA_MAX, 500)
        edges = np.geomspace(3e-4, OMEGA_MAX, len(grid) + 1)
        log_grid = np.log10(grid)
        def spectrum(key, poles, weights):
            return deposit_log(poles, weights, log_grid)
    else:
        grid = np.linspace(0.0, OMEGA_MAX, 600)
        edges = np.linspace(0.0, OMEGA_MAX, len(grid) + 1)
        def spectrum(key, poles, weights):
            eta = ETA_GAUGE if key[0] == "z" else ETA_SPINON
            return deposit_linear(poles, weights, grid, eta)

    panels = {}
    for key in ORDER:
        panel = np.zeros((len(grid), len(COUPLINGS)))
        for column, (poles, weights) in enumerate(lines[key]):
            panel[:, column] = spectrum(key, poles, weights)
        panels[key] = panel

    fig, axes = plt.subplots(2, 4, figsize=(15.2, 6.4), sharex=True,
                             sharey=True)
    fig.subplots_adjust(left=0.065, right=0.985, top=0.90, bottom=0.09,
                        hspace=0.14, wspace=0.32)
    x_edges = np.arange(len(COUPLINGS) + 1) - 0.5
    for ax, key, title in zip(axes.ravel(), ORDER, TITLES):
        vmax = float(panels[key].max())
        empty = vmax < 1e-12
        if empty:
            vmax = 1.0
        if mode == "log":
            norm = LogNorm(vmin=vmax * 1e-4, vmax=vmax)
            data = np.clip(panels[key], vmax * 1e-4, None)
        else:
            norm = plt.Normalize(vmin=0.0, vmax=vmax)
            data = panels[key]
        mesh = ax.pcolormesh(x_edges, edges, data, norm=norm,
                             cmap="Blues", rasterized=True)
        if not empty:
            bar = fig.colorbar(mesh, ax=ax, pad=0.03, aspect=16)
            bar.ax.tick_params(labelsize=7, colors=MUTED)
        if mode == "log":
            ax.set_yscale("log")
        ax.set_title(title, fontsize=10, color=INK)
        ax.set_xticks(range(len(COUPLINGS)))
        ax.set_xticklabels([f"{j:.2f}" if i % 3 == 0 else ""
                            for i, j in enumerate(COUPLINGS)], fontsize=8)
        ax.tick_params(colors=MUTED, labelsize=8.5)
        for spine in ax.spines.values():
            spine.set_color("#cccccc")

    if panels[("z", 0, "wf")].max() < 1e-10:
        axes[0, 1].text(0.5, 0.5,
                        "exactly zero\n(transport\nselection rule)",
                        transform=axes[0, 1].transAxes, ha="center",
                        va="center", fontsize=10, color=INK)

    for ax in axes[1]:
        ax.set_xlabel(r"$J_\pm/J_{zz}$", fontsize=10, color=INK)
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$\omega/J_{zz}$", fontsize=10, color=INK)
    fig.suptitle(
        "DSSF weight per site, cubic-16, ground-multiplet averaged  "
        "(top: longitudinal, ice-block fold at the anchor;  "
        "bottom: transverse, exact $S_z=\\pm1$ Lanczos)",
        fontsize=10.5, color=INK)
    stem = "fig_dssf_sweep" if mode == "log" else "fig_dssf_sweep_linear"
    for suffix in ("png", "pdf"):
        fig.savefig(DATA / f"{stem}.{suffix}", dpi=150)
    plt.close(fig)
    print("saved", DATA / f"{stem}.png")


def main() -> None:
    lines = load_lines()
    for mode in ("log", "linear"):
        render(lines, mode)


if __name__ == "__main__":
    main()
