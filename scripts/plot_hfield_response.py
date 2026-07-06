"""
Plot the transverse-field (h_perp * S^x_i) response computed by
sweep_transverse_field.py: twist-averaged vs bare-corner ground-state
curvature d^2E0/dh^2 at both flux signs (Jpm=-0.1 "pi-flux" and
Jpm=+0.05 "0-flux"), and the extracted curvature ratio compared against
the hexagon-order (J_pm^3) and 4-cycle-order (J_pm^2) combinatorial
predictions.

Reads ../output/hfield_sweep/summary.json and
../output/hfield_sweep_fine/summary.json (produced by
sweep_transverse_field.py --corners eight), writes
fig_hfield_response.{png,pdf} into ../paper/figs/.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output"
FIGS = ROOT / "paper" / "figs"

# Fine-grid points below the Lanczos tolerance floor (1e-9) for these h
# values give |dE| below solver noise -- drop them.
DROP_H = {0.0005, 0.001, -0.0005, -0.001}

COL_PI = "#2a78d6"     # blue  -- pi-flux, Jpm = -0.1
COL_ZERO = "#e34948"   # red   -- 0-flux, Jpm = +0.05
GRID = "#c9c8c2"
INK_SEC = "#52514e"

FLUX_LABEL = {"piflux": r"$\pi$-flux ($J_\pm=-0.1$)", "zeroflux": r"0-flux ($J_\pm=+0.05$)"}
FLUX_COLOR = {"piflux": COL_PI, "zeroflux": COL_ZERO}


def load_data():
    rows = []
    for name in ["hfield_sweep", "hfield_sweep_fine"]:
        p = OUT / name / "summary.json"
        if p.exists():
            rows += json.loads(p.read_text())["rows"]

    per_corner = defaultdict(lambda: defaultdict(list))
    for r in rows:
        with h5py.File(r["spec_h5"], "r") as f:
            e0 = float(np.sort(f["/eigendata/eigenvalues"][...])[0])
        per_corner[r["flux"]][r["h_perp"]].append((tuple(r["phi"]), e0))

    data = defaultdict(dict)
    for flux, hmap in per_corner.items():
        for h, lst in hmap.items():
            if h in DROP_H:
                continue
            vals = [e for _, e in lst]
            data[flux].setdefault("avg", {})[h] = float(np.mean(vals))
            bare = [e for phi, e in lst if phi == (0.0, 0.0, 0.0)]
            if bare:
                data[flux].setdefault("bare", {})[h] = bare[0]
    return data


def fit_a2(hs, Es, hmax=0.02):
    hs = np.asarray(hs); Es = np.asarray(Es)
    mask = np.abs(hs) <= hmax
    h0 = Es[np.argmin(np.abs(hs))]
    A = np.stack([hs[mask] ** 2, hs[mask] ** 4], axis=1)
    coef, *_ = np.linalg.lstsq(A, Es[mask] - h0, rcond=None)
    return coef[0], coef[1]


def main():
    data = load_data()

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.3))

    a2_vals = {}
    for ax, kind, title in [(axes[0], "avg", "(a) twist-averaged (8 corners)"),
                            (axes[1], "bare", r"(b) bare corner ($\varphi=0$)")]:
        for flux in ["piflux", "zeroflux"]:
            hd = data[flux][kind]
            hs = np.array(sorted(hd.keys()))
            Es = np.array([hd[h] for h in hs])
            pos = hs > 0
            h0 = Es[hs == 0][0]
            dE = np.abs(Es[pos] - h0)
            hp = hs[pos]
            a2, a4 = fit_a2(hs, Es)
            a2_vals.setdefault(kind, {})[flux] = a2
            ax.plot(hp, dE, "o", ms=6.5, color=FLUX_COLOR[flux], mec="white", mew=0.6,
                    zorder=4, label=FLUX_LABEL[flux])
            hfit = np.geomspace(hp.min(), hp.max(), 50)
            ax.plot(hfit, np.abs(a2) * hfit ** 2, "-", lw=1.6, color=FLUX_COLOR[flux],
                    alpha=0.55, zorder=3)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel(r"$h_\perp\ /\ J_{zz}$")
        ax.set_ylabel(r"$|E_0(h_\perp)-E_0(0)|\ /\ J_{zz}$")
        ax.set_title(title, fontsize=11)
        ax.grid(alpha=0.25, color=GRID, which="both")
        ax.legend(frameon=False, fontsize=8.5, loc="upper left")

    ax0 = axes[0]
    x0, x1 = 0.006, 0.03
    y0 = 0.0006
    ax0.plot([x0, x1], [y0, y0 * (x1 / x0) ** 2], color=INK_SEC, lw=1.0, ls=(0, (2, 2)))
    ax0.text(x1 * 1.15, y0 * (x1 / x0) ** 2, "slope 2\n(quadratic in $h$)",
             fontsize=7.5, color=INK_SEC, va="center")

    ax2 = axes[2]
    ratio_avg = abs(a2_vals["avg"]["zeroflux"] / a2_vals["avg"]["piflux"])
    ratio_bare = abs(a2_vals["bare"]["zeroflux"] / a2_vals["bare"]["piflux"])

    xs = [0, 1]
    ax2.bar(xs, [ratio_avg, ratio_bare], width=0.55,
            color=["#1baf7a", "#9a9890"], edgecolor="black", linewidth=0.6, zorder=3)
    ax2.set_xticks(xs)
    ax2.set_xticklabels(["twist-averaged\n(cleaned)", "bare corner\n(uncleaned)"], fontsize=9)
    ax2.axhline(8, color=COL_PI, ls="--", lw=1.3,
                label=r"$(J_\pm^{\pi}/J_\pm^{0})^3=8$  (hexagon / $g_{\rm hex}$ scaling)")
    ax2.axhline(4, color="#eda100", ls=":", lw=1.3,
                label=r"$(J_\pm^{\pi}/J_\pm^{0})^2=4$  (4-cycle / $g_4^{\rm spur}$ scaling)")
    for x, v in zip(xs, [ratio_avg, ratio_bare]):
        ax2.text(x, v + 1.0, f"{v:.1f}", ha="center", fontsize=10, fontweight="bold")
    ax2.set_ylabel(r"curvature ratio  $|a_2^{\rm 0\text{-}flux}/a_2^{\pi\text{-}flux}|$")
    ax2.set_title("(c) which combinatorics sets the field response?", fontsize=11)
    ax2.legend(frameon=False, fontsize=8, loc="upper left")
    ax2.grid(alpha=0.25, color=GRID, axis="y")
    ax2.set_ylim(0, 45)

    fig.suptitle(
        r"Transverse-field response $h_\perp S^x_i$ on the $1{\times}1{\times}1$ cluster: "
        r"twist-averaging isolates hexagon-order ($J_\pm^3$) scaling",
        fontsize=11.5, y=1.03)
    fig.tight_layout()

    FIGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGS / "fig_hfield_response.png", dpi=170, bbox_inches="tight")
    fig.savefig(FIGS / "fig_hfield_response.pdf", bbox_inches="tight")
    print("wrote", FIGS / "fig_hfield_response.png")
    print("ratio_avg =", ratio_avg, "ratio_bare =", ratio_bare)


if __name__ == "__main__":
    main()
