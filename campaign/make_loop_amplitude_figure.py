"""K_L versus coupling: absolute amplitudes, suppression, hierarchy, normalization.

Four panels, one message each:
  (a) FCC-32 absolute K_L with the leading perturbative power laws
  (b) FCC-32 suppression K_L / K_L^(0) -- how far past perturbation theory
  (c) FCC-32 hierarchy K_L / K_6 against the perturbative ratios
  (d) cubic-16 Brillouin-Wigner vs Hermitian normalization (L = 6, 8)

Panel (d) is the calibration that decides whether (b) and (c) are conventions
or physics; cubic-16 masks every L >= 8 loop, so its L = 8 row is an unmasked
class amplitude and is labelled as such.

Data: campaign/outputs/wp1_tomography/tomo_fcc32_L{2,3}_*.json
      campaign/outputs/map_normalization/normalization.json
"""

from __future__ import annotations

import glob
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TOMO = ROOT / "campaign" / "outputs" / "wp1_tomography"
NORM = ROOT / "campaign" / "outputs" / "map_normalization" / "normalization.json"
OUT = ROOT / "campaign" / "outputs" / "map_normalization"

PERIMS = (6, 8, 10, 12)
PERT = {6: 12.0, 8: 40.0, 10: 140.0, 12: 504.0}
# dataviz categorical slots 1-4 (validated: all checks pass, light surface)
COLOR = {6: "#2a78d6", 8: "#eb6834", 10: "#1baf7a", 12: "#eda100"}
INK, INK2, GRID = "#1a1a19", "#55554e", "#dedcd5"


def k0(length, jpm):
    return PERT[length] * abs(jpm) ** (length // 2)


def load_fcc32(depth, sampled):
    """{perimeter: (x, K)} from the depth-`depth` tomography files."""
    pattern = "tomo_fcc32_L%d_*_n512.json" if sampled else "tomo_fcc32_L%d_*.json"
    rows = []
    for path in sorted(glob.glob(str(TOMO / (pattern % depth)))):
        if not sampled and path.endswith("_n512.json"):
            continue
        with open(path) as fh:
            d = json.load(fh)
        entry = {"jpm": abs(float(d["jpm"]))}
        for L in PERIMS:
            k = d.get("K::%d" % L, {}).get("K")
            entry[L] = k if k else np.nan
        rows.append(entry)
    rows.sort(key=lambda r: r["jpm"])
    x = np.array([r["jpm"] for r in rows])
    return x, {L: np.array([r[L] for r in rows]) for L in PERIMS}


def load_cubic16():
    with open(NORM) as fh:
        recs = json.load(fh)
    out = {}
    for r in recs:
        if r["jpm"] > 0:                      # negative (pi-flux) leg only
            continue
        x = abs(r["jpm"])
        for c in r["classes"]:
            L, scope = int(c["L"]), c["scope"]
            # L=6 is identical masked/unmasked here; L>=8 exists unmasked only
            if L == 6 and scope != "masked":
                continue
            if L != 6 and scope != "unmasked":
                continue
            nan = float("nan")
            out.setdefault(L, []).append((
                x,
                float(c["bw"]) if c["bw"] else nan,
                float(c["herm"]) if c["herm"] else nan,
                float(c["pert"])))
    return {L: np.array(sorted(v), dtype=float) for L, v in out.items()}


def style(ax, xlabel, ylabel, title):
    ax.set_xscale("log")
    ax.set_xlabel(xlabel, fontsize=9.5, color=INK)
    ax.set_ylabel(ylabel, fontsize=9.5, color=INK)
    ax.set_title(title, fontsize=10.5, color=INK, loc="left", pad=8)
    ax.grid(True, which="major", lw=0.6, color=GRID, alpha=0.9)
    ax.grid(True, which="minor", lw=0.4, color=GRID, alpha=0.45)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(labelsize=8.5, colors=INK2, length=3)


def endlabel(ax, x, y, text, dy=1.0):
    """Direct label at the right end of a curve (relief rule: labels required)."""
    good = np.isfinite(y)
    if not good.any():
        return
    ax.annotate(text, (x[good][-1], y[good][-1] * dy),
                textcoords="offset points", xytext=(6, 0), va="center",
                fontsize=8.5, color=INK, fontweight="medium",
                annotation_clip=False)


def main():
    x2, k2 = load_fcc32(2, False)
    x3, k3 = load_fcc32(3, True)
    cub = load_cubic16()

    fig, axes = plt.subplots(2, 3, figsize=(16.6, 8.4))
    fig.patch.set_facecolor("#fcfcfb")
    for ax in axes.ravel():
        ax.set_facecolor("#fcfcfb")

    # ---------------------------------------------------------------- (a)
    ax = axes[0, 0]
    guide = np.logspace(np.log10(0.025), np.log10(0.33), 60)
    for L in PERIMS:
        ax.plot(guide, [k0(L, g) for g in guide], ls=(0, (4, 3)), lw=1.4,
                color=COLOR[L], alpha=0.55, zorder=1)
        ax.plot(x2, k2[L], "-o", lw=2.0, ms=6.5, color=COLOR[L],
                mec="#fcfcfb", mew=1.2, zorder=3)
        endlabel(ax, x2, k2[L], "L=%d" % L)
    ax.set_yscale("log")
    style(ax, r"$|J_\pm| / J_{zz}$", r"$K_L\ /\ J_{zz}$",
          "(a)  FCC-32 loop amplitudes, virtual depth 2")
    ax.text(0.03, 0.06, "dashed: leading perturbation theory",
            transform=ax.transAxes, fontsize=8.2, color=INK2, va="bottom")

    # ---------------------------------------------------------------- (b)
    ax = axes[0, 1]
    for L in PERIMS:
        ax.plot(x2, k2[L] / np.array([k0(L, x) for x in x2]), "-o", lw=2.0,
                ms=6.5, color=COLOR[L], mec="#fcfcfb", mew=1.2, zorder=3)
        if len(x3):
            ax.plot(x3, k3[L] / np.array([k0(L, x) for x in x3]), "o", ms=6.5,
                    mfc="none", mec=COLOR[L], mew=1.6, ls="none", zorder=2)
        endlabel(ax, x2, k2[L] / np.array([k0(L, x) for x in x2]), "L=%d" % L)
    ax.axhline(1.0, lw=1.2, color=INK2, alpha=0.6, zorder=1)
    ax.set_yscale("log")
    style(ax, r"$|J_\pm| / J_{zz}$", r"$K_L\ /\ K_L^{(0)}$",
          "(b)  Suppression below perturbation theory")
    ax.text(0.032, 1.25, "perturbative value", fontsize=8.2, color=INK2)
    ax.text(0.032, 0.012, "filled: depth 2\nopen: depth 3 (512-column sample)",
            fontsize=8.2, color=INK2, va="bottom")

    # ---------------------------------------------------------------- (c)
    ax = axes[0, 2]
    ratio0 = {8: lambda x: (10 / 3) * x, 10: lambda x: (35 / 3) * x ** 2,
              12: lambda x: 42 * x ** 3}
    for L in (8, 10, 12):
        ax.plot(guide, ratio0[L](guide), ls=(0, (4, 3)), lw=1.4,
                color=COLOR[L], alpha=0.55, zorder=1)
        ax.plot(x2, k2[L] / k2[6], "-o", lw=2.0, ms=6.5, color=COLOR[L],
                mec="#fcfcfb", mew=1.2, zorder=3)
        endlabel(ax, x2, k2[L] / k2[6], r"$K_{%d}/K_6$" % L)
    ax.axhline(1.0, lw=1.2, color=INK2, alpha=0.6, zorder=1)
    ax.set_yscale("log")
    ax.set_ylim(2e-4, 4)
    style(ax, r"$|J_\pm| / J_{zz}$", r"$K_L\ /\ K_6$",
          "(c)  Hierarchy relative to the hexagon")
    ax.text(0.032, 1.35, "parity with the hexagon", fontsize=8.2, color=INK2)
    ax.text(0.03, 0.06, "Brillouin-Wigner ratios; see (d) for the\nnormalization dependence of $K_8/K_6$",
            transform=ax.transAxes, fontsize=8.2, color=INK2, va="bottom")

    # ---------------------------------------------------------------- (d)
    ax = axes[1, 0]
    for L in (6, 8):
        if L not in cub:
            continue
        arr = cub[L]
        xs, bw, herm, pert = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]
        ax.plot(xs, bw / pert, ls=(0, (4, 3)), lw=2.0, marker="s", ms=6.0,
                color=COLOR[L], mec="#fcfcfb", mew=1.2, zorder=3)
        good = np.isfinite(herm)
        ax.plot(xs[good], (herm / pert)[good], "-o", lw=2.0, ms=6.5,
                color=COLOR[L], mec="#fcfcfb", mew=1.2, zorder=4)
        endlabel(ax, xs[good], (herm / pert)[good], "L=%d" % L)
    ax.axhline(1.0, lw=1.2, color=INK2, alpha=0.6, zorder=1)
    ax.set_yscale("log")
    style(ax, r"$|J_\pm| / J_{zz}$", r"$K_L\ /\ K_L^{(0)}$",
          "(d)  cubic-16: Hermitian (solid) vs Brillouin-Wigner (dashed)")
    ax.text(0.0055, 0.09,
            "exact fold, no depth truncation.\n"
            "L=8 is an unmasked class amplitude:\n"
            "cubic-16 masks every L>=8 loop.\n"
            "Hermitian pullback fails past -0.20\n"
            "(band no longer isolated).",
            fontsize=8.0, color=INK2, va="bottom")


    # ---------------------------------------------------------------- (e)
    def mu_of(L, x, k):
        return (k / k0(L, x)) ** (1.0 / (L // 2 - 1))

    def mu_fit(x, ks):
        num = sum((L // 2 - 1) * np.log(ks[L] / k0(L, x))
                  for L in PERIMS if np.isfinite(ks[L]) and ks[L] > 0)
        den = sum((L // 2 - 1) ** 2 for L in PERIMS
                  if np.isfinite(ks[L]) and ks[L] > 0)
        return float(np.exp(num / den))

    ax = axes[1, 1]
    for L in PERIMS:
        ax.plot(x2, [mu_of(L, x, k) for x, k in zip(x2, k2[L])], "-o", lw=2.0,
                ms=6.5, color=COLOR[L], mec="#fcfcfb", mew=1.2, zorder=3)
    cub6 = cub.get(6)
    if cub6 is not None:
        ax.plot(cub6[:, 0], np.sqrt(np.abs(cub6[:, 1] / cub6[:, 3])), "s",
                ms=6.0, mfc="none", mec=INK2, mew=1.6, ls="none", zorder=4)
        ax.annotate("cubic-16", (cub6[-1, 0],
                                 np.sqrt(abs(cub6[-1, 1] / cub6[-1, 3]))),
                    textcoords="offset points", xytext=(6, -2), fontsize=8.5,
                    color=INK, annotation_clip=False)
    style(ax, r"$|J_\pm| / J_{zz}$", r"$\mu_L=(K_L/K_L^{(0)})^{1/(L/2-1)}$",
          "(e)  One parameter collapses every perimeter")
    ax.text(0.03, 0.06,
            "all four perimeters on one curve;\n"
            "open squares: 32 -> 16 sites barely moves it\n"
            "(the renormalization is intensive)",
            transform=ax.transAxes, fontsize=8.2, color=INK2, va="bottom")

    # ---------------------------------------------------------------- (f)
    ax = axes[1, 2]
    for L in PERIMS:
        res = []
        for i, x in enumerate(x2):
            mu = mu_fit(x, {p: k2[p][i] for p in PERIMS})
            res.append(k2[L][i] / (k0(L, x) * mu ** (L // 2 - 1)))
        ax.plot(x2, res, "-o", lw=2.0, ms=6.5, color=COLOR[L],
                mec="#fcfcfb", mew=1.2, zorder=3)
        endlabel(ax, x2, np.array(res), "L=%d" % L)
    ax.axhline(1.0, lw=1.2, color=INK2, alpha=0.6, zorder=1)
    style(ax, r"$|J_\pm| / J_{zz}$",
          r"$K_L\ /\ \left[K_L^{(0)}\mu^{L/2-1}\right]$",
          "(f)  Residual: the hexagon is protected")
    ax.text(0.03, 0.06,
            "collapse holds to a few % in mu (the per-step quantity);\n"
            "in the amplitude that is ~2% for L=8,12 and up to 13%\n"
            "low for L=10.  The hexagon is the outlier: +44% at x=0.30",
            transform=ax.transAxes, fontsize=8.2, color=INK2, va="bottom")

    handles = [plt.Line2D([], [], color=COLOR[L], lw=2.4, marker="o", ms=6.5,
                          mec="#fcfcfb", mew=1.2, label="$L = %d$" % L)
               for L in PERIMS]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False,
               fontsize=9.5, bbox_to_anchor=(0.5, 0.947),
               labelcolor=INK, columnspacing=2.4)

    fig.suptitle("Nonperturbative loop amplitudes of quantum spin ice",
                 fontsize=12.5, color=INK, y=0.988, x=0.5)
    fig.tight_layout(rect=(0, 0, 0.985, 0.915))
    for ext in ("png", "pdf"):
        fig.savefig(OUT / ("fig_loop_amplitudes." + ext), dpi=200,
                    facecolor=fig.get_facecolor())
    print("wrote", OUT / "fig_loop_amplitudes.png")

    print("\nFCC-32 depth 2:")
    print("  |Jpm|   " + "  ".join("K%-2d      " % L for L in PERIMS))
    for i, xv in enumerate(x2):
        print("  %.3f  " % xv + "  ".join("%.3e" % k2[L][i] for L in PERIMS))


if __name__ == "__main__":
    main()
