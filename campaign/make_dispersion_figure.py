"""THE dispersion figure: S^zz(q, omega) along a continuous momentum
path, every exact pole marked, colored by excitation type.

Path: Gamma --Lambda-- L --soft valley-- (0,1/2,1/2) ---- X --Delta--
Gamma.  Every exact-diagonalization momentum (48, 64, 72, 96 spins)
that lies ON the path is placed at its true position; marker area is
the level's share of S(q) at its momentum.  Colors follow the paper's
system: red = the soft (roton-like) level, indigo = the two dominant
transverse-channel levels (the operational photon descendants), teal =
everything else (residual / multi-photon weight).  The dashed amber
curve is the Gaussian photon evaluated analytically on the infinite
lattice along the same path (two transverse branches; single fitted
scale sqrt(2UK) = 0.5868K), so the roton's position at roughly half
the free-theory energy is visible at a glance.

GATE printed at runtime: the analytic template curl reproduces the
48-site cluster singular value at L (2*sqrt(3)) to machine precision.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif": ["cmr10", "Computer Modern Roman", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "axes.unicode_minus": False,
})

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "notes"))
sys.path.insert(0, str(ROOT / "campaign"))
from pyro_lattice import extract_templates, FCC_A, SUBLAT       # noqa: E402

D = ROOT / "campaign" / "outputs" / "ring_model_dssf"
OUTDIR = ROOT / "paper_multiphoton" / "figs"
INK, INK2, EDGE = "#0f172a", "#475569", "#cbd5e1"
PHOTON, ROTON, MULTI, GAUSS = "#4f46e5", "#e11d48", "#0d9488", "#d97706"
SCALE = 0.58684          # sqrt(2UK), the single fitted Gaussian scale

# ------------------------------------------------------------------ path
NODES = [np.array(v, float) for v in
         [(0, 0, 0), (0.5, 0.5, 0.5), (0, 0.5, 0.5), (0, 0, 1), (0, 0, 0)]]
NODE_LABELS = [r"$\Gamma$", "$L$", r"$(0,\frac{1}{2},\frac{1}{2})$",
               "$X$", r"$\Gamma$"]


def path_x(q):
    """Position of q on the path, or None if not on it (up to the cubic
    symmetries that map the cluster momentum onto the path leg)."""
    a = np.sort(np.abs(np.asarray(q, float)))
    x0 = 0.0
    for i in range(len(NODES) - 1):
        p, r = NODES[i], NODES[i + 1]
        seg = np.linalg.norm(r - p)
        for t in np.linspace(0, 1, 2001):
            v = np.sort(np.abs(p + t * (r - p)))
            if np.allclose(v, a, atol=2e-3):
                return x0 + t * seg
        x0 += seg
    return None


def gaussian_branches():
    """sigma_{1,2}(q) from the analytic template curl, any q."""
    htem, _ = extract_templates()
    rows = []
    for tem in htem:
        pos = np.array([np.array(e[:3], float) @ FCC_A + SUBLAT[e[3]]
                        for e in tem])
        sub = [e[3] for e in tem]
        signs = np.array([1.0 if j % 2 == 0 else -1.0
                          for j in range(len(tem))])
        rows.append((pos, sub, signs))

    def sig(q):
        G = np.zeros((len(rows), 4), dtype=complex)
        for r, (pos, sub, signs) in enumerate(rows):
            ph = signs * np.exp(-2j * np.pi * (pos @ q))
            for j, mu in enumerate(sub):
                G[r, mu] += ph[j]
        s = np.linalg.svd(G, compute_uv=False)
        return s[:2]

    # GATE: the 48-site cluster gives sigma(L) = 2 sqrt(3)
    g = sig(np.array([0.5, 0.5, -0.5]))
    print(f"GATE  analytic sigma(L) = {g[0]:.10f}, {g[1]:.10f}   "
          f"(2 sqrt3 = {2*np.sqrt(3):.10f})")
    xs, b1, b2 = [], [], []
    x0 = 0.0
    for i in range(len(NODES) - 1):
        p, r = NODES[i], NODES[i + 1]
        seg = np.linalg.norm(r - p)
        for t in np.linspace(0, 1, 160):
            s = sig(p + t * (r - p))
            xs.append(x0 + t * seg)
            b1.append(SCALE * s[0])
            b2.append(SCALE * s[1])
        x0 += seg
    return np.array(xs), np.array(b1), np.array(b2)


# ------------------------------------------------------------- ED poles
def load_levels():
    """(x, omega, share, kind) for every on-path exact level."""
    out = []

    def add(q, S, lines, soft_expected):
        if S < 0.05:
            return          # Gamma: the uniform mode is the ground state
        x = path_x(q)
        if x is None:
            return
        L = sorted([(e, w) for e, w, *_ in lines if w / S > 4e-3],
                   key=lambda p: -p[1])
        for rank, (e, w) in enumerate(L):
            if soft_expected and e < 1.4 and rank < 2:
                kind = "roton"
            elif rank < 2:
                kind = "photon"
            else:
                kind = "multi"
            out.append((x, e, w / S, kind))

    # 48 sites: complete diagonalization (polarization file has all lines)
    p48 = json.load(open(D / "ring48_polarization_all.json"))
    for k, m in p48.items():
        if not isinstance(m, dict) or "lines" not in m:
            continue
        soft = m["star"] == "L"
        add(m["q"], m["S"], [(r["omega"], r["w"]) for r in m["lines"]],
            soft)
    # 64 and 72: Lanczos, all momenta
    for fn in ("ring_allq_fcc224.json", "ring_allq_fcc233.json"):
        d = json.load(open(D / fn))
        seen = set()
        for m in d["momenta"]:
            key = (m["star"], round(m["S"], 6))
            if key in seen:
                continue
            seen.add(key)
            soft = m["star"] in ("L", "?[0.167, 0.5, 0.5]") \
                and m["S"] > 0.3
            add(m["q"], m["S"], m["lines"], soft)
    # 96: the two L classes
    d = json.load(open(D / "ring96_fcc234.json"))
    for m in d["momenta"]:
        add(m["q"], m["S"], m["lines"], m["S"] > 0.3)
    return out


def main():
    xs, b1, b2 = gaussian_branches()
    levels = load_levels()
    print(f"{len(levels)} on-path levels "
          f"({sum(1 for l in levels if l[3]=='roton')} roton, "
          f"{sum(1 for l in levels if l[3]=='photon')} photon, "
          f"{sum(1 for l in levels if l[3]=='multi')} multi)")

    fig, ax = plt.subplots(figsize=(7.0, 3.1))
    fig.patch.set_facecolor("white")
    ax.plot(xs, b1, "--", color=GAUSS, lw=1.3, zorder=2)
    ax.plot(xs, b2, "--", color=GAUSS, lw=1.0, alpha=0.7, zorder=2)
    ax.annotate("Gaussian photon", xy=(2.47, 2.66), fontsize=8.4,
                color=GAUSS)

    order = {"multi": 3, "photon": 4, "roton": 5}
    cols = {"multi": MULTI, "photon": PHOTON, "roton": ROTON}
    for x, e, sh, kind in levels:
        ax.scatter(x, e, s=12 + 600 * sh, color=cols[kind],
                   edgecolor="white", lw=0.5, zorder=order[kind],
                   alpha=0.92)
    for kind, lab in (("roton", "soft (roton-like) mode"),
                      ("photon", "transverse-channel doublet"),
                      ("multi", "residual weight")):
        ax.scatter([], [], s=70, color=cols[kind], label=lab)
    ax.legend(loc="upper right", fontsize=7.8, frameon=False,
              borderaxespad=0.3, handletextpad=0.25)

    x0, ticks = 0.0, [0.0]
    for i in range(len(NODES) - 1):
        x0 += np.linalg.norm(NODES[i + 1] - NODES[i])
        ticks.append(x0)
    for t in ticks[1:-1]:
        ax.axvline(t, color=EDGE, lw=0.7, zorder=1)
    ax.set_xticks(ticks)
    ax.set_xticklabels(NODE_LABELS, fontsize=9.5)
    ax.set_ylabel(r"$\omega/K$", fontsize=10, color=INK)
    ax.set_xlim(ticks[0], ticks[-1])
    ax.set_ylim(0, 4.6)
    ax.tick_params(labelsize=8.5, colors=INK2, length=3)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(EDGE)
    fig.subplots_adjust(left=0.065, right=0.985, top=0.965, bottom=0.10)
    for ext in ("pdf", "png"):
        fig.savefig(OUTDIR / f"fig_dispersion.{ext}", dpi=300)
    print("wrote", OUTDIR / "fig_dispersion.pdf")


if __name__ == "__main__":
    main()
