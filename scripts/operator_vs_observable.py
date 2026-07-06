"""Twist averaging: OBSERVABLE average vs OPERATOR average, exactly.

16-site cluster (all exact sparse ED):
  (a) bare C(T) at phi=0,
  (b) corner-averaged OBSERVABLE  Cbar(T) = (1/8) sum_corners C(T, phi),
  (c) corner-averaged OPERATOR: exact Loewdin-downfolded H_eff(phi) at each
      corner, averaged as matrices, then diagonalized,
  (d) zero-net-transport projector (delta=0), two implementations:
        - PT(<=4) row tables with the delta=0 selection,
        - exact counterterm proxy H0 + g4*S4 + gam6*S_hex^wrap
      (the target physics: only contractible hexagons survive).

32-site FCC cluster (PT row tables, order 3): same comparison for (a),(b),
(c'=N-even operator average),(d) -- exact ED is out of reach at 2^32.

Figures -> twist_qsi_demo/paper/figs/, data -> avg_schemes_{16,32}.npz and
avg_schemes_summary.json.
"""
from __future__ import annotations

import json
import sys
import time
from itertools import product
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse.linalg import eigsh

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ice_pt_lib as ipl  # noqa: E402
import exact_ed_lib as eel  # noqa: E402

HERE = Path(__file__).resolve().parent
FIGS = HERE.parents[0] / "paper" / "figs"
FIGS.mkdir(parents=True, exist_ok=True)

JZZ = 1.0
C_BARE = "#d1495b"
C_OBS = "#1f6feb"
C_OP = "#2a9d8f"
C_PROJ = "#6f42c1"

plt.rcParams.update({
    "font.size": 9, "axes.labelsize": 9.5, "axes.titlesize": 9.5,
    "legend.fontsize": 7.2, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "figure.dpi": 200, "savefig.bbox": "tight",
})

TGRID = np.geomspace(2e-4, 0.3, 700)


def corners():
    return [np.array(p, dtype=float) for p in product([0.0, np.pi], repeat=3)]


def gapdeg(E, tol=1e-8):
    E = np.sort(E) - np.min(E)
    # degeneracy = # of levels within tol-cluster of E0; gap = next level
    deg = int(np.sum(E < 1e-6))
    gap = float(E[deg]) if deg < len(E) else float("nan")
    return gap, deg


def run16(Jpm, summary):
    print(f"\n=== 16-site, Jpm={Jpm} (exact) ===")
    cl = ipl.build_cluster("cubic", (1, 1, 1))
    B = eel.SzBasis(cl)
    g4 = 4 * Jpm ** 2 / JZZ
    gam6 = 12 * Jpm ** 3 / JZZ ** 2
    ghex = abs(gam6)

    C_corners, Heffs, corner_info = [], [], []
    for phi in corners():
        t0 = time.time()
        H = B.H_xxz(Jpm, phi)
        Heff, Eband = B.downfold(H)
        E = np.sort(Eband) - np.min(Eband)
        C_corners.append(ipl.C_of_T(E, TGRID))
        Heffs.append(Heff)
        gap, deg = gapdeg(E)
        corner_info.append(dict(phi=(phi / np.pi).tolist(), gap=gap, deg=deg,
                                levels=np.round(E[:12], 6).tolist()))
        print(f"  corner {tuple((phi/np.pi).astype(int))}: gap={gap:.5f} "
              f"deg={deg} [{time.time()-t0:.0f}s]")

    C_bare = C_corners[0]
    C_obs = np.mean(C_corners, axis=0)
    Hbar = np.mean(Heffs, axis=0)
    E_op = np.linalg.eigvalsh(Hbar)
    E_op -= E_op[0]
    C_op = ipl.C_of_T(E_op, TGRID)

    # delta=0 projector via PT(<=4)
    pt = ipl.sw_effective(cl, 1.0, order=4)
    def Hsel(mode):
        M = np.zeros((cl.n_ice, cl.n_ice), dtype=complex)
        for k, tab in (("H2", pt["H2"]), ("H3", pt["H3"]), ("H4", pt["H4"])):
            p = {"H2": 2, "H3": 3, "H4": 4}[k]
            M += (Jpm ** p) * ipl.rows_to_matrix(cl, tab,
                                                 select=ipl.select_rows(cl, tab, mode))
        return M
    E_d0 = np.linalg.eigvalsh(Hsel("delta0")); E_d0 -= E_d0[0]
    C_d0 = ipl.C_of_T(E_d0, TGRID)
    # exact counterterm proxy for the projected model
    S4 = B.ring_sum(cl.loops4)
    Shw = B.ring_sum([(p, w) for p, w in cl.hexes if tuple(w) != (0, 0, 0)])
    E_ct = eigsh(B.H_xxz(Jpm) + g4 * S4 + gam6 * Shw, k=140, which="SA",
                 return_eigenvectors=False, tol=1e-9)
    E_ct = np.sort(E_ct) - np.min(E_ct)
    C_ct = ipl.C_of_T(E_ct, TGRID)

    def pk(C):
        return float(TGRID[np.argmax(C)])
    info = dict(
        Jpm=Jpm, g4=g4, ghex=ghex,
        Tpk_bare=pk(C_bare), Tpk_obs=pk(C_obs), Tpk_op=pk(C_op),
        Tpk_d0=pk(C_d0), Tpk_ct=pk(C_ct),
        gap_op=gapdeg(E_op)[0], deg_op=gapdeg(E_op)[1],
        gap_d0=gapdeg(E_d0)[0], gap_ct=gapdeg(E_ct)[0],
        corners=corner_info,
    )
    print("  T_peaks: bare={Tpk_bare:.5f} obs-avg={Tpk_obs:.5f} "
          "op-avg={Tpk_op:.5f} d0-proj={Tpk_d0:.5f} "
          "counterterm={Tpk_ct:.5f}".format(**info))
    summary[f"16site_{Jpm}"] = info
    np.savez(FIGS / f"avg_schemes_16_jpm{Jpm}.npz",
             T=TGRID, C_corners=np.array(C_corners), C_bare=C_bare,
             C_obs=C_obs, C_op=C_op, C_d0=C_d0, C_ct=C_ct,
             E_op=E_op, E_d0=E_d0, E_ct=E_ct, g4=g4, ghex=ghex)
    return dict(C_corners=C_corners, C_bare=C_bare, C_obs=C_obs, C_op=C_op,
                C_d0=C_d0, C_ct=C_ct, g4=g4, ghex=ghex)


def run32(Jpm, summary, order=3):
    print(f"\n=== 32-site FCC, Jpm={Jpm} (PT order<={order}) ===")
    cl = ipl.build_cluster("fcc", (2, 2, 2))
    t0 = time.time()
    pt = ipl.sw_effective(cl, 1.0, order=order)
    print(f"  PT run: {time.time()-t0:.0f}s "
          f"(H2 {len(pt['H2']['c'])} rows, H3 {len(pt['H3']['c'])} rows)")
    tabs = [("H2", 2), ("H3", 3)] + ([("H4", 4)] if order >= 4 else [])

    def Hof(phi=None, mode=None):
        M = np.zeros((cl.n_ice, cl.n_ice), dtype=complex)
        for k, p in tabs:
            sel = ipl.select_rows(cl, pt[k], mode) if mode else None
            M += (Jpm ** p) * ipl.rows_to_matrix(cl, pt[k], phi=phi, select=sel)
        return M

    g4 = 4 * Jpm ** 2 / JZZ
    ghex = 12 * abs(Jpm) ** 3 / JZZ ** 2
    C_corners, Heffs, corner_info = [], [], []
    for phi in corners():
        t0 = time.time()
        H = Hof(phi=phi)
        E = np.linalg.eigvalsh(H)
        Heffs.append(H)
        E -= E[0]
        C_corners.append(ipl.C_of_T(E, TGRID))
        gap, deg = gapdeg(E)
        corner_info.append(dict(phi=(phi / np.pi).tolist(), gap=gap, deg=deg,
                                levels=np.round(E[:12], 6).tolist()))
        print(f"  corner {tuple((phi/np.pi).astype(int))}: gap={gap:.6f} "
              f"deg={deg} [{time.time()-t0:.0f}s]")

    C_bare = C_corners[0]
    C_obs = np.mean(C_corners, axis=0)
    E_op = np.linalg.eigvalsh(np.mean(Heffs, axis=0)); E_op -= E_op[0]
    C_op = ipl.C_of_T(E_op, TGRID)
    E_d0 = np.linalg.eigvalsh(Hof(mode="delta0")); E_d0 -= E_d0[0]
    C_d0 = ipl.C_of_T(E_d0, TGRID)

    def pk(C):
        return float(TGRID[np.argmax(C)])
    info = dict(Jpm=Jpm, g4=g4, ghex=ghex, order=order,
                Tpk_bare=pk(C_bare), Tpk_obs=pk(C_obs), Tpk_op=pk(C_op),
                Tpk_d0=pk(C_d0), gap_d0=gapdeg(E_d0)[0],
                corners=corner_info)
    print("  T_peaks: bare={Tpk_bare:.5f} obs-avg={Tpk_obs:.5f} "
          "op-avg={Tpk_op:.5f} d0-proj={Tpk_d0:.5f}".format(**info))
    summary[f"32site_{Jpm}"] = info
    np.savez(FIGS / f"avg_schemes_32_jpm{Jpm}.npz",
             T=TGRID, C_corners=np.array(C_corners), C_bare=C_bare,
             C_obs=C_obs, C_op=C_op, C_d0=C_d0, g4=g4, ghex=ghex)
    return dict(C_corners=C_corners, C_bare=C_bare, C_obs=C_obs, C_op=C_op,
                C_d0=C_d0, g4=g4, ghex=ghex)


def hero_figure(d16, d32, Jpm16, Jpm32, fname):
    fig, axs = plt.subplots(1, 2, figsize=(6.9, 2.9))
    for ax, d, N, J, extra in ((axs[0], d16, 16, Jpm16, True),
                               (axs[1], d32, 32, Jpm32, False)):
        for Cc in d["C_corners"]:
            ax.plot(TGRID, Cc, color="0.8", lw=0.5, zorder=1)
        ax.plot(TGRID, d["C_bare"], color=C_BARE, lw=1.6,
                label=r"bare ($\bvarphi=0$)" if False else
                r"bare $\varphi=0$", zorder=3)
        ax.plot(TGRID, d["C_obs"], color=C_OBS, lw=1.6,
                label=r"observable avg $\overline{C}$", zorder=4)
        ax.plot(TGRID, d["C_op"], color=C_OP, lw=1.6,
                label=r"operator avg $C[\overline{H}_{\rm eff}]$", zorder=5)
        ax.plot(TGRID, d["C_d0"], color=C_PROJ, lw=1.6, ls="--",
                label=r"transport projector $\delta=0$", zorder=6)
        if extra:
            ax.plot(TGRID, d["C_ct"], color="k", lw=1.0, ls=":",
                    label="counterterm reference", zorder=6)
        ax.axvline(d["g4"], color=C_BARE, ls=":", lw=1, alpha=0.7)
        ax.text(d["g4"] * 1.05, ax.get_ylim()[1] * 0.02, r"$g_4$",
                color=C_BARE, fontsize=7)
        ax.axvline(d["ghex"], color=C_OP, ls=":", lw=1, alpha=0.7)
        ax.text(d["ghex"] * 1.05, ax.get_ylim()[1] * 0.02, r"$g_{\rm hex}$",
                color=C_OP, fontsize=7)
        ax.set_xscale("log")
        ax.set_xlabel(r"$T/J_{zz}$")
        ax.set_title(f"({'ab'[N==32]}) {N}-site"
                     f"{' cubic (exact)' if N==16 else ' FCC (PT)'}"
                     f", $J_\\pm={J}$")
        ax.set_xlim(3e-4, 0.3)
    axs[0].set_ylabel(r"$C(T)$ (ice band)")
    axs[0].legend(frameon=False, loc="upper left")
    fig.savefig(FIGS / fname)
    plt.close(fig)


def main():
    summary = {}
    d16a = run16(-0.10, summary)
    d16b = run16(-0.05, summary)
    d32a = run32(-0.05, summary)
    d32b = run32(-0.10, summary)
    hero_figure(d16a, d32b, -0.10, -0.10, "fig_avg_schemes.pdf")
    hero_figure(d16b, d32a, -0.05, -0.05, "fig_avg_schemes_jpm005.pdf")
    with open(FIGS / "avg_schemes_summary.json", "w") as f:
        json.dump(summary, f, indent=1)
    print("\nDONE. Figures + JSON in", FIGS)


if __name__ == "__main__":
    main()
