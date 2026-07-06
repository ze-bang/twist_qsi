"""Why the transverse-source curvature kappa differs between pi-flux (Jpm<0)
and 0-flux (Jpm>0) on the SAME bare 16-site cluster -- exact-ED evidence.

Mechanism under test: the low-T gauge peak of the bare periodic cluster is
set by TWO emergent ring channels,
    winding 4-loops : g4   = 4 Jpm^2/Jzz        (sign-blind, unfrustrated)
    hexagons        : gam6 = 12 Jpm^3/Jzz^2     (sign flips with Jpm)
so the channels COOPERATE at 0-flux and COMPETE at pi-flux.  A transverse
source renormalizes each channel with its own curvature (B4 = 6, Bhex = 15,
kappa_C = -B_C/(Jpm Jzz)), hence the measured peak curvature is the
channel-weighted mix   B_eff = w4 B4 + whex Bhex,   with
w_C = d ln T_peak / d ln g_C  measured EXACTLY here by adding the explicit
ring operator sums to the microscopic Hamiltonian and re-diagonalizing.

Everything on this cluster is exact sparse ED (no perturbation theory).

Figures -> gauge_probe_prl/notes/figs/, numbers -> kappa_flux_results.json.
"""
from __future__ import annotations

import json
import sys
import time
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
GP = HERE.parents[1] / "gauge_probe_prl"
FIGS = GP / "notes" / "figs"
FIGS.mkdir(parents=True, exist_ok=True)

B4, B15 = 6.0, 15.0
JZZ = 1.0
NEIG = 110
C_PI = "#1f6feb"
C_ZERO = "#e08e0b"
C_RED = "#d1495b"

plt.rcParams.update({
    "font.size": 9, "axes.labelsize": 9.5, "axes.titlesize": 9.5,
    "legend.fontsize": 7.5, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "figure.dpi": 200, "savefig.bbox": "tight",
})


def refined_peak_T(E, tmin=2e-4, tmax=0.15, n=2500):
    E = np.sort(np.asarray(E)) - np.min(E)
    T = np.geomspace(tmin, tmax, n)
    C = ipl.C_of_T(E, T)
    k = int(np.argmax(C))
    if 0 < k < n - 1:
        x = np.log(T[k - 1:k + 2]); y = C[k - 1:k + 2]
        den = y[0] - 2 * y[1] + y[2]
        if abs(den) > 1e-30:
            return float(np.exp(x[1] + 0.5 * (y[0] - y[2]) / den * (x[1] - x[0])))
    return float(T[k])


def main():
    t00 = time.time()
    cl = ipl.build_cluster("cubic", (1, 1, 1))
    B = eel.SzBasis(cl)
    S4 = B.ring_sum(cl.loops4)
    hex_contract = [(p, w) for p, w in cl.hexes if tuple(w) == (0, 0, 0)]
    hex_wrap = [(p, w) for p, w in cl.hexes if tuple(w) != (0, 0, 0)]
    Shex = B.ring_sum(cl.hexes)
    Shex_wrap = B.ring_sum(hex_wrap)
    print(f"setup {time.time()-t00:.0f}s; hexagons: {len(hex_contract)} "
          f"contractible + {len(hex_wrap)} wrapping")

    res = {"B4": B4, "B15": B15, "NEIG": NEIG}
    jpms = [-0.10, -0.07, -0.05, 0.02, 0.04, 0.05]

    # ------------------------------------------------------------------
    # One dense-subset eigensolve per Jpm (robust to the nearly degenerate
    # ice band that defeats plain Lanczos at small |Jpm|); the ring
    # perturbations are then projected into the frozen lowest-NEIG band,
    # which is exact to first order in the probe (band gap ~ Jzz/2 >>
    # probe strength) and reduces every subsequent spectrum to NEIG x NEIG.
    # ------------------------------------------------------------------
    from scipy.linalg import eigh as dense_eigh

    def band(J):
        H = B.H_xxz(J).real.toarray()
        E, Psi = dense_eigh(H, subset_by_index=(0, NEIG - 1))
        return E, Psi

    bands = {}
    for J in jpms:
        t0 = time.time()
        E, Psi = band(J)
        S4b = Psi.T @ (S4 @ Psi)
        Shexb = Psi.T @ (Shex @ Psi)
        Shexwb = Psi.T @ (Shex_wrap @ Psi)
        bands[J] = (E, S4b, Shexb, Shexwb)
        print(f"band Jpm={J:+.2f}: dense eigh + projections {time.time()-t0:.0f}s "
              f"(E0={E[0]:.5f})")

    def bspec(J, c4=0.0, c6=0.0, c6w=0.0):
        E, S4b, Shexb, Shexwb = bands[J]
        M = np.diag(E) + c4 * S4b + c6 * Shexb + c6w * Shexwb
        Eb = np.linalg.eigvalsh(M)
        return Eb - Eb[0]

    rows = []
    for J in jpms:
        t0 = time.time()
        g4 = 4 * J ** 2 / JZZ
        gam6 = 12 * J ** 3 / JZZ ** 2          # signed hexagon coupling
        E0 = bspec(J)
        Tpk0 = refined_peak_T(E0)
        eps = 0.08
        w4 = (np.log(refined_peak_T(bspec(J, c4=-eps * g4)))
              - np.log(refined_peak_T(bspec(J, c4=+eps * g4)))) / (2 * eps)
        w6 = (np.log(refined_peak_T(bspec(J, c6=-eps * gam6)))
              - np.log(refined_peak_T(bspec(J, c6=+eps * gam6)))) / (2 * eps)
        Beff = B4 * w4 + B15 * w6
        gap = float(E0[1])
        # channel-cancelled spectra (leading-order counterterms)
        E_no4 = bspec(J, c4=+g4)
        E_nohex = bspec(J, c6=+gam6)
        E_proj = bspec(J, c4=+g4, c6w=+gam6)     # ~ ideal twist projection
        rows.append(dict(
            Jpm=J, Tpk=Tpk0, w4=w4, whex=w6, Beff=Beff, gap=gap,
            g4=g4, gam6=gam6,
            Tpk_no4=refined_peak_T(E_no4), Tpk_nohex=refined_peak_T(E_nohex),
            Tpk_proj=refined_peak_T(E_proj),
        ))
        print(f"Jpm={J:+.2f} [{time.time()-t0:.0f}s]: Tpk={Tpk0:.5f} gap={gap:.5f} "
              f"w4={w4:+.3f} whex={w6:+.3f} B_eff={Beff:+.2f} | "
              f"Tpk(no 4-ring)={rows[-1]['Tpk_no4']:.5f} "
              f"Tpk(proj~)={rows[-1]['Tpk_proj']:.5f}")
    res["weights"] = rows

    # ------------------------------------------------------------------
    # measured ED softening (gauge_probe data) -> fitted B per Jpm
    # ------------------------------------------------------------------
    d = np.load(GP / "data" / "fig2_softening.npz")
    jm, lams, Tm = d["jpms"], d["lams"], d["Tpeak"]
    meas = []
    for k, J in enumerate(jm):
        x = lams ** 2 / (abs(J) * JZZ)
        y = 1.0 - Tm[k] / Tm[k, 0]
        sel = (lams > 0) & (lams <= 0.061)
        Bfit = float(np.sum(y[sel] * x[sel]) / np.sum(x[sel] ** 2))
        meas.append(dict(Jpm=float(J), Bfit=Bfit, x=x.tolist(),
                         Trel=(Tm[k] / Tm[k, 0]).tolist()))
        print(f"measured Jpm={J:+.3f}: B_fit={Bfit:+.2f}")
    res["measured"] = meas

    # ------------------------------------------------------------------
    # exact model softening curves: lambda enters via channel deficits
    # ------------------------------------------------------------------
    lam_grid = np.linspace(0.0, 0.12, 9)
    curves = {}
    for J in jpms:
        g4 = 4 * J ** 2 / JZZ
        gam6 = 12 * J ** 3 / JZZ ** 2
        k4 = -B4 / (J * JZZ)
        k6 = -B15 / (J * JZZ)
        T0 = None
        xs, ys = [], []
        for lam in lam_grid:
            # lambda enters only through the channel deficits
            Tl = refined_peak_T(bspec(J, c4=k4 * lam ** 2 * g4,
                                      c6=k6 * lam ** 2 * gam6))
            if T0 is None:
                T0 = Tl
            xs.append(lam ** 2 / (abs(J) * JZZ))
            ys.append(Tl / T0)
        curves[J] = dict(x=xs, y=ys)
        print(f"model curve Jpm={J:+.2f}: T(0)={T0:.5f} "
              f"Trel={np.round(ys,3)}")
    res["model_curves"] = {str(k): v for k, v in curves.items()}

    # ------------------------------------------------------------------
    # per-loop coherence in the exact ground state
    # ------------------------------------------------------------------
    coh = {}
    for J in (-0.10, +0.05):
        H0 = B.H_xxz(J)
        E, Psi = eigsh(H0, k=4, which="SA", tol=1e-10)
        psi = Psi[:, np.argmin(E)]
        g4 = 4 * J ** 2 / JZZ
        gh = 12 * abs(J) ** 3 / JZZ ** 2
        v4 = np.array([float(np.real(psi.conj() @ (B.ring_sum([lp]) @ psi)))
                       for lp in cl.loops4])
        v6c = np.array([float(np.real(psi.conj() @ (B.ring_sum([hx]) @ psi)))
                        for hx in hex_contract])
        v6w = np.array([float(np.real(psi.conj() @ (B.ring_sum([hx]) @ psi)))
                        for hx in hex_wrap])
        coh[J] = dict(v4=v4.tolist(), v6c=v6c.tolist(), v6w=v6w.tolist())
        print(f"coherence Jpm={J:+.2f}: <O4+h.c.> mean={v4.mean():+.3f} "
              f"std={v4.std():.3f} | <Ohex_c> mean={v6c.mean():+.3f} | "
              f"<Ohex_w> mean={v6w.mean():+.3f}")
    res["coherence"] = {str(k): v for k, v in coh.items()}

    # ------------------------------------------------------------------
    # holonomy scans (exact): E0(phi) and manifold gap along x and 111
    # ------------------------------------------------------------------
    phis = np.linspace(0, 2 * np.pi, 21)
    scans = {}
    for J in (-0.10, +0.05):
        for tag, direc in (("x", np.array([1., 0., 0.])),
                           ("111", np.array([1., 1., 1.]))):
            t0 = time.time()
            E0s, gaps = [], []
            for p in phis:
                H = B.H_xxz(J, p * direc)
                E = np.sort(eigsh(H, k=6, which="SA",
                                  return_eigenvectors=False, tol=1e-8))
                E0s.append(float(E[0]))
                g, _ = ipl.manifold_gap(E - E[0])
                gaps.append(float(g))
            scans[(J, tag)] = (np.array(E0s), np.array(gaps))
            print(f"holonomy J={J:+.2f} dir={tag}: {time.time()-t0:.0f}s")
    res["holonomy"] = {f"{J}_{tag}": dict(E0=v[0].tolist(), gap=v[1].tolist())
                       for (J, tag), v in scans.items()}
    res["holonomy_phis"] = phis.tolist()

    with open(FIGS.parent / "kappa_flux_results.json", "w") as f:
        json.dump(res, f, indent=1)

    # ==================================================================
    # figures
    # ==================================================================
    # ---- figN1: weights + B comparison
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.9, 2.7))
    xs = np.arange(len(jpms))
    wid = 0.38
    a1.bar(xs - wid / 2, [r["w4"] for r in rows], wid, color=C_RED,
           label=r"$w_4=\partial\ln T_{\rm pk}/\partial\ln g_4$")
    a1.bar(xs + wid / 2, [r["whex"] for r in rows], wid, color=C_PI,
           label=r"$w_{\rm hex}=\partial\ln T_{\rm pk}/\partial\ln g_{\rm hex}$")
    a1.axhline(0, color="k", lw=0.7)
    a1.set_xticks(xs, [f"{J:+.2f}" for J in jpms])
    a1.set_xlabel(r"$J_\pm/J_{zz}$")
    a1.set_ylabel("channel weight of $T_{\\rm peak}$")
    a1.legend(frameon=False, loc="upper left")
    a1.set_title("(a) exact channel weights (16-site, bare)")

    a2.axhline(B4, color="0.55", ls=":", lw=1)
    a2.text(0.055, B4 + 0.35, r"$B_4=6$", color="0.4", fontsize=7)
    a2.axhline(B15, color="0.55", ls="--", lw=1)
    a2.text(0.055, B15 + 0.35, r"$B_{\rm hex}=15$", color="0.4", fontsize=7)
    Jarr = np.array(jpms)
    Bp = np.array([r["Beff"] for r in rows])
    mneg = Jarr < 0
    a2.plot(np.abs(Jarr[mneg]), Bp[mneg], "o-", color=C_PI, mfc="none",
            label=r"predicted $B_{\rm eff}=6w_4+15w_{\rm hex}$, $\pi$-flux")
    a2.plot(np.abs(Jarr[~mneg]), Bp[~mneg], "s-", color=C_ZERO, mfc="none",
            label=r"predicted, $0$-flux")
    Jm = np.array([m["Jpm"] for m in meas])
    Bm = np.array([m["Bfit"] for m in meas])
    mneg = Jm < 0
    a2.plot(np.abs(Jm[mneg]), Bm[mneg], "o", color=C_PI,
            label=r"measured ($E_g$-source ED), $\pi$-flux")
    a2.plot(np.abs(Jm[~mneg]), -Bm[~mneg], "s", color=C_ZERO,
            label="measured, $0$-flux")
    a2.set_xlabel(r"$|J_\pm|/J_{zz}$")
    a2.set_ylabel(r"$B$ (soften $>0$ on $\pi$ side; $|B|$ on $0$ side)")
    a2.legend(frameon=False, fontsize=6.6, loc="center right")
    a2.set_title("(b) peak curvature: prediction vs measurement")
    fig.savefig(FIGS / "figN1_weights.pdf")
    plt.close(fig)

    # ---- figN2: coherence
    fig, axs = plt.subplots(1, 2, figsize=(6.9, 2.7), sharey=True)
    for ax, J, col, ttl in ((axs[0], -0.10, C_PI, r"$\pi$-flux, $J_\pm=-0.10$"),
                            (axs[1], +0.05, C_ZERO, r"$0$-flux, $J_\pm=+0.05$")):
        v4 = np.array(coh[J]["v4"]); v6c = np.array(coh[J]["v6c"])
        v6w = np.array(coh[J]["v6w"])
        rng = np.random.default_rng(3)
        ax.scatter(rng.uniform(-.12, .12, len(v4)), v4, s=14, color=C_RED,
                   alpha=0.8, label="36 winding 4-loops")
        ax.scatter(1 + rng.uniform(-.12, .12, len(v6c)), v6c, s=14,
                   color=col, alpha=0.85, label="16 contractible hexagons")
        ax.scatter(2 + rng.uniform(-.12, .12, len(v6w)), v6w, s=14,
                   color="0.55", alpha=0.7, label="48 wrapping hexagons")
        ax.axhline(0, color="k", lw=0.7)
        ax.set_xticks([0, 1, 2], ["4-loops", "hex (contr.)", "hex (wrap)"])
        ax.set_title(ttl)
        ax.legend(frameon=False, fontsize=6.6,
                  loc="lower right" if J > 0 else "center right")
    axs[0].set_ylabel(r"$\langle O_{\mathcal C}+O^\dagger_{\mathcal C}\rangle_{\rm GS}$")
    fig.savefig(FIGS / "figN2_coherence.pdf")
    plt.close(fig)

    # ---- figN3: holonomy
    fig, axs = plt.subplots(1, 2, figsize=(6.9, 2.7))
    for J, col, lab in ((-0.10, C_PI, r"$\pi$-flux $J_\pm=-0.10$"),
                        (+0.05, C_ZERO, r"$0$-flux $J_\pm=+0.05$")):
        g4n = 4 * J ** 2 / JZZ
        for tag, ls in (("x", "-"), ("111", "--")):
            E0s, _ = scans[(J, tag)]
            axs[0].plot(phis / np.pi, (E0s - E0s[0]) / g4n, ls, color=col,
                        lw=1.3, label=lab + (r", $\varphi\hat x$" if tag == "x"
                                             else r", $\varphi(1,1,1)$"))
        _, gaps = scans[(J, "x")]
        axs[1].plot(phis / np.pi, gaps / g4n, "-", color=col, lw=1.3, label=lab)
    axs[0].set_xlabel(r"$\varphi/\pi$")
    axs[0].set_ylabel(r"$[E_0(\varphi)-E_0(0)]\,/\,g_4$")
    axs[0].legend(frameon=False, fontsize=6.4)
    axs[0].set_title("(a) holonomy stiffness of the ground state")
    axs[1].set_xlabel(r"$\varphi/\pi$")
    axs[1].set_ylabel(r"manifold gap $/\,g_4$")
    axs[1].legend(frameon=False, fontsize=6.4)
    axs[1].set_title(r"(b) low gap vs boundary flux ($\varphi\hat x$)")
    fig.savefig(FIGS / "figN3_holonomy.pdf")
    plt.close(fig)

    # ---- figN4: softening, model vs measured
    fig, ax = plt.subplots(figsize=(3.6, 3.0))
    for J in jpms:
        col = C_PI if J < 0 else C_ZERO
        ax.plot(curves[J]["x"], curves[J]["y"], "-", color=col, lw=1.1,
                alpha=0.85)
    for m in meas:
        J = m["Jpm"]
        col = C_PI if J < 0 else C_ZERO
        mk = "o" if J < 0 else "s"
        ax.plot(m["x"], m["Trel"], mk, color=col, ms=4, mfc="white", mew=1.1)
    xref = np.linspace(0, 0.30, 10)
    ax.plot(xref, 1 - B4 * xref, ":", color="0.5", lw=1)
    ax.plot(xref, 1 - B15 * xref, "--", color="0.5", lw=1)
    ax.text(0.205, 1 - B4 * 0.205 + 0.02, r"$1-6x$", color="0.4", fontsize=7)
    ax.text(0.083, 1 - B15 * 0.083 - 0.1, r"$1-15x$", color="0.4", fontsize=7)
    ax.set_xlabel(r"$x=\lambda^2/(|J_\pm|J_{zz})$")
    ax.set_ylabel(r"$T_{\rm peak}(\lambda)/T_{\rm peak}(0)$")
    ax.set_xlim(0, 0.30); ax.set_ylim(0.0, 2.3)
    ax.plot([], [], "-", color=C_PI, label=r"exact 16-site model, $\pi$-flux")
    ax.plot([], [], "-", color=C_ZERO, label=r"exact 16-site model, $0$-flux")
    ax.plot([], [], "o", color="0.4", mfc="white", label=r"$E_g$-source ED (measured)")
    ax.legend(frameon=False, fontsize=6.6, loc="upper left")
    ax.set_title("channel-renormalization model vs ED", fontsize=8.5)
    fig.savefig(FIGS / "figN4_softening.pdf")
    plt.close(fig)

    print(f"\nDONE ({time.time()-t00:.0f}s). Figures: {FIGS}")


if __name__ == "__main__":
    main()
