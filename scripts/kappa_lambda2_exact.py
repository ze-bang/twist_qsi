"""EXACT second-order-in-lambda response of the ice band to the uniform
transverse (E_g) source X = sum_i (S+_i + S-_i)  --  the answer to WHY the
measured peak curvature B differs so strongly between pi-flux and 0-flux.

Method (per Jpm):
  1. exact lowest ice band (E_n, psi_n), n = 1..90, of the Sz=0 sector
     (shift-invert Lanczos);
  2. the source connects Sz=0 -> Sz=+-1 (single spinon-pair sectors);
     for every band state solve  (H_{Sz=+1} - E_n) y_n = X_+ psi_n
     (conjugate gradients; the operator is positive definite as long as
     E_n lies below the Sz=+-1 spectrum -- states too close are dropped);
  3. the exact des Cloizeaux second-order band Hamiltonian
        dH2[m,n] = -( <x_m|y_n> + <y_m|x_n> )      (per lambda^2; the
     Sz=-1 sector contributes identically by spin-flip symmetry, factor
     included), so  H_band(lambda) = diag(E) + lambda^2 dH2 + O(lambda^4);
  4. T_peak(lambda) and the small-lambda slope B_exact, fitted exactly as
     the production data (lambda = 0.03, 0.06 through the origin);
  5. DECOMPOSITION of dH2 onto {1, H0, S4, S_hex, residual} in the band
     (Frobenius least squares) and of the slope B into per-component
     contributions;
  6. the spinon-sector minimum E_min(Sz=+1) - E0: the flux-dependent
     denominator scale.

Outputs: gauge_probe_prl/notes/kappa_lambda2_results.json + figN5/figN6.
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
from scipy.sparse import identity
from scipy.sparse.linalg import LinearOperator, cg, eigsh, splu

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ice_pt_lib as ipl  # noqa: E402
import exact_ed_lib as eel  # noqa: E402

HERE = Path(__file__).resolve().parent
GP = HERE.parents[1] / "gauge_probe_prl"
FIGS = GP / "notes" / "figs"
CACHE = GP / "notes" / "cache"
CACHE.mkdir(parents=True, exist_ok=True)

JZZ = 1.0
NB = 90                      # ice band
C_PI = "#1f6feb"
C_ZERO = "#e08e0b"
C_RED = "#d1495b"
C_TEAL = "#2a9d8f"

plt.rcParams.update({
    "font.size": 9, "axes.labelsize": 9.5, "axes.titlesize": 9.5,
    "legend.fontsize": 7.2, "xtick.labelsize": 8, "ytick.labelsize": 8,
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


def band_states(B0, J):
    f = CACHE / f"band_J{J:+.3f}.npz"
    if f.exists():
        d = np.load(f)
        return d["E"], d["Psi"]
    H = B0.H_xxz(J).real.tocsc()
    lu = splu(H - (-6.0) * identity(B0.dim, format="csc"))
    OPinv = LinearOperator((B0.dim, B0.dim), matvec=lu.solve)
    E, Psi = eigsh(H, k=NB + 20, sigma=-6.0, which="LM", OPinv=OPinv)
    o = np.argsort(E)
    E, Psi = E[o], Psi[:, o]
    # re-orthonormalize degenerate clusters (ARPACK slop)
    edges = np.flatnonzero(np.diff(E) > 1e-9) + 1
    for lo, hi in zip(np.r_[0, edges], np.r_[edges, len(E)]):
        if hi - lo > 1:
            Psi[:, lo:hi] = np.linalg.qr(Psi[:, lo:hi])[0]
    np.savez(f, E=E, Psi=Psi)
    return E, Psi


def build_Xplus(B0, B1):
    """X_+ = sum_i S+_i : Sz=0 basis -> Sz=+1 basis (sparse)."""
    from scipy.sparse import coo_matrix
    N = B0.cl.n_sites
    rows, cols = [], []
    one = np.uint64(1)
    for i in range(N):
        bi = one << np.uint64(i)
        m = (B0.states & bi) == 0
        new = B0.states[m] | bi
        rows.append(np.array([B1.index[int(x)] for x in new], dtype=np.int64))
        cols.append(np.nonzero(m)[0])
    r = np.concatenate(rows); c = np.concatenate(cols)
    return coo_matrix((np.ones(len(r)), (r, c)),
                      shape=(B1.dim, B0.dim)).tocsr()


def main():
    t00 = time.time()
    cl = ipl.build_cluster("cubic", (1, 1, 1))
    B0 = eel.SzBasis(cl)                    # Sz = 0
    B1 = eel.SzBasis(cl, nup=cl.n_sites // 2 + 1)   # Sz = +1
    Xp = build_Xplus(B0, B1)
    S4 = B0.ring_sum(cl.loops4)
    Shex = B0.ring_sum(cl.hexes)
    print(f"setup {time.time()-t00:.0f}s: dim0={B0.dim} dim1={B1.dim}")

    jpms = [-0.10, -0.07, -0.05, 0.02, 0.04, 0.05]
    lams_fit = np.array([0.03, 0.06])
    res = {}

    for J in jpms:
        t0 = time.time()
        E, Psi = band_states(B0, J)
        E = E - E[0]
        Eb, Psib = E[:NB], Psi[:, :NB]
        H1 = B1.H_xxz(J).real.tocsr()
        E1min = float(eigsh(H1, k=1, which="SA", return_eigenvectors=False,
                            tol=1e-7)[0])
        E0abs = float(band_states(B0, J)[0][0])
        spinon_gap = E1min - E0abs
        # keep band states safely below the Sz=1 spectrum
        keep = Eb < spinon_gap - 0.05
        nb = int(keep.sum())
        Ebk, Psik = Eb[:nb], Psib[:, :nb]
        print(f"J={J:+.2f}: band {time.time()-t0:.0f}s, spinon gap "
              f"E_min(Sz=1)-E0 = {spinon_gap:.4f}, usable band states {nb}/90")

        # solve (H1 - E0abs - Ebk_n) y = x_n
        X = (Xp @ Psik)                      # (dim1, nb)
        Y = np.empty_like(X)
        t0 = time.time()
        for n in range(nb):
            A = H1 - (E0abs + Ebk[n]) * identity(B1.dim, format="csr")
            y, info = cg(A, X[:, n], rtol=1e-8, maxiter=2000)
            if info != 0:
                print(f"   CG warning n={n}: info={info}")
            Y[:, n] = y
        print(f"   {nb} CG solves in {time.time()-t0:.0f}s")

        XtY = X.T @ Y
        dH2 = -(XtY + XtY.T)                 # both Sz sectors included
        # exact T_peak(lambda) and B
        def Tpk(lam2):
            Ev = np.linalg.eigvalsh(np.diag(Ebk) + lam2 * dH2)
            return refined_peak_T(Ev)
        T0 = Tpk(0.0)
        y = np.array([1.0 - Tpk(l ** 2) / T0 for l in lams_fit])
        x = lams_fit ** 2 / (abs(J) * JZZ)
        B_exact = float(np.sum(y * x) / np.sum(x ** 2))
        # decomposition basis in the truncated band
        S4b = Psik.T @ (S4 @ Psik)
        S6b = Psik.T @ (Shex @ Psik)
        I = np.eye(nb)
        H0b = np.diag(Ebk)
        basis = {"const": I, "rescale": H0b, "S4": np.asarray(S4b),
                 "Shex": np.asarray(S6b)}
        keys = list(basis)
        G = np.array([[np.sum(basis[a] * basis[b]) for b in keys] for a in keys])
        v = np.array([np.sum(basis[a] * dH2) for a in keys])
        coef = np.linalg.solve(G, v)
        fit = sum(c * basis[k] for c, k in zip(coef, keys))
        resid = dH2 - fit
        expl = 1.0 - np.sum(resid ** 2) / np.sum((dH2 - np.mean(np.diag(dH2)) * I) ** 2)
        # slope contribution of each component (and residual)
        def slope_of(M):
            def T(l2):
                Ev = np.linalg.eigvalsh(np.diag(Ebk) + l2 * M)
                return refined_peak_T(Ev)
            yv = np.array([1.0 - T(l ** 2) / T0 for l in lams_fit])
            return float(np.sum(yv * x) / np.sum(x ** 2))
        contrib = {k: slope_of(c * basis[k]) for k, c in zip(keys, coef)}
        contrib["residual"] = slope_of(resid)
        # effective per-channel curvatures implied by the fitted coefficients
        g4 = 4 * J ** 2 / JZZ
        gam6 = 12 * J ** 3 / JZZ ** 2
        B4_eff = float(-coef[keys.index("S4")] * (J * JZZ) / g4)
        B15_eff = float(-coef[keys.index("Shex")] * (J * JZZ) / gam6)
        res[str(J)] = dict(
            spinon_gap=spinon_gap, nb=nb, T0=T0, B_exact=B_exact,
            coef={k: float(c) for k, c in zip(keys, coef)},
            explained=float(expl), contrib=contrib,
            B4_eff=B4_eff, B15_eff=B15_eff,
            diag_dH2_mean=float(np.mean(np.diag(dH2))),
            xx_norm=float(np.mean(np.diag(X.T @ X))),
        )
        print(f"   B_exact={B_exact:+.2f} | coef: " +
              ", ".join(f"{k}={c:+.3f}" for k, c in zip(keys, coef)) +
              f" | var explained {expl:.3f}")
        print(f"   slope contributions: " +
              ", ".join(f"{k}={v:+.2f}" for k, v in contrib.items()))
        print(f"   implied B4_eff={B4_eff:+.2f}  B15_eff={B15_eff:+.2f}")

    with open(GP / "notes" / "kappa_lambda2_results.json", "w") as f:
        json.dump(res, f, indent=1)

    # ---- figN5: exact B + decomposition; figN6: spinon gaps ----------
    meas = {-0.05: 3.20, -0.07: 2.92, -0.10: 2.72,
            0.02: 10.83, 0.04: 12.60, 0.05: 12.85}
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.9, 2.8))
    Js = np.array(jpms)
    Bex = np.array([res[str(J)]["B_exact"] for J in jpms])
    Bms = np.array([meas[J] for J in jpms])
    for m, col, mk, lab in ((Js < 0, C_PI, "o", r"$\pi$-flux"),
                            (Js > 0, C_ZERO, "s", r"$0$-flux")):
        a1.plot(np.abs(Js[m]), Bex[m], mk + "-", color=col, mfc="none",
                label=f"exact $\\lambda^2$ band theory, {lab}")
        a1.plot(np.abs(Js[m]), Bms[m], mk, color=col,
                label=f"measured full ED, {lab}")
    a1.axhline(6, color="0.6", ls=":", lw=1)
    a1.axhline(15, color="0.6", ls="--", lw=1)
    a1.set_xlabel(r"$|J_\pm|/J_{zz}$")
    a1.set_ylabel(r"$|B|$")
    a1.legend(frameon=False, fontsize=6.4)
    a1.set_title("(a) exact second-order slope vs measured")
    # stacked contributions
    comps = ["rescale", "S4", "Shex", "residual"]
    cols = {"rescale": C_TEAL, "S4": C_RED, "Shex": C_PI, "residual": "0.6"}
    xpos = np.arange(len(jpms))
    bottom_p = np.zeros(len(jpms)); bottom_m = np.zeros(len(jpms))
    for c in comps:
        vals = np.array([res[str(J)]["contrib"][c] for J in jpms])
        pos = np.where(vals > 0, vals, 0.0)
        neg = np.where(vals < 0, vals, 0.0)
        a2.bar(xpos, pos, 0.62, bottom=bottom_p, color=cols[c], label=c)
        a2.bar(xpos, neg, 0.62, bottom=bottom_m, color=cols[c])
        bottom_p += pos; bottom_m += neg
    a2.plot(xpos, Bex, "k_", ms=16, label="total (exact)")
    a2.axhline(0, color="k", lw=0.7)
    a2.set_xticks(xpos, [f"{J:+.2f}" for J in jpms])
    a2.set_xlabel(r"$J_\pm/J_{zz}$")
    a2.set_ylabel(r"contribution to $B$")
    a2.legend(frameon=False, fontsize=6.2, ncol=2)
    a2.set_title("(b) decomposition of the slope")
    fig.savefig(FIGS / "figN5_lambda2.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(3.4, 2.7))
    gaps = np.array([res[str(J)]["spinon_gap"] for J in jpms])
    for m, col, mk, lab in ((Js < 0, C_PI, "o", r"$\pi$-flux ($J_\pm<0$)"),
                            (Js > 0, C_ZERO, "s", r"$0$-flux ($J_\pm>0$)")):
        ax.plot(np.abs(Js[m]), gaps[m], mk + "-", color=col, mfc="none", label=lab)
    ax.set_xlabel(r"$|J_\pm|/J_{zz}$")
    ax.set_ylabel(r"$E_{\min}(S^z{=}1)-E_0$  $[J_{zz}]$")
    ax.legend(frameon=False, fontsize=7)
    ax.set_title("virtual spinon-pair gap, flux-resolved", fontsize=9)
    fig.savefig(FIGS / "figN6_spinongap.pdf")
    plt.close(fig)
    print(f"\nDONE {time.time()-t00:.0f}s -> figN5, figN6, kappa_lambda2_results.json")


if __name__ == "__main__":
    main()
