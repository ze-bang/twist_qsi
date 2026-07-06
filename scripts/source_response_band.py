"""Ice-band response for THREE source form factors (uniform / staggered /
[111]-type), with the full detailed analysis at each coupling:

  * exact second-order band matrix dH (resolvents into Sz=+1 AND Sz=-1,
    needed separately for complex/staggered patterns),
  * decomposition {const, rescale, S4, Shex, residual} + slope contributions,
  * B(lambda) sweep: T_peak(lambda)/T_peak(0) for lambda up to 0.12,
  * transport-projected B (dipole mask on both H_eff and dH): the part of
    the response that survives removal of the winding sector = the
    bulk-physics prediction.

Couplings: J = -0.10, -0.05 (pi-flux), +0.02, +0.04 (0-flux).
Outputs: gauge_probe_prl/notes/source_response.json + figN11.
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
from scipy.sparse.linalg import cg

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ice_pt_lib as ipl  # noqa: E402
import exact_ed_lib as eel  # noqa: E402
from kappa_lambda2_exact import band_states, CACHE, FIGS, NB, refined_peak_T  # noqa: E402
from alpha_selection_full import patterns, build_Xup  # noqa: E402

GP = Path(__file__).resolve().parents[2] / "gauge_probe_prl"
JS = [-0.10, -0.05, +0.02, +0.04]
LAMS_FIT = np.array([0.03, 0.06])
C_PI = "#1f6feb"; C_ZERO = "#e08e0b"; C_RED = "#d1495b"; C_TEAL = "#2a9d8f"

plt.rcParams.update({
    "font.size": 9, "axes.labelsize": 9.5, "axes.titlesize": 9,
    "legend.fontsize": 6.6, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "figure.dpi": 200, "savefig.bbox": "tight"})


def dH_for(cl, B0, Bp, Bm, Psik, Eb, E0abs, Hp, Hm, f):
    """Exact second-order band matrix for source sum_i (f_i S+_i + h.c.)."""
    Xup = build_Xup(cl, B0, Bp, f)              # Sz=0 -> +1, amplitudes f
    # lowering part X_- = sum_i conj(f_i) S^-_i : Sz=0 -> -1, built directly:
    from scipy.sparse import coo_matrix
    rows, cols, vals = [], [], []
    one = np.uint64(1)
    for i in range(cl.n_sites):
        if abs(f[i]) < 1e-14:
            continue
        bi = one << np.uint64(i)
        m = (B0.states & bi) != 0
        new = B0.states[m] & ~bi
        rows.append(np.array([Bm.index[int(x)] for x in new], dtype=np.int64))
        cols.append(np.nonzero(m)[0])
        vals.append(np.full(int(m.sum()), np.conj(f[i]), dtype=complex))
    Xm = coo_matrix((np.concatenate(vals),
                     (np.concatenate(rows), np.concatenate(cols))),
                    shape=(Bm.dim, B0.dim)).tocsr()

    dH = np.zeros((NB, NB), dtype=complex)
    for X, Hs in ((Xup, Hp), (Xm, Hm)):
        Xn = X @ Psik                            # (dim_s, NB) complex
        Y = np.empty_like(Xn)
        has_imag = np.max(np.abs(Xn.imag)) > 1e-13
        for n in range(NB):
            A = Hs - (E0abs + Eb[n]) * identity(X.shape[0], format="csr")
            yr, _ = cg(A, Xn[:, n].real, rtol=1e-8, maxiter=3000)
            if has_imag:
                yi, _ = cg(A, Xn[:, n].imag, rtol=1e-8, maxiter=3000)
                Y[:, n] = yr + 1j * yi
            else:
                Y[:, n] = yr
        XtY = Xn.conj().T @ Y
        dH += -0.5 * (XtY + XtY.conj().T)
    return dH


def main():
    t00 = time.time()
    cl = ipl.build_cluster("cubic", (1, 1, 1))
    B0 = eel.SzBasis(cl)
    Bp = eel.SzBasis(cl, nup=9)
    Bm = eel.SzBasis(cl, nup=7)
    pats = patterns(cl)
    S4 = B0.ring_sum(cl.loops4)
    Shex = B0.ring_sum(cl.hexes)
    mask = B0.transport_mask()
    pats = {k: pats[k] for k in ("unif", "stag", "field111")}
    print(f"setup {time.time()-t00:.0f}s; mask keeps "
          f"{mask.mean():.3f} of elements; drives: {list(pats)}", flush=True)

    results = {}
    for J in JS:
        E, Psi = band_states(B0, J)
        E0abs = float(E[0]); Eb = (E - E0abs)[:NB]; Psik = Psi[:, :NB]
        Hp = Bp.H_xxz(J).real.tocsr()
        Hm = Bm.H_xxz(J).real.tocsr()
        S4b = np.asarray(Psik.T @ (S4 @ Psik))
        S6b = np.asarray(Psik.T @ (Shex @ Psik))
        # Loewdin frame for the mask projection
        A = Psik[B0.ice_rows, :]
        U, s, Vh = np.linalg.svd(A, full_matrices=False)
        Tl = U @ Vh                               # ice-config <- band
        Hcfg = (Tl * Eb) @ Tl.conj().T
        Em_p = np.linalg.eigvalsh(np.where(mask, Hcfg, 0))
        Em_p -= Em_p[0]

        for pname, f in pats.items():
            t0 = time.time()
            dH = dH_for(cl, B0, Bp, Bm, Psik, Eb, E0abs, Hp, Hm, f)
            herm = np.max(np.abs(dH - dH.conj().T))
            dHr = dH  # complex Hermitian
            # ---- decomposition (real part carries everything by symmetry)
            tems = {"c": np.eye(NB), "r": np.diag(Eb),
                    "S4": S4b, "S6": S6b}
            ks = list(tems)
            G = np.array([[np.sum(tems[a] * tems[b]) for b in ks] for a in ks])
            v = np.array([np.real(np.sum(tems[a] * dHr.conj())) for a in ks])
            co = dict(zip(ks, np.linalg.solve(G, v)))
            fit = sum(co[k] * tems[k] for k in ks)
            R = dHr - fit
            base = dHr - np.mean(np.diag(dHr).real) * np.eye(NB)
            varexp = float(1 - np.sum(np.abs(R) ** 2) / np.sum(np.abs(base) ** 2))

            def Tpk(dh, lam):
                Ev = np.linalg.eigvalsh(np.diag(Eb) + lam ** 2 * dh)
                return refined_peak_T(Ev)
            T0 = Tpk(dHr, 0.0)
            x = LAMS_FIT ** 2 / abs(J)
            def Bof(dh):
                D = np.array([1 - Tpk(dh, l) / T0 for l in LAMS_FIT])
                return float(np.sum(D * x) / np.sum(x ** 2))
            B_exact = Bof(dHr)
            contrib = {k: Bof(co[k] * tems[k]) for k in ("r", "S4", "S6")}
            contrib["resid"] = Bof(R)
            # ---- lambda sweep
            lam_grid = np.linspace(0, 0.12, 9)
            sweep = [Tpk(dHr, l) / T0 for l in lam_grid]
            # ---- transport-projected response
            dHcfg = Tl @ dHr @ Tl.conj().T
            dHp = np.where(mask, dHcfg, 0)
            Hp_ = np.where(mask, Hcfg, 0)
            Ep, Vp = np.linalg.eigh(Hp_)
            Ep -= Ep[0]
            dHp_band = Vp.conj().T @ dHp @ Vp
            def Tpk_p(lam):
                Ev = np.linalg.eigvalsh(np.diag(Ep) + lam ** 2 * dHp_band)
                return refined_peak_T(Ev)
            T0p = Tpk_p(0.0)
            Dp = np.array([1 - Tpk_p(l) / T0p for l in LAMS_FIT])
            B_proj = float(np.sum(Dp * x) / np.sum(x ** 2))

            results[f"{J}_{pname}"] = dict(
                B_exact=B_exact, contrib=contrib, coef=co, varexp=varexp,
                herm=float(herm), T0=T0, T0_proj=T0p, B_proj=B_proj,
                lam_grid=lam_grid.tolist(), sweep=list(map(float, sweep)),
                rJ=float(co["r"] * J),
            )
            print(f"J={J:+.2f} [{pname:4s}] ({time.time()-t0:.0f}s): "
                  f"B={B_exact:+7.2f}  rJ={co['r']*J:+7.2f} "
                  f"S4={contrib['S4']:+6.2f} varexp={varexp:.3f} "
                  f"B_proj={B_proj:+7.2f}  (T0={T0:.5f} T0p={T0p:.5f})",
                  flush=True)

    with open(GP / "notes" / "source_response.json", "w") as fjs:
        json.dump(results, fjs, indent=1)

    # ------------------------------------------------------------------ fig
    fig, axs = plt.subplots(1, 2, figsize=(6.9, 2.9))
    styles = {"unif": ("-", "o"), "stag": ("--", "s"), "field111": (":", "^")}
    labels = {"unif": r"$E_g$ (uniform)", "stag": r"$T_{2g}$ (staggered)",
              "field111": r"$[111]$ field ($\hat n\!\cdot\!\hat z_i$)"}
    for ax, J in ((axs[0], -0.05), (axs[1], +0.04)):
        col = C_PI if J < 0 else C_ZERO
        for pname, (ls, mk) in styles.items():
            d = results[f"{J}_{pname}"]
            ax.plot(np.array(d["lam_grid"]) ** 2 / abs(J), d["sweep"],
                    ls, marker=mk, ms=3, color=col, mfc="none",
                    label=f"{labels[pname]}:  $B={d['B_exact']:+.1f}$, "
                          f"$B_{{\\rm proj}}={d['B_proj']:+.1f}$")
        ax.axhline(1, color="0.8", lw=0.7)
        ax.set_xlabel(r"$x=\lambda^2/(|J_\pm|J_{zz})$")
        ax.set_title(f"$J_\\pm={J:+.2f}$ "
                     + (r"($\pi$-flux)" if J < 0 else r"($0$-flux)"))
        ax.legend(frameon=False)
    axs[0].set_ylabel(r"$T_{\rm peak}(\lambda)/T_{\rm peak}(0)$")
    fig.savefig(FIGS / "figN11_sources.pdf")
    print(f"\nDONE {time.time()-t00:.0f}s -> source_response.json, figN11")


if __name__ == "__main__":
    main()
