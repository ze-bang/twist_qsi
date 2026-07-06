"""Config-basis anatomy of the exact O(lambda^2) response dH, to settle:
does the transverse probe renormalize the GAUGE (ring) sector, or only
add a state-dependent potential?

In the ICE-CONFIGURATION basis {|c>}:
  * DIAGONAL  dH_cc  = "-s_c": flip a spin out of |c> and back through a
    virtual spinon pair -> a POTENTIAL on ice configs. Does NOT connect
    configs, does NOT renormalize the ring dynamics.
  * OFF-DIAG  dH_c'c (c'!=c): the virtual pair created by lambda hops via
    Jpm and re-annihilates elsewhere, flipping a whole ice-preserving loop
    -> RENORMALIZES the ring exchange (the genuine ghex renormalization).

We also verify the operator identity behind the "rescaling": diag(E_n) in
the ENERGY eigenbasis == H_gauge, which is PURELY off-diagonal in the
config basis (a ring operator). So the dominant "rescale" template is the
off-diagonal ring renormalization, not a diagonal potential.

For gauge (16-site) and spinon (charge-manifold, 16-site) sectors:
  - split dH into config-diag V and off-diag W;
  - std(V)/|mean(V)|: is the potential ~constant (trivial) or varying?
  - corr(W, H_gauge): is the off-diag proportional to the ring operator?
  - peak-shift B from V-only vs W-only (gauge sector).

Outputs printed; used to rewrite notes Secs 2-4.
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import numpy as np
from scipy.sparse import identity
from scipy.sparse.linalg import cg

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ice_pt_lib as ipl, exact_ed_lib as eel
from kappa_lambda2_exact import band_states, NB, refined_peak_T


def gauge_dH_configbasis(cl, B0, Bp, J):
    """Exact dH in the ice-config basis (90x90) + the ring Hamiltonian
    H_gauge in the same basis, via the Loewdin frame."""
    E, Psi = band_states(B0, J)
    E0 = float(E[0]); Eb = (E - E0)[:NB]; Psik = Psi[:, :NB]
    Hp = Bp.H_xxz(J).real.tocsr()
    # dH in the band eigenbasis (uniform drive)
    from alpha_selection_full import patterns, build_Xup
    f = np.asarray(patterns(cl)["unif"]).real
    Xup = build_Xup(cl, B0, Bp, f + 0j)
    Xn = (Xup @ Psik)
    Y = np.empty_like(Xn, dtype=float)
    for n in range(NB):
        A = Hp - (E0 + Eb[n]) * identity(Bp.dim, format="csr")
        Y[:, n], _ = cg(A, Xn[:, n].real, rtol=1e-9, maxiter=3000)
    XtY = (Xn.real.T @ Y)
    dH_band = -(XtY + XtY.T)          # both Sz sectors, real f (factor 1 each -> 2*; matches kappa_lambda2)
    # Loewdin frame: ice-config <- band
    A = Psik[B0.ice_rows, :]
    U, s, Vh = np.linalg.svd(A, full_matrices=False)
    Tl = U @ Vh
    dH_cfg = Tl @ dH_band @ Tl.T
    Hg_cfg = (Tl * Eb) @ Tl.T          # H_gauge in ice-config basis
    return dH_cfg, Hg_cfg, Eb, dH_band, s.min()


def peakB(Eb, M_band, J):
    lams = [0.0, 0.03, 0.06]
    T = [refined_peak_T(np.linalg.eigvalsh(np.diag(Eb) + l**2 * M_band))
         for l in lams]
    x = np.array([l**2 / abs(J) for l in lams[1:]])
    D = np.array([1 - T[i+1]/T[0] for i in range(2)])
    return float(np.sum(D*x)/np.sum(x**2))


def main():
    cl = ipl.build_cluster("cubic", (1, 1, 1))
    B0 = eel.SzBasis(cl); Bp = eel.SzBasis(cl, nup=9)
    Tl_cache = {}
    for J in (-0.05, +0.04):
        t0 = time.time()
        dH_cfg, Hg_cfg, Eb, dH_band, smin = gauge_dH_configbasis(cl, B0, Bp, J)
        n = dH_cfg.shape[0]
        # --- config-basis diagonal (potential) vs off-diagonal (ring renorm)
        V = np.diag(dH_cfg).copy()                  # potential on ice configs
        W = dH_cfg - np.diag(V)                      # off-diagonal
        # H_gauge config-basis: diagonal should be ~0 (pure ring operator)
        Vg = np.diag(Hg_cfg)
        offg = Hg_cfg - np.diag(Vg)
        # correlation of off-diagonal dH with the ring Hamiltonian off-diag
        wl = W[np.triu_indices(n, 1)]
        gl = offg[np.triu_indices(n, 1)]
        corr = np.corrcoef(np.real(wl), np.real(gl))[0, 1]
        # back to band basis to feed the peak estimator
        # (V and W are config-basis; transform: need Tl^{-1}=Tl^T since unitary)
        # dH_band = Tl^T dH_cfg Tl ; split similarly
        A = eel.SzBasis(cl)  # dummy
        # reuse Tl: recompute
        E, Psi = band_states(B0, J); Psik = Psi[:, :NB]
        Aov = Psik[B0.ice_rows, :]
        U, s, Vh = np.linalg.svd(Aov, full_matrices=False); Tl = U @ Vh
        V_band = Tl.T @ np.diag(V) @ Tl
        W_band = Tl.T @ W @ Tl
        B_full = peakB(Eb, dH_band, J)
        B_V = peakB(Eb, V_band, J)
        B_W = peakB(Eb, W_band, J)
        print(f"\n===== GAUGE sector, J={J:+.2f}  ({time.time()-t0:.0f}s, "
              f"Loewdin smin={smin:.3f}) =====")
        print(f" H_gauge config-diag: |mean|={np.abs(Vg.mean()):.4f} "
              f"std={Vg.std():.4f}  (ring op => should be ~0)")
        print(f" dH config-diagonal potential V_c: mean={V.mean():+.3f} "
              f"std={V.std():.3f}  std/|mean|={V.std()/abs(V.mean()):.3f}")
        print(f" dH off-diagonal vs ring Hamiltonian: corr={corr:+.3f}")
        print(f" peak-shift B: full={B_full:+.2f}  "
              f"config-diagonal-only={B_V:+.2f}  off-diagonal-only={B_W:+.2f}")
        # how much of B is the potential vs the ring renorm
        print(f"   => the POTENTIAL contributes {B_V:+.2f}, "
              f"the RING RENORM contributes {B_W:+.2f} (sum~full)")


if __name__ == "__main__":
    main()
