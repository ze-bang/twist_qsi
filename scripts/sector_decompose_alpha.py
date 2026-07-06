#!/usr/bin/env python3
r"""Exact energy- and spinon-resolved decomposition of alpha_Q(T).

For a source-field ED run the thermal expansion coefficient is

    alpha_Q(T) = d<Q>/dT = sum_n  c_n(T),
    c_n(T)     = p_n (E_n - <H>) q_n / T^2,

with p_n the Boltzmann weight, E_n the eigenvalue, q_n = <n|Q|n> the
(finite-source) diagonal matrix element.  This is an *exact additive*
decomposition over eigenstates, so we may attribute alpha to any
partition of the spectrum.

We partition by:
  (i)  excitation energy  w_n = E_n - E_0   (gauge manifold vs spinon sector)
  (ii) spinon number      n_sp(n) = <n| sum_t Q_t^2 |n>,  Q_t = sum_{i in t} S^z_i

n_sp is the standard ice-rule-violation count: Q_t=0 for a 2-in-2-out
tetrahedron, +-1 for 3-in-1-out, +-2 for all-in/all-out, so sum_t Q_t^2 counts
charged tetrahedra (one spinon-antispinon pair contributes +2).  On the
corner-sharing pyrochlore (8 tetrahedra, each Ising bond in exactly one
tetrahedron) this operator is an *exact* affine function of the diagonal Ising
energy that we already store per eigenstate:

    sum_t Q_t^2 = 2 H_Ising + 8        (verified to 2e-14 on all 2^16 states)

so <n|sum_t Q_t^2|n> = 2 <n|H_Ising|n> + 8 needs no re-diagonalisation.

The goal is to test whether the low-T peak of alpha is carried by the
low-energy gauge manifold (photon/vison scale = lower C peak) rather than
by thermally activated spinons (scale ~ Jzz = upper C peak).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

ROOT = Path(__file__).resolve().parents[2]

# Fixed split between gauge manifold and spinon sector for the illustrative
# energy-split panel.  ~ half the bare spinon (Ising) scale J_zz=1.  The
# headline conclusions use the cut-free centroid / spinon-content diagnostics
# below, not this cut.
W_CUT = 0.5


def find_C_peaks(T, C):
    """Genuine C(T) maxima only: require height and prominence >= 5% of max."""
    Cmax = float(C.max())
    idx, props = find_peaks(C, height=0.05 * Cmax, prominence=0.05 * Cmax)
    return [(float(T[i]), float(C[i])) for i in idx]


def contributions(E, q, T):
    """Return c_n(T) array: shape (nT, nstates). Memory heavy; use chunks."""
    e_min = E.min()
    shifted = E - e_min
    cn = np.zeros((T.size, E.size))
    for i, t in enumerate(T):
        w = np.exp(-shifted / t)
        Z = w.sum()
        p = w / Z
        Hbar = (E * p).sum()
        cn[i] = p * (E - Hbar) * q / (t * t)
    return cn


def analyze(npz_path: Path, q_name: str, out_dir: Path):
    d = np.load(npz_path, allow_pickle=True)
    E = d["eigenvalues"]
    q = d[f"{q_name}_diagonal"]
    eis = d["H_Ising_diagonal"]
    T = d["temperatures"]
    C = d["specific_heat"]
    alpha = d[f"{q_name}_alpha"]
    lam = float(d[f"source_{q_name}"]) if f"source_{q_name}" in d.files else np.nan

    w = E - E.min()                 # excitation energy
    s = 2.0 * eis + 8.0             # true spinon number <n|sum_t Q_t^2|n> (exact)

    Cpeaks = find_C_peaks(T, C)
    Cpeaks_sorted = sorted(Cpeaks, key=lambda x: x[0])
    T_low = Cpeaks_sorted[0][0] if Cpeaks_sorted else None
    T_high = Cpeaks_sorted[-1][0] if len(Cpeaks_sorted) > 1 else None

    i_apk = int(np.argmax(np.abs(alpha)))
    T_apk = float(T[i_apk])

    # --- exact additive contributions at the alpha-peak temperature ---
    e_min = E.min()
    shifted = E - e_min

    def c_at(t):
        ww = np.exp(-shifted / t)
        p = ww / ww.sum()
        Hbar = (E * p).sum()
        return p * (E - Hbar) * q / (t * t)

    c_peak = c_at(T_apk)

    # cumulative alpha vs energy cutoff at T_apk
    order = np.argsort(w)
    w_sorted = w[order]
    c_cum = np.cumsum(c_peak[order])
    alpha_peak_total = c_peak.sum()
    frac = c_cum / alpha_peak_total

    # The cumulative is NOT monotonic (contributions cancel in sign), so a
    # searchsorted "crossing" is meaningless.  Instead report the smallest
    # energy beyond which the cumulative stays within tol of its final value
    # -> the true saturation scale of alpha at the peak temperature.
    def omega_saturation(tol):
        dev = np.abs(frac - 1.0)
        beyond = dev <= tol
        # smallest index i such that all j>=i are within tol
        ok = np.flatnonzero(~beyond)
        if ok.size == 0:
            return 0.0
        last_bad = ok[-1]
        idx = min(last_bad + 1, w_sorted.size - 1)
        return float(w_sorted[idx])

    w50 = omega_saturation(0.10)   # within 10%
    w90 = omega_saturation(0.05)   # within 5%

    # --- energy-split alpha(T): gauge manifold vs spinon sector ---
    # Fixed, transparent cut (~ half the bare spinon/Ising scale J_zz).
    w_cut = W_CUT
    low_mask = w < w_cut

    alpha_low = np.array([c_at(t)[low_mask].sum() for t in T])
    alpha_high = alpha - alpha_low

    # --- cut-free diagnostics vs T (|c_n|-weighted) ---
    # mean excitation energy and mean spinon number of the states that build
    # alpha at each temperature.  These need no energy cut.
    s_bar = np.zeros_like(T)
    w_centroid = np.zeros_like(T)
    for i, t in enumerate(T):
        cc = np.abs(c_at(t))
        wsum = cc.sum()
        s_bar[i] = (cc * s).sum() / wsum if wsum > 0 else np.nan
        w_centroid[i] = (cc * w).sum() / wsum if wsum > 0 else np.nan

    # --- energy-resolved contribution density heatmap c(T, w) ---
    nbins = 60
    w_edges = np.linspace(0, np.percentile(w, 99.5), nbins + 1)
    w_cent = 0.5 * (w_edges[:-1] + w_edges[1:])
    dens = np.zeros((T.size, nbins))
    bidx = np.clip(np.digitize(w, w_edges) - 1, 0, nbins - 1)
    for i, t in enumerate(T):
        cc = c_at(t)
        np.add.at(dens[i], bidx, cc)

    # ---------------------------------------------------------------- plot
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    title = f"{npz_path.parent.name}  |  source {q_name}={lam:.0e}"
    fig.suptitle(title, fontsize=12)

    # Panel A: C(T) + alpha(T) twin
    ax = axes[0, 0]
    ax.plot(T, C, color="darkorange", lw=1.8, label=r"$C(T)$")
    for (tp, cp) in Cpeaks:
        ax.axvline(tp, color="orange", ls=":", lw=1.0)
    ax.set_xscale("log")
    ax.set_xlabel(r"$T/J_{zz}$")
    ax.set_ylabel(r"$C(T)$", color="darkorange")
    ax.tick_params(axis="y", labelcolor="darkorange")
    ax2 = ax.twinx()
    ax2.plot(T, alpha, color="navy", lw=1.8, label=r"$\alpha_Q$")
    ax2.axhline(0, color="gray", lw=0.6)
    ax2.axvline(T_apk, color="navy", ls="--", lw=1.0)
    ax2.set_ylabel(r"$\alpha_Q$", color="navy")
    ax2.tick_params(axis="y", labelcolor="navy")
    if T_high is not None:
        txt = f"lower $C$ peak $T={T_low:.4f}$\n"
        txt += f"upper $C$ peak $T={T_high:.4f}$\n"
    elif T_low is not None:
        txt = f"single $C$ peak $T={T_low:.4f}$\n"
    else:
        txt = ""
    txt += rf"$\alpha$ peak $T={T_apk:.4f}$"
    ax.set_title("C(T) and $\\alpha_Q(T)$:  " + txt.replace("\n", ";  "), fontsize=9)

    # Panel B: cumulative alpha vs energy cutoff at T_apk
    ax = axes[0, 1]
    ax.plot(w_sorted, frac, color="purple", lw=1.6)
    ax.axhline(1.0, color="gray", lw=0.6, ls=":")
    ax.axvline(w_cut, color="green", ls="--", lw=1.2,
               label=rf"$\omega_{{\rm cut}}={w_cut:.3f}$")
    ax.axvline(w50, color="red", ls=":", lw=1.0, label=rf"within 10% by $\omega={w50:.3f}$")
    ax.axvline(w90, color="brown", ls=":", lw=1.0, label=rf"within 5% by $\omega={w90:.3f}$")
    ax.set_xlabel(r"excitation-energy cutoff $\Omega = E_n-E_0$")
    ax.set_ylabel(r"cumulative $\alpha(T_{\rm peak};\,\omega_n<\Omega)\,/\,\alpha_{\rm tot}$")
    ax.set_title(rf"Energy origin of the $\alpha$ peak at $T={T_apk:.4f}$", fontsize=10)
    ax.set_xlim(0, min(2.0, w_sorted.max()))
    ax.legend(fontsize=8)
    ax.grid(True, ls=":", lw=0.4, alpha=0.5)

    # Panel C: alpha split into gauge manifold vs spinon sector
    ax = axes[1, 0]
    ax.plot(T, alpha, color="black", lw=2.0, label=r"$\alpha_Q$ total")
    ax.plot(T, alpha_low, color="green", lw=1.6, ls="--",
            label=rf"$\omega_n<{w_cut:g}$ (gauge manifold)")
    ax.plot(T, alpha_high, color="crimson", lw=1.6, ls="-.",
            label=rf"$\omega_n>{w_cut:g}$ (spinon sector)")
    ax.axhline(0, color="gray", lw=0.6)
    ax.axvline(T_apk, color="navy", ls=":", lw=1.0)
    ax.set_xscale("log")
    ax.set_xlabel(r"$T/J_{zz}$")
    ax.set_ylabel(r"$\alpha_Q$")
    ax.set_title("Energy-split decomposition of $\\alpha_Q(T)$", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, which="both", ls=":", lw=0.4, alpha=0.5)

    # Panel D: contribution-density heatmap c(T, w)
    ax = axes[1, 1]
    vmax = np.percentile(np.abs(dens), 99)
    pcm = ax.pcolormesh(T, w_cent, dens.T, cmap="RdBu_r",
                        vmin=-vmax, vmax=vmax, shading="auto")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(max(1e-3, w_cent[w_cent > 0].min()), w_cent.max())
    ax.axvline(T_apk, color="k", ls="--", lw=1.0)
    if T_low:
        ax.axvline(T_low, color="orange", ls=":", lw=1.2)
    ax.axhline(w_cut, color="green", ls="--", lw=1.0)
    ax.plot(T, w_centroid, color="k", lw=1.4,
            label=r"centroid $\langle\omega\rangle_{|c|}$")
    ax.set_xlabel(r"$T/J_{zz}$")
    ax.set_ylabel(r"$\omega_n = E_n-E_0$")
    ax.set_title(r"$\alpha$-contribution density $c(T,\omega)$", fontsize=10)
    ax.legend(fontsize=8, loc="upper left")
    fig.colorbar(pcm, ax=ax, label=r"$\sum_n c_n$ per $\omega$-bin")

    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = f"{npz_path.parent.name}_{q_name}_sector.png"
    out_path = out_dir / fname
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)

    summary = dict(
        case=npz_path.parent.name, q_name=q_name, lam=lam,
        n_C_peaks=len(Cpeaks), T_low=T_low, T_high=T_high, T_apk=T_apk,
        w_cut=float(w_cut), w50=w50, w90=w90,
        alpha_peak=float(alpha[i_apk]),
        frac_low_at_peak=float(alpha_low[i_apk] / alpha[i_apk]) if alpha[i_apk] != 0 else np.nan,
        nsp_bar_at_peak=float(s_bar[i_apk]),
        nsp_GS=float(s[int(np.argmin(E))]),
        w_centroid_at_peak=float(w_centroid[i_apk]),
    )
    return out_path, summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True, type=Path)
    ap.add_argument("--q", default="Eg_Q1")
    ap.add_argument("--out", type=Path,
                    default=ROOT / "twist_qsi_demo" / "output" / "sector_decomp")
    args = ap.parse_args()
    out_path, summary = analyze(args.npz, args.q, args.out)
    print(f"saved {out_path}")
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
