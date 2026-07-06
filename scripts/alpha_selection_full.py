"""EXACT O(lambda^2) thermodynamic response alpha(T) ~ 2*lambda*d<s>/dT of the
16-site cluster over the FULL temperature range (gauge peak AND charge peak),
for three source form factors:

    unif : f = (1, 1, 1, 1)          uniform (E_g-type)
    stag : f = (1, 1, -1, -1)        staggered, sum f = 0 (T_2g-type)
    tri  : f = (0, 1, w, w*)  w=e^{2pi i/3}, x 2/sqrt(3) norm  ([111]-type:
           apical sublattice silent, 120-degree phases; sum f = 0)

Method: full dense diagonalization of the Sz = 0,1,2,3 sectors (Sz<0 by
spin-flip symmetry: identical spectra; X-weights obtained from the
conjugate pattern). The exact second-order free-energy shift is

  F(lambda) = F0 - lambda^2 <s>_T,   <s>_T = (1/Z) sum_{a<b} |X_ab|^2
              (e^{-bEa} - e^{-bEb})/(E_b - E_a)   [regular as Eb->Ea],

so  <X> = 2 lambda <s>_T  and  alpha = d<X>/dT = 2 lambda d<s>_T/dT.
|X_ab|^2 is accumulated into 2D energy histograms per sector pair, making
the T-sweep free. Patterns are normalized to sum_a |f_a|^2 / 4 = 1 so the
same lambda means the same total drive weight.

Outputs: gauge_probe_prl/notes/alpha_selection_{J}.npz + figN10.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ice_pt_lib as ipl  # noqa: E402
import exact_ed_lib as eel  # noqa: E402

HERE = Path(__file__).resolve().parent
GP = HERE.parents[1] / "gauge_probe_prl"
OUT = GP / "notes"

EBIN = 0.01
JS = [-0.05, +0.04]


def sublattice(cl):
    frac = np.mod(cl.positions, 0.5)
    key = np.round(frac * 4).astype(int) % 2
    codes = key[:, 0] * 4 + key[:, 1] * 2 + key[:, 2]
    uniq = sorted(set(codes.tolist()))
    lut = {u: i for i, u in enumerate(uniq)}
    return np.array([lut[c] for c in codes])


def patterns(cl):
    sub = sublattice(cl)
    w = np.exp(2j * np.pi / 3)
    pats = {
        "unif": np.array([1, 1, 1, 1], complex),
        "stag": np.array([1, 1, -1, -1], complex),
        "tri": np.array([0, 1, w, np.conj(w)], complex) * (2 / np.sqrt(3)),
        # [111] field: f_i = n.z_i with n=[111]/sqrt3 and the four local
        # <111> easy axes -> (1,-1/3,-1/3,-1/3); normalized to sum|f|^2=4.
        "field111": np.array([1, -1 / 3, -1 / 3, -1 / 3], complex) * np.sqrt(3),
    }
    return {k: v[sub] for k, v in pats.items()}


def build_Xup(cl, Blo, Bhi, f):
    """sum_i f_i S+_i : basis Blo (nup) -> Bhi (nup+1), sparse complex."""
    rows, cols, vals = [], [], []
    one = np.uint64(1)
    for i in range(cl.n_sites):
        if abs(f[i]) < 1e-14:
            continue
        bi = one << np.uint64(i)
        m = (Blo.states & bi) == 0
        new = Blo.states[m] | bi
        rows.append(np.array([Bhi.index[int(x)] for x in new], dtype=np.int64))
        cols.append(np.nonzero(m)[0])
        vals.append(np.full(int(m.sum()), f[i], dtype=complex))
    return coo_matrix((np.concatenate(vals),
                       (np.concatenate(rows), np.concatenate(cols))),
                      shape=(Bhi.dim, Blo.dim)).tocsr()


def hist_pair(Vlo, Elo, Vhi, Ehi, X, e0, nbins):
    """Accumulate |<hi|X|lo>|^2 into a 2D histogram over (E_lo, E_hi) bins.
    Complex X handled via separate real/imag GEMMs (V real)."""
    H = np.zeros((nbins, nbins))
    blo = np.minimum((np.round((Elo - e0) / EBIN)).astype(int), nbins - 1)
    bhi = np.minimum((np.round((Ehi - e0) / EBIN)).astype(int), nbins - 1)
    Xr, Xi = X.real.tocsr(), X.imag.tocsr()
    chunk = 2000
    for c0 in range(0, Vlo.shape[1], chunk):
        c1 = min(c0 + chunk, Vlo.shape[1])
        Wr = Xr @ Vlo[:, c0:c1]
        Mr = Vhi.T @ Wr
        M2 = Mr ** 2
        if Xi.nnz:
            Wi = Xi @ Vlo[:, c0:c1]
            Mi = Vhi.T @ Wi
            M2 += Mi ** 2
        idx = (bhi[:, None] * nbins + blo[None, c0:c1]).ravel()
        H_flat = np.bincount(idx, weights=M2.ravel(), minlength=nbins * nbins)
        H += H_flat.reshape(nbins, nbins)
    return H


def main():
    t00 = time.time()
    cl = ipl.build_cluster("cubic", (1, 1, 1))
    pats = patterns(cl)
    print("patterns:", {k: np.round(v[:4], 3).tolist() for k, v in pats.items()},
          flush=True)

    for J in JS:
        print(f"\n===== J = {J:+.2f} =====", flush=True)
        bases, spec = {}, {}
        nups = [8, 9, 10, 11]
        Emax_all = 0.0
        Vcur = {}
        # sector energies + eigenvectors (kept only while needed)
        for nu in nups:
            t0 = time.time()
            B = eel.SzBasis(cl, nup=nu)
            H = B.H_xxz(J).real.toarray()
            E, V = np.linalg.eigh(H)
            bases[nu], spec[nu], Vcur[nu] = B, E, V
            Emax_all = max(Emax_all, E.max())
            print(f"  sector nup={nu} (dim {B.dim}): eigh {time.time()-t0:.0f}s",
                  flush=True)
        e0 = spec[8][0]
        nbins = int((Emax_all - e0) / EBIN) + 2

        hists = {}
        for lo, hi in ((8, 9), (9, 10), (10, 11)):
            for pname, f in pats.items():
                variants = [f] if np.allclose(f.imag, 0) else [f, np.conj(f)]
                Hh = np.zeros((nbins, nbins))
                t0 = time.time()
                for fv in variants:
                    X = build_Xup(cl, bases[lo], bases[hi], fv)
                    Hh += hist_pair(Vcur[lo], spec[lo], Vcur[hi], spec[hi],
                                    X, e0, nbins)
                if len(variants) == 1:
                    Hh *= 2.0        # the (-Sz) mirror pair, identical weight
                hists[(lo, hi, pname)] = Hh
                print(f"  pair {lo}->{hi} [{pname}]: {time.time()-t0:.0f}s",
                      flush=True)
            if lo > 8:
                del Vcur[lo]         # free eigenvectors no longer needed
        del Vcur

        np.savez_compressed(
            OUT / f"alpha_selection_J{J:+.2f}.npz",
            e0=e0, ebin=EBIN, nbins=nbins,
            **{f"E{nu}": spec[nu] for nu in nups},
            **{f"H_{lo}_{hi}_{p}": hists[(lo, hi, p)]
               for (lo, hi, p) in hists},
        )
        print(f"  saved alpha_selection_J{J:+.2f}.npz "
              f"({time.time()-t00:.0f}s elapsed)", flush=True)


if __name__ == "__main__":
    main()
