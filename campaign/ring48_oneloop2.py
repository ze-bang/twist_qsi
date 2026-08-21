"""Part D rerun with a self-consistent gate.

Gate v2: the exact 3-mode Fock Hamiltonian is built with THE SAME
B-amplitudes g = G/sqrt(2 w) the Wick formulas use, and the quartic
coupling is chosen small enough that shifts are <5% of the level spacing.
Only then are the production formulas trusted.
"""

from __future__ import annotations

import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "notes"))
sys.path.insert(0, str(ROOT / "campaign"))

import recompute_finite_size_artifact as geometry               # noqa: E402
from run_ring48_dssf import contractible                         # noqa: E402
from ring48_close_pendings import oneloop_part                   # noqa: E402

OUT = ROOT / "campaign" / "outputs" / "ring_model_dssf"
SCALE = 0.586840


def gate_v2(gfull, w):
    nm, nmax = 3, 7
    g = gfull[:, :nm]
    wm = w[:nm]
    idx = list(itertools.product(range(nmax), repeat=nm))
    pos = {t: i for i, t in enumerate(idx)}
    dim = len(idx)
    H0 = np.zeros((dim, dim))
    for i, occ in enumerate(idx):
        H0[i, i] = sum(wm[m] * occ[m] for m in range(nm))
    X = []
    for m in range(nm):
        xm = np.zeros((dim, dim))
        for i, occ in enumerate(idx):
            if occ[m] + 1 < nmax:
                j = pos[tuple(o + (1 if k == m else 0)
                              for k, o in enumerate(occ))]
                amp = np.sqrt(occ[m] + 1)
                xm[j, i] += amp
                xm[i, j] += amp
        X.append(xm)
    lam = 1e-3                       # weak quartic for a perturbative gate
    Hp = H0.copy()
    npl = 8                          # few plaquettes keeps shifts small
    for p in range(npl):
        B = sum(g[p, m] * X[m] for m in range(nm))
        Hp += lam * (B @ B @ B @ B)
    ev0 = np.linalg.eigvalsh(H0)
    ev1 = np.linalg.eigvalsh(Hp)
    S = np.sum(g[:npl] ** 2, axis=1)
    ok = True
    print("D-gate v3 (degeneracy-aware comparison):")
    vac_pred = lam * float(np.sum(3 * S ** 2))
    vac_exact = float(ev1[0] - ev0[0])
    print(f"   vacuum: exact {vac_exact:+.6e} vs Wick {vac_pred:+.6e}")
    if abs(vac_exact - vac_pred) > 0.05 * abs(vac_pred):
        ok = False
    # degenerate-PT matrix over the 3 modes, eigenvalues per multiplet
    M = lam * 12 * (g[:npl] * S[:, None]).T @ g[:npl]
    cand = ev1 - ev1[0]
    for val in np.unique(np.round(wm, 9)):
        sel = np.where(np.abs(wm - val) < 1e-9)[0]
        pred = np.sort(np.linalg.eigvalsh(M[np.ix_(sel, sel)]))
        got = np.sort([float(cand[np.argsort(np.abs(cand - (val + p)))[0]]
                             - val) for p in pred])
        for pr, ex in zip(pred, got):
            print(f"   multiplet w={val:.4f}: exact {ex:+.6e} vs "
                  f"Wick {pr:+.6e}  ratio {ex/pr if pr else np.nan:.3f}")
            if abs(ex - pr) > 0.10 * max(abs(pr), 1e-12):
                ok = False
    return ok


def main() -> int:
    t0 = time.perf_counter()
    cl = geometry.build_cluster("fcc", (2, 2, 3))
    hexes = contractible(cl)
    n = int(cl.n_sites)
    C = np.zeros((len(hexes), n))
    for h, path in enumerate(hexes):
        for j, site in enumerate(path):
            C[h, site] = 1.0 if j % 2 == 0 else -1.0
    u, s, vt = np.linalg.svd(C)
    keep = s > 1e-9
    sig = s[keep]
    w = sig * SCALE
    # self-consistent amplitudes at the physical U/K = SCALE^2/2
    uok = SCALE ** 2 / 2
    g = (u[:, keep] * sig[None, :]) * np.sqrt(uok / (2 * w))[None, :]
    ok = gate_v2(g, w)
    print(f"D-gate v2 -> {'PASS' if ok else 'FAIL'}")
    rec = {"gate_pass": bool(ok)}
    if ok:
        rec["oneloop"] = oneloop_part(cl)
    with open(OUT / "ring48_oneloop2.json", "w") as fh:
        json.dump(rec, fh, indent=1, default=float)
    print(f"done ({time.perf_counter()-t0:.0f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
