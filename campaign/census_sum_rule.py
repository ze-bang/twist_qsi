"""Must the flippability census be negative for a photon?

The intuition is that an excitation disrupts ring resonance and should
therefore REMOVE flippable hexagons, making
d omega/dV = <n|N|n> - <0|N|0> negative for every ordinary level, with
the roton the lone exception.  A small POSITIVE value was measured for
one photon branch at 48 sites, so either the intuition is wrong or that
number is noise.

There is an exact statement available, and it constrains the AVERAGE,
not individual states.  N = sum_p n_p^flip is diagonal in the
configuration basis, and each flippable hexagon of a configuration
contributes exactly one off-diagonal element of the ring Hamiltonian,
so

    Tr N = nnz(H)   exactly,

whence the census averaged over ALL eigenstates of the sector is

    (1/D) sum_n d omega_n/dV = nnz(H)/D - <0|N|0> ,

negative precisely because the ground state is MORE flippable than a
typical configuration.  Excitations lose flippability on average --
but nothing forces any individual level negative.

This computes the whole census distribution by dense diagonalization,
checks the trace sum rule to machine precision, and reports the bright
levels member by member instead of averaging over multiplets.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "notes"))
sys.path.insert(0, str(ROOT / "campaign"))

import recompute_finite_size_artifact as geometry               # noqa: E402
from run_momentum_resolved_spectrum import star                 # noqa: E402
from run_ring48_dssf import contractible                        # noqa: E402
from ring48_momenta_fix import fcc_momenta                      # noqa: E402
from ice_enumerate_fast import enumerate_ice_fast, bit_of       # noqa: E402
from ring96_pipeline import (Table128, winding_keys, hex_words,
                             ring_matrix, nflip_counts)          # noqa: E402

OUT = ROOT / "campaign" / "outputs" / "ring_model_dssf"
U64 = np.uint64


def log(m, t0):
    print(f"[{time.perf_counter()-t0:7.1f}s] {m}", flush=True)


def main() -> int:
    t0 = time.perf_counter()
    shape = tuple(int(x) for x in sys.argv[1:4]) if len(sys.argv) >= 4 \
        else (2, 2, 3)

    orig = geometry.enumerate_ice_states
    geometry.enumerate_ice_states = lambda a, b: np.array([0], dtype=U64)
    cl = geometry.build_cluster("fcc", shape)
    geometry.enumerate_ice_states = orig
    n = int(cl.n_sites)
    positions = np.asarray(cl.positions, dtype=float)
    tets = [tuple(int(s) for s in t) for t in cl.tets]
    hexes = contractible(cl)

    states = enumerate_ice_fast(n, tets, log=lambda m: None)
    ipos = np.rint(positions * 4).astype(np.int64)
    keys = winding_keys(states, ipos)
    uk, cnt = np.unique(keys, return_counts=True)
    hw = hex_words(hexes)
    best = None
    for c in uk[np.argsort(cnt)[::-1][:3]]:
        rows = np.where(keys == c)[0]
        tb = Table128(states[rows])
        Hc = ring_matrix(tb, hw, t0)
        e = float(eigsh(Hc, k=1, which="SA", tol=1e-10, ncv=30,
                        return_eigenvectors=False)[0])
        if best is None or e < best[0] - 1e-9:
            best = (e, tb, Hc)
    _, tab, H = best
    del states, keys
    D = len(tab)
    log(f"fcc{shape}: sector dim {D:,}, nnz {H.nnz:,}", t0)

    nf = nflip_counts(tab.w, hw)
    log(f"trace check: sum_states n_flip = {nf.sum():.1f}, nnz(H) = "
        f"{H.nnz}  -> "
        f"{'IDENTICAL' if abs(nf.sum() - H.nnz) < 0.5 else 'DIFFER'}", t0)

    vals, vecs = eigh(np.asarray(H.todense()))
    omega = vals - vals[0]
    gs = vecs[:, 0]
    nf_gs = float(np.sum(gs ** 2 * nf))
    cens_all = (vecs ** 2).T @ nf - nf_gs
    log(f"E0 = {vals[0]:.8f} K,  <N>_0 = {nf_gs:.6f}", t0)

    mean_meas = float(cens_all.mean())
    mean_pred = float(H.nnz) / D - nf_gs
    log(f"\nSUM RULE  mean census over all {D:,} eigenstates:", t0)
    log(f"  measured  {mean_meas:+.10f}", t0)
    log(f"  predicted {mean_pred:+.10f}   (= nnz/D - <N>_0)", t0)
    log(f"  difference {abs(mean_meas - mean_pred):.2e}", t0)

    npos = int((cens_all > 0).sum())
    log(f"\ncensus over all eigenstates: positive {npos:,} of {D:,} "
        f"({100 * npos / D:.2f}%)   min {cens_all.min():+.4f}   "
        f"max {cens_all.max():+.4f}", t0)
    for lo, hi in ((0.0, 0.05), (0.05, 0.2), (0.2, 0.4), (0.4, 100.0)):
        c = int(((cens_all > lo) & (cens_all <= hi)).sum())
        log(f"    {lo:+.2f} < census <= {hi:+.2f} : {c:,}", t0)

    subl = np.arange(n) % 4
    momenta, _ = fcc_momenta(cl)
    spins = np.stack([bit_of(tab.w, s).astype(np.float64) - 0.5
                      for s in range(n)], axis=1)
    seen, out = set(), []
    log("\nbright levels, each degenerate member resolved:", t0)
    for q in momenta:
        lab = star(q)
        phase = np.exp(-2j * np.pi * (positions @ np.asarray(q)))
        A = np.zeros((D, 4), dtype=complex)
        for mu in range(4):
            A[:, mu] = vecs.conj().T @ ((spins @ (phase * (subl == mu)))
                                        * gs)
        A /= np.sqrt(n)
        w = np.sum(np.abs(A) ** 2, axis=1)
        S = float(w.sum())
        if S < 1e-12:
            continue
        key = (lab, round(S, 6))
        if key in seen:
            continue
        seen.add(key)
        idx = np.argsort(w)[::-1][:8]
        idx = idx[w[idx] / S > 1e-3]
        log(f"  {lab}   S = {S:.5f}", t0)
        for j in sorted(idx, key=lambda j: omega[j]):
            log(f"      omega {omega[j]:7.4f}  share {w[j] / S:.4f}  "
                f"census {cens_all[j]:+.6f}", t0)
            out.append({"star": lab, "omega": float(omega[j]),
                        "share": float(w[j] / S),
                        "census": float(cens_all[j])})

    tag = "".join(map(str, shape))
    with open(OUT / f"census_sum_rule_fcc{tag}.json", "w") as fh:
        json.dump({"shape": list(shape), "dim": D, "nnz": int(H.nnz),
                   "nflip_ground": nf_gs,
                   "mean_census_measured": mean_meas,
                   "mean_census_predicted": mean_pred,
                   "n_positive": npos, "levels": out}, fh, indent=1,
                  default=float)
    log("\nwrote census_sum_rule json", t0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
