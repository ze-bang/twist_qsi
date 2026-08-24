"""Why can the soft mode hide on the dark family?  There is an exact reason.

The commutator that generates the first moment is

    [A_q, W_p] = g_p(q) W_p ,     g_p(q) = sum_{j in p} (-1)^j e^{-i q.r_j}

and a DARK plaquette is precisely one with g_p(q) = 0.  So for dark p

    [A_q, W_p] = 0 ,

i.e. the ring-exchange operator on a dark plaquette COMMUTES with the probe.
Consequences, all exact:

  * Dark plaquettes contribute nothing to f1 -- already visible as the zero
    entry in geom_by_family, but now with a reason rather than an observation.
  * H splits as H = H_bright + H_dark with [H_dark, A_q] = 0.  The dark ring
    exchanges therefore act WITHIN each eigenspace of the imposed density
    wave: they can dress the state without disturbing the modulation the
    probe created.  That is the sense in which the dark family is "free"
    real estate for the excitation.

This script checks the commutator numerically on the sector, then asks the
question that the commutator makes sharp: the rigid single-mode state
A_q|0> costs f1/S = 1.846 K, while the true soft eigenstate costs 1.024 K.
WHERE does that 0.82 K of relaxation happen -- on the bright families the
probe drives, or on the dark family it does not?
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.sparse.linalg import eigsh

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for sub in ("src", "notes", "campaign"):
    sys.path.insert(0, str(ROOT / sub))

import recompute_finite_size_artifact as geometry              # noqa: E402
from run_ring48_dssf import contractible                       # noqa: E402
from run_momentum_resolved_spectrum import star                # noqa: E402
from ring48_momenta_fix import fcc_momenta                     # noqa: E402
from ice_enumerate_fast import enumerate_ice_fast, bit_of      # noqa: E402
from ring96_pipeline import (Table128, winding_keys, hex_words,  # noqa: E402
                             ring_matrix)
from f1_all_momenta import hex_family, geom_factor             # noqa: E402

U64 = np.uint64
OUT = ROOT / "campaign" / "outputs" / "ring_model_dssf"


def log(m, t0):
    print(f"[{time.perf_counter()-t0:8.1f}s] {m}", flush=True)


def main() -> int:
    t0 = time.perf_counter()
    shape = tuple(int(x) for x in (sys.argv[1:4] or (2, 2, 3)))
    KMAX = int(sys.argv[4]) if len(sys.argv) > 4 else 40

    orig = geometry.enumerate_ice_states
    geometry.enumerate_ice_states = lambda a, b: np.array([0], dtype=U64)
    cl = geometry.build_cluster("fcc", shape)
    geometry.enumerate_ice_states = orig
    n = int(cl.n_sites)
    pos = np.asarray(cl.positions, dtype=float)
    tets = [tuple(int(s) for s in t) for t in cl.tets]
    hexes = contractible(cl)
    fam = hex_family(hexes, n)

    states = enumerate_ice_fast(n, tets, log=lambda m: None)
    ipos = np.rint(pos * 4).astype(np.int64)
    keys = winding_keys(states, ipos)
    uk, cnt = np.unique(keys, return_counts=True)
    hw_all = hex_words(hexes)
    best = None
    for c in uk[np.argsort(cnt)[::-1][:3]]:
        rows = np.where(keys == c)[0]
        tab = Table128(states[rows])
        H = ring_matrix(tab, hw_all, t0)
        e = float(eigsh(H, k=1, which="SA", tol=1e-10, ncv=30,
                        return_eigenvectors=False)[0])
        if best is None or e < best[0] - 1e-9:
            best = (e, tab, H)
    _, tab, H = best
    del states, keys
    log(f"fcc{shape}: sector dim {len(tab):,}", t0)

    vals, vecs = eigsh(H, k=KMAX, which="SA", tol=1e-13,
                       ncv=min(len(tab) - 1, max(80, 3 * KMAX)))
    o = np.argsort(vals)
    vals, vecs = vals[o], vecs[:, o]
    e0 = float(vals[0])
    gs = vecs[:, 0]
    subl = np.arange(n) % 4

    # every distinct L momentum, so both inequivalent classes are covered
    allq = fcc_momenta(cl)[0]
    Lqs = [q for q in allq if star(q) == "L"]

    results = {}
    for qi, q in enumerate(Lqs):
        g = geom_factor(hexes, pos, q, n)
        gf = np.array([float(g[fam == mu].sum()) for mu in range(4)])
        dark = [mu for mu in range(4) if gf[mu] < 1e-9]
        if not dark:
            continue

        # probe vectors, and the rigid single-mode state they define
        ph = np.exp(-2j * np.pi * (pos @ q))
        cols = []
        for mu in range(4):
            acc = np.zeros(len(tab), dtype=complex)
            for s in np.where(subl == mu)[0]:
                acc += (bit_of(tab.w, int(s)) - 0.5) * ph[s]
            cols.append(acc * gs)
        wts = np.array([float(np.vdot(c, c).real) for c in cols])
        S = float(wts.sum()) / n
        if S < 1e-9:
            continue

        Hfam = [ring_matrix(tab, hex_words(
            [h for h, f in zip(hexes, fam) if f == mu]), t0) for mu in range(4)]
        base = np.array([float(gs @ (Hm @ gs)) for Hm in Hfam])

        # ---- exact commutator check on the dark family -------------------
        Hd = ring_matrix(tab, hex_words(
            [h for h, f in zip(hexes, fam) if f in dark]), t0)
        Hb = ring_matrix(tab, hex_words(
            [h for h, f in zip(hexes, fam) if f not in dark]), t0)
        # apply A_q (sublattice-summed) as a diagonal operator
        diag = np.zeros(len(tab), dtype=complex)
        for s in range(n):
            diag += (bit_of(tab.w, int(s)) - 0.5) * ph[s]
        v = gs
        cd = Hd @ (diag * v) - diag * (Hd @ v)
        cb = Hb @ (diag * v) - diag * (Hb @ v)

        # ---- rigid SMA state, decomposed by family -----------------------
        sma = np.zeros(4)
        for k in range(4):
            if wts[k] < 1e-16:
                continue
            psi = cols[k] / np.linalg.norm(cols[k])
            fr = wts[k] / wts.sum()
            sma += fr * np.array(
                [float(np.real(np.vdot(psi, Hm @ psi))) for Hm in Hfam])
        sma -= base

        # ---- the true soft eigenstate, probe-projected -------------------
        groups, i = [], 1
        while i < len(vals):
            j = i
            while j + 1 < len(vals) and abs(vals[j + 1] - vals[i]) < 1e-8:
                j += 1
            groups.append(list(range(i, j + 1)))
            i = j + 1

        rows = []
        for grp in groups:
            V = vecs[:, grp]
            comp = np.array([V.conj().T @ c for c in cols])
            w = np.array([float((np.abs(comp[k]) ** 2).sum())
                          for k in range(4)])
            if w.sum() < 1e-14:
                continue
            per = np.zeros(4)
            for k in range(4):
                if w[k] < 1e-16:
                    continue
                psi = V @ comp[k]
                psi = psi / np.linalg.norm(psi)
                per += (w[k] / w.sum()) * np.array(
                    [float(np.real(np.vdot(psi, Hm @ psi))) for Hm in Hfam])
            per -= base
            rows.append({"omega": float(vals[grp[0]] - e0),
                         "w": float(w.sum()) / n, "per": per})

        rows = [r for r in rows if r["omega"] > 1e-9]
        rows.sort(key=lambda r: -r["w"])
        top = sorted(rows[:2], key=lambda r: r["omega"])
        if len(top) < 2:
            continue

        d = dark
        b = [mu for mu in range(4) if mu not in d]

        print()
        print("=" * 78)
        print(f"  q = {np.round(q, 4)}   S(q) = {S:.4f}   "
              f"dark family {d}   f1/S = {'n/a'}")
        print("=" * 78)
        print(f"  [H_dark , A_q] on the ground state : "
              f"||.|| = {np.linalg.norm(cd):.3e}")
        print(f"  [H_bright, A_q] on the ground state: "
              f"||.|| = {np.linalg.norm(cb):.3e}   "
              f"(must NOT vanish)")
        print()
        print(f"  {'state':<22s} {'omega':>8s} {'BRIGHT':>9s} {'DARK':>9s}")
        print("  " + "-" * 52)
        print(f"  {'rigid A_q|0> (SMA)':<22s} {sma.sum():>8.4f} "
              f"{sma[b].sum():>9.4f} {sma[d].sum():>9.4f}")
        for r, nm in zip(top, ("soft eigenstate", "partner")):
            print(f"  {nm:<22s} {r['omega']:>8.4f} "
                  f"{r['per'][b].sum():>9.4f} {r['per'][d].sum():>9.4f}")
        rel = sma - top[0]["per"]
        print("  " + "-" * 52)
        print(f"  {'RELAXATION SMA->soft':<22s} {rel.sum():>8.4f} "
              f"{rel[b].sum():>9.4f} {rel[d].sum():>9.4f}")
        if abs(rel.sum()) > 1e-6:
            print(f"  {'  as a fraction':<22s} {1.0:>8.2f} "
                  f"{rel[b].sum()/rel.sum():>9.2f} "
                  f"{rel[d].sum()/rel.sum():>9.2f}")
        print()
        print(f"  bright-family cost ratio soft/partner : "
              f"{top[0]['per'][b].sum()/top[1]['per'][b].sum():.4f}")
        print(f"  overall energy ratio                  : "
              f"{top[0]['omega']/top[1]['omega']:.4f}")

        results[f"q{qi}"] = {
            "q": [float(x) for x in q], "S": S, "dark": d,
            "sma": sma.tolist(),
            "levels": [{"omega": r["omega"], "w": r["w"],
                        "per": r["per"].tolist()} for r in top]}

    (OUT / ("dark_algebra_fcc%s.json" % "".join(map(str, shape)))).write_text(
        json.dumps(results, indent=1))
    print()
    log("done", t0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
