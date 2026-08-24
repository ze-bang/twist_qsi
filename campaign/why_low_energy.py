"""Why is the soft level low in energy?  Ask the exact eigenvector.

For this Hamiltonian the excitation energy is not merely correlated with ring
coherence -- it IS lost ring coherence, exactly and plaquette by plaquette:

    H = -K sum_p (W_p + W_p^dagger)
    omega_n = E_n - E_0 = K sum_p [ <0|W_p+W_p'|0> - <n|W_p+W_p'|n> ]
                        = K sum_p  Delta_p(n)

So every excitation has an exact, additive, real-space bill.  At 48 sites we
have the eigenvectors, so we can read that bill off for the soft level and for
its photon-like partner AT THE SAME MOMENTUM, and see where each one pays.

The hypothesis worth killing or confirming: at L one hexagon family (the one
whose plane is perpendicular to q) has identically vanishing curl.  If the soft
level pays little or nothing on THAT family while its partner pays its share,
then the geometric cancellation is not merely a property of the momentum -- it
is exploited by one eigenstate and not the other, which is exactly what the
momentum-only mechanism could not explain.

Control: repeat at X, where no family is dark, and the family-selective saving
should disappear.
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
from f1_all_momenta import hex_family                          # noqa: E402

U64 = np.uint64
OUT = ROOT / "campaign" / "outputs" / "ring_model_dssf"


def log(m, t0):
    print(f"[{time.perf_counter()-t0:8.1f}s] {m}", flush=True)


def main() -> int:
    t0 = time.perf_counter()
    shape = tuple(int(x) for x in (sys.argv[1:4] or (2, 2, 3)))
    KMAX = int(sys.argv[4]) if len(sys.argv) > 4 else 60

    orig = geometry.enumerate_ice_states
    geometry.enumerate_ice_states = lambda a, b: np.array([0], dtype=U64)
    cl = geometry.build_cluster("fcc", shape)
    geometry.enumerate_ice_states = orig
    n = int(cl.n_sites)
    pos = np.asarray(cl.positions, dtype=float)
    tets = [tuple(int(s) for s in t) for t in cl.tets]
    hexes = contractible(cl)
    fam = hex_family(hexes, n)
    log(f"fcc{shape}: {n} sites, {len(hexes)} hexagons, "
        f"families {np.bincount(fam)}", t0)

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
    log(f"ground sector dim {len(tab):,}", t0)

    vals, vecs = eigsh(H, k=KMAX, which="SA", tol=1e-13,
                       ncv=min(len(tab)-1, max(80, 3*KMAX)))
    o = np.argsort(vals)
    vals, vecs = vals[o], vecs[:, o]
    e0 = float(vals[0])
    log(f"E0 = {e0:.8f} K", t0)

    # per-family ring-coherence operators, as separate sparse matrices
    Hfam = []
    for mu in range(4):
        sel = [h for h, f in zip(hexes, fam) if f == mu]
        Hfam.append(ring_matrix(tab, hex_words(sel), t0))

    # sanity: the families must reconstruct the full Hamiltonian
    gs = vecs[:, 0]
    tot = sum(float(gs @ (Hm @ gs)) for Hm in Hfam)
    log(f"GATE  sum_mu <0|H_mu|0> = {tot:.10f} vs E0 = {e0:.10f}  "
        f"(diff {abs(tot-e0):.2e})", t0)

    HP_CACHE = [ring_matrix(tab, hex_words([h]), t0) for h in hexes]
    # identify which levels carry the L / X response
    subl = np.arange(n) % 4
    out = {}
    allq = fcc_momenta(cl)[0]
    todo = []
    for label in ("L", "X"):
        qs = [q for q in allq if star(q) == label]
        if qs:
            todo.append((label, qs[0]))
    for label, q in todo:
        ph = np.exp(-2j * np.pi * (pos @ q))
        cols = []
        for mu in range(4):
            sites = np.where(subl == mu)[0]
            acc = np.zeros(len(tab), dtype=complex)
            for s in sites:
                acc += (bit_of(tab.w, int(s)) - 0.5) * ph[s]
            cols.append(acc * gs)

        print()
        print("=" * 78)
        print(f"  {label}:  where does each level pay its energy?")
        print("=" * 78)
        print(f"  q = {np.round(q, 4)}")

        # geometric weight per family at this q, for reference
        from f1_all_momenta import geom_factor
        g = geom_factor(hexes, pos, q, n)
        gf = [float(g[fam == mu].sum()) for mu in range(4)]
        print(f"  geometric weight by family : "
              f"{'  '.join(f'{x:7.1f}' for x in gf)}")
        dark = [mu for mu in range(4) if gf[mu] < 1e-9]
        print(f"  dark family (zero curl)    : "
              f"{dark if dark else 'none'}")

        base = np.array([float(gs @ (Hm @ gs)) for Hm in Hfam])
        print(f"  ground-state coherence by family: "
              f"{'  '.join(f'{-x:7.4f}' for x in base)}   "
              f"(NOT equal: the 2x2x3 box makes the families inequivalent)")

        # Degenerate multiplets must be treated as a whole: the per-family
        # decomposition of an individual member is basis-dependent (the two
        # members of the 1.0235 doublet differ by a mu=2 <-> mu=3 swap).  The
        # multiplet TRACE is basis-invariant, so average over each multiplet.
        groups, i = [], 1
        while i < len(vals):
            j = i
            while j + 1 < len(vals) and abs(vals[j + 1] - vals[i]) < 1e-8:
                j += 1
            groups.append(list(range(i, j + 1)))
            i = j + 1

        # per-plaquette operators, for the localization measure
        Hp = HP_CACHE
        bp = np.array([float(gs @ (Hq @ gs)) for Hq in Hp])

        print()
        print(f"  {'omega':>8s} {'deg':>4s} {'weight':>8s} | "
              f"{'probe-projected share of omega  ':^34s} | {'spread':>7s}")
        print(f"  {'':>8s} {'':>4s} {'':>8s} | "
              f"{'mu=0':>8s}{'mu=1':>8s}{'mu=2':>8s}{'mu=3':>8s} | "
              f"{'N_eff':>7s}")
        print("  " + "-" * 74)

        # The multiplet spans several members of the q-star, so a plain
        # multiplet average symmetrises over which family is dark and destroys
        # the very distinction we are after.  The correct q-specific and
        # basis-invariant object is the state the PROBE creates at this q,
        # projected onto the multiplet:
        #     rho = sum_mu P A^mu|0><0|A^mu' P  /  sum_mu ||P A^mu|0>||^2
        # whose family decomposition is Tr(rho H_family).  P is basis
        # independent, so rho is too.
        rows = []
        for grp in groups:
            om = float(vals[grp[0]] - e0)
            if om < 1e-9:
                continue
            V = vecs[:, grp]                       # basis of the eigenspace
            # components of each probe vector inside the multiplet
            comp = np.array([V.conj().T @ c for c in cols])   # (4, deg)
            wts = (np.abs(comp) ** 2).sum(axis=1)
            tot = float(wts.sum())
            if tot < 1e-14:
                continue
            per = np.zeros(4)
            pp = np.zeros(len(hexes))
            for k in range(4):
                if wts[k] < 1e-16:
                    continue
                psi = V @ comp[k]
                psi = psi / np.linalg.norm(psi)
                psi = np.real(psi) if np.max(np.abs(np.imag(psi))) < 1e-9 \
                    else psi
                fr = float(wts[k] / tot)
                per += fr * np.array(
                    [float(np.real(np.vdot(psi, Hm @ psi))) for Hm in Hfam])
                pp += fr * np.array(
                    [float(np.real(np.vdot(psi, Hq @ psi))) for Hq in Hp])
            per -= base
            pp -= bp
            if abs(per.sum()) < 1e-12:
                continue
            frac = per / per.sum()
            a = np.abs(pp)
            neff = float(a.sum() ** 2 / (a ** 2).sum()) if a.sum() > 0 else 0.0
            ds = sum(frac[mu] for mu in dark) if dark else float("nan")
            w = tot / n
            print(f"  {om:>8.4f} {len(grp):>4d} {w:>8.4f} | "
                  f"{frac[0]:>8.4f}{frac[1]:>8.4f}{frac[2]:>8.4f}{frac[3]:>8.4f}"
                  f" | {neff:>7.2f}")
            rows.append({"omega": om, "deg": len(grp), "weight": w,
                         "family_share": frac.tolist(), "n_eff": neff,
                         "dark_share": None if not dark else float(ds)})

        if dark:
            bright = sorted([r for r in rows if r["weight"] > 0.02],
                            key=lambda r: r["omega"])
            if len(bright) >= 2:
                lo, hi = bright[0], bright[1]
                d = dark[0]
                b = [mu for mu in range(4) if mu != d]
                lob = sum(lo["family_share"][mu] for mu in b) * lo["omega"]
                hib = sum(hi["family_share"][mu] for mu in b) * hi["omega"]
                lod = lo["family_share"][d] * lo["omega"]
                hid = hi["family_share"][d] * hi["omega"]
                print()
                print("  " + "-" * 74)
                print(f"  Energy paid, in K, split by whether the probe drives"
                      f" the family:")
                print(f"    {'':<14s} {'3 BRIGHT families':>20s} "
                      f"{'1 DARK family':>16s} {'total':>9s}")
                print(f"    {'soft  level':<14s} {lob:>20.4f} {lod:>16.4f} "
                      f"{lo['omega']:>9.4f}")
                print(f"    {'partner':<14s} {hib:>20.4f} {hid:>16.4f} "
                      f"{hi['omega']:>9.4f}")
                print(f"    {'ratio':<14s} {lob/hib:>20.4f} {lod/hid:>16.4f} "
                      f"{lo['omega']/hi['omega']:>9.4f}")
                print()
                print(f"    The soft level costs {lo['omega']/hi['omega']:.3f}"
                      f" of its partner overall, but only "
                      f"{lob/hib:.3f} of it on the three families the probe")
                print(f"    actually drives.  Its saving is entirely there;"
                      f" on the dark family it pays {lod/hid:.2f}x MORE.")
                print()
                print(f"    localization: cost carried by N_eff = "
                      f"{lo['n_eff']:.1f} plaquettes (soft) vs "
                      f"{hi['n_eff']:.1f} (partner), of {len(hexes)}")
        out[label] = rows
        out[label] = rows

    (OUT / ("why_low_energy_fcc%s.json" % "".join(map(str, shape)))).write_text(json.dumps(out, indent=1))
    print()
    log("wrote why_low_energy_fcc%s.json" % "".join(map(str, shape)), t0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
