"""Quantitative checks demanded by the referee report.

A. Few-photon kinematics.  Parity only excludes EVEN photon number, so
   the L level must be tested against the three-photon threshold, not
   merely against parity.  We build the Gaussian photon branch on the
   same torus and minimize the total energy over all momentum-conserving
   n-photon partitions of each cluster momentum.

B. Is the 3/4 reduction of f_1(L) exact or numerical?  Separate the
   purely geometric factor sum_mu |g^mu_p(q)|^2, resolved by hexagon
   family, from the ring-coherence weights w_p that multiply it.

C. Error bars on the classical 2048-site structure factor, by binning
   the loop-update Monte Carlo.

D. Sensitivity of the "off-pole weight" to the ad hoc energy window
   used to call an exact level one-photon.  If 13-38% moves a lot with
   the window, the number is not defensible as stated.

E. Polarization resolution of every bright level, exported as a table.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.sparse.linalg import eigsh

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "notes"))
sys.path.insert(0, str(ROOT / "campaign"))

import recompute_finite_size_artifact as geometry               # noqa: E402
from run_momentum_resolved_spectrum import star                 # noqa: E402
from run_ring48_dssf import contractible                        # noqa: E402
from ring48_momenta_fix import fcc_momenta                      # noqa: E402
from ring48_gaussian import build_curl, build_gradient, bloch_basis  # noqa: E402
from ring48_roton_landing import hex_signed, f1_analytic        # noqa: E402
from ring64_pipeline import (fast_labels, fast_ring,            # noqa: E402
                             hex_masks_patterns)

OUT = ROOT / "campaign" / "outputs" / "ring_model_dssf"
SHAPE = (2, 2, 3)


def log(m, t0):
    print(f"[{time.perf_counter()-t0:7.1f}s] {m}", flush=True)


# ---------------------------------------------------------------- A
def photon_branch(cl, hexes, momenta, shape):
    """Gaussian photon singular values sigma_lambda(m) per momentum."""
    positions = np.asarray(cl.positions, dtype=float)
    n = int(cl.n_sites)
    sub = np.arange(n) % 4
    C = build_curl(cl, hexes)
    d0 = build_gradient(cl)
    gauge = float(np.abs(C @ d0).max())
    mlist = list(np.ndindex(*shape))
    sig = {}
    for qi, q in enumerate(momenta):
        Uq = bloch_basis(positions, sub, q)
        s = np.linalg.svd(C @ Uq, compute_uv=False)
        sig[mlist[qi]] = np.sort(s[s > 1e-10])
    return mlist, sig, gauge


def n_photon_threshold(mlist, sig, shape, target, nph):
    """Lowest total Gaussian energy of nph photons whose momenta add to
    `target` modulo the supercell reciprocal lattice.  Momenta with no
    photon branch (Gamma) are excluded: they carry no quantum."""
    live = [m for m in mlist if len(sig[m]) > 0]
    sh = np.array(shape)
    best = np.inf
    combo = None
    if nph == 2:
        for a in live:
            b = tuple((np.array(target) - np.array(a)) % sh)
            if b not in sig or len(sig[b]) == 0:
                continue
            e = sig[a][0] + sig[b][0]
            if e < best:
                best, combo = e, (a, b)
    elif nph == 3:
        for i, a in enumerate(live):
            for b in live[i:]:
                c = tuple((np.array(target) - np.array(a)
                           - np.array(b)) % sh)
                if c not in sig or len(sig[c]) == 0:
                    continue
                e = sig[a][0] + sig[b][0] + sig[c][0]
                if e < best:
                    best, combo = e, (a, b, c)
    return best, combo


# ---------------------------------------------------------------- B
def hex_family(cl, hexes):
    """Family label of each hexagon = the sublattice it omits.  Immune to
    periodic wrapping, unlike a plane fit."""
    n = int(cl.n_sites)
    sub = np.arange(n) % 4
    fam = []
    for path in hexes:
        present = set(sub[np.asarray(path, dtype=int)])
        missing = [mu for mu in range(4) if mu not in present]
        fam.append(missing[0] if len(missing) == 1 else -1)
    return np.array(fam)


def geometric_factor(hexsign, positions, q, n):
    """sum_mu |g^mu_p(q)|^2 per hexagon -- geometry only, no w_p."""
    sub = np.arange(n) % 4
    phase = np.exp(-2j * np.pi * (positions @ q))
    out = []
    for sites, signs in hexsign:
        z = signs * phase[sites]
        tot = 0.0
        for mu in range(4):
            sel = sub[sites] == mu
            if sel.any():
                tot += abs(z[sel].sum()) ** 2
        out.append(tot)
    return np.array(out)


def main() -> int:
    t0 = time.perf_counter()
    rec = {}
    cl = geometry.build_cluster("fcc", SHAPE)
    n = int(cl.n_sites)
    positions = np.asarray(cl.positions, dtype=float)
    states = np.sort(np.asarray(cl.ice_states, dtype=np.uint64))
    hexes = contractible(cl)
    hmp = hex_masks_patterns(hexes)
    momenta, _ = fcc_momenta(cl)
    hs = hex_signed(hexes, n)
    fam = hex_family(cl, hexes)
    log(f"48-site cluster: {len(hexes)} hexagons, families "
        f"{np.bincount(fam[fam >= 0])}", t0)

    # ground state in its winding sector -------------------------------
    labels, spins_all = fast_labels(states, positions)
    ham, _ = fast_ring(states, hmp, 0.0)
    _, v0 = eigsh(ham, k=1, which="SA", tol=1e-11, ncv=60)
    per = {int(s): float(np.sum(v0[labels == s, 0] ** 2))
           for s in np.unique(labels)}
    sector = max(per, key=per.get)
    rows = np.where(labels == sector)[0]
    hsec = ham[np.ix_(rows, rows)]
    val, vec = np.linalg.eigh(np.asarray(hsec.todense()))
    gs = vec[:, 0]
    e0 = float(val[0])
    log(f"sector {sector}: dim {len(rows)}, E0 = {e0:.6f} K", t0)

    # per-plaquette ring coherence w_p ---------------------------------
    wp = []
    sec_states = states[rows]
    idx = {int(s): i for i, s in enumerate(sec_states)}
    for mask, p1, p2 in hmp:
        sel = sec_states & mask
        hit = np.where((sel == p1) | (sel == p2))[0]
        tgt = sec_states[hit] ^ mask
        acc = 0.0
        for a, tv in zip(hit, tgt):
            b = idx.get(int(tv))
            if b is not None:
                acc += 2.0 * gs[a] * gs[b]
        wp.append(acc)
    wp = np.array(wp)
    log(f"sum_p w_p = {wp.sum():.6f}   -E0/K = {-e0:.6f}   "
        f"(gate ratio {wp.sum()/(-e0):.6f})", t0)
    rec["wp_gate"] = float(wp.sum() / (-e0))

    # ---- A. few-photon kinematics ------------------------------------
    mlist, sig, gauge = photon_branch(cl, hexes, momenta, SHAPE)
    gm = json.load(open(OUT / "ring48_gaussian_match.json"))
    scale = float(gm["scale"])
    log(f"gauge gate |C d0|_max = {gauge:.2e};  "
        f"fitted scale sqrt(2UK) = {scale:.5f}", t0)
    kin = []
    for qi, q in enumerate(momenta):
        m = mlist[qi]
        if len(sig[m]) == 0:
            continue
        one = scale * sig[m][0]
        two, c2 = n_photon_threshold(mlist, sig, SHAPE, m, 2)
        three, c3 = n_photon_threshold(mlist, sig, SHAPE, m, 3)
        kin.append({"m": list(m), "star": star(q),
                    "one": float(one),
                    "two": float(scale * two) if np.isfinite(two) else None,
                    "three": float(scale * three)
                    if np.isfinite(three) else None})
        log(f"  {star(q):<20s} 1ph {one:6.3f}  2ph "
            f"{scale*two:6.3f}  3ph {scale*three:6.3f}", t0)
    rec["kinematics"] = kin

    # ---- B. is the 3/4 reduction exact? ------------------------------
    famtab = {}
    for qi, q in enumerate(momenta):
        g = geometric_factor(hs, positions, np.asarray(q), n)
        byfam = [float(g[fam == mu].sum()) for mu in range(4)]
        famtab[star(q)] = {"per_family": byfam,
                           "geom_total": float(g.sum()),
                           "f1_full": float(f1_analytic(
                               [np.asarray(q)], hs, positions, wp, n)[0]),
                           "f1_uniform": float(g.sum() * wp.mean()
                                               / (2.0 * n))}
    rec["families"] = famtab
    log("per-family geometric factor (sum_p sum_mu |g|^2):", t0)
    for k, v in famtab.items():
        log(f"  {k:<20s} {np.round(v['per_family'],4)}  "
            f"total {v['geom_total']:8.3f}  f1 {v['f1_full']:.5f}", t0)

    # ---- D. window sensitivity of the off-pole weight ----------------
    sens = {}
    for tol in (0.10, 0.15, 0.20, 0.25, 0.30):
        fr = []
        for r in gm["rows"]:
            harm = [scale * s for s in r["sigma"]]
            matched, used = 0.0, []
            for e, w in zip(r["ed_first"], r["ed_firstw"]):
                hit = [h for h in harm
                       if abs(e - h) / h < tol and h not in used]
                if hit:
                    matched += w
                    used.append(hit[0])
            fr.append(1.0 - matched)
        sens[f"{tol:.2f}"] = [float(min(fr)), float(max(fr))]
        log(f"  window {tol:.0%}: off-pole fraction range "
            f"[{min(fr):.3f}, {max(fr):.3f}]", t0)
    rec["offpole_window_sensitivity"] = sens

    # ---- E. polarization table ---------------------------------------
    pol = json.load(open(OUT / "ring48_polarization.json"))
    tab = {}
    for k, v in pol.items():
        if not isinstance(v, dict) or "lines" not in v:
            continue
        rowsout = []
        for ln in v["lines"]:
            tot = ln["total"]
            p = ln["pol"]
            rowsout.append({"omega": ln["omega"], "share": tot,
                            "pol_frac": [p[0] / tot, p[1] / tot],
                            "kernel": ln["kernel"] / max(tot, 1e-300)})
        tab[k] = rowsout
        log(f"polarization at {k}:", t0)
        for r in rowsout:
            log(f"    omega {r['omega']:7.4f}  share {r['share']:.4f}  "
                f"pol ({r['pol_frac'][0]:.3f}, {r['pol_frac'][1]:.3f})  "
                f"long {r['kernel']:.1e}", t0)
    rec["polarization"] = tab

    with open(OUT / "referee_response.json", "w") as fh:
        json.dump(rec, fh, indent=1, default=float)
    log("wrote referee_response.json", t0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
