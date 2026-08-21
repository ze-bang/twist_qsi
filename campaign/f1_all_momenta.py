"""Is the anomaly tied to the L POINT, or to the RATIO f_1/S?

The 72-site sweep finds a second momentum, (1/2,1/2,1/6), whose
spectrum is nearly identical to L's: a dominant level at 1.075K with
55% of the weight against L's 1.068K with 54%.  Either the effect is
not specific to L, or that momentum shares whatever makes L special.

The mechanism makes a sharp, falsifiable prediction, so this tests it
directly.  Both factors are ground-state quantities:

  f_1(q) = (K/2N) sum_p <W_p + W_p^dag> sum_mu |g^mu_p(q)|^2
  S(q)   = equal-time structure factor,

and the single-mode energy is their ratio.  If the two momenta soften
because f_1/S is small at both, the mechanism is confirmed and the
paper's framing should be "wherever f_1/S is minimal", not "at L".  If
f_1/S is small only at L, something else is going on.

The per-family geometric factor is reported too, so we can see whether
the second momentum also has a hexagon family that decouples.

GATE: sum_p <W_p + W_p^dag> = -E_0/K exactly.  Note the correct
plaquette formula is w_p = sum_{a flippable} gs[a] gs[a^mask]; summing
2*gs[a]*gs[b] over all flippable a double counts each pair.
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
from ice_enumerate_fast import enumerate_ice_fast, bit_of       # noqa: E402
from ring96_pipeline import (Table128, winding_keys, hex_words,  # noqa: E402
                             ring_matrix)

OUT = ROOT / "campaign" / "outputs" / "ring_model_dssf"
U64 = np.uint64
CHUNK = 1 << 21


def log(m, t0):
    print(f"[{time.perf_counter()-t0:7.1f}s] {m}", flush=True)


def plaquette_w(tab, hw, gs):
    """w_p = <W_p + W_p^dag> per hexagon, correctly normalized."""
    W = tab.w
    out = []
    for mask, p1, p2 in hw:
        s0 = W[:, 0] & mask[0]
        s1 = W[:, 1] & mask[1]
        hit = np.where(((s0 == p1[0]) & (s1 == p1[1]))
                       | ((s0 == p2[0]) & (s1 == p2[1])))[0]
        if not len(hit):
            out.append(0.0)
            continue
        tgt = np.empty((len(hit), 2), dtype=U64)
        tgt[:, 0] = W[hit, 0] ^ mask[0]
        tgt[:, 1] = W[hit, 1] ^ mask[1]
        pos = tab.index_of(tgt)
        ok = pos >= 0
        out.append(float(np.sum(gs[hit[ok]] * gs[pos[ok]])))
    return np.array(out)


def hex_family(hexes, n):
    """Family = the sublattice each hexagon omits (wrapping-immune)."""
    sub = np.arange(n) % 4
    fam = []
    for path in hexes:
        present = set(sub[np.asarray(path, dtype=int)])
        missing = [mu for mu in range(4) if mu not in present]
        fam.append(missing[0] if len(missing) == 1 else -1)
    return np.array(fam)


def geom_factor(hexes, positions, q, n):
    """sum_mu |g^mu_p(q)|^2 per hexagon -- geometry only."""
    sub = np.arange(n) % 4
    phase = np.exp(-2j * np.pi * (positions @ q))
    out = []
    for path in hexes:
        sites = np.asarray(path, dtype=int)
        signs = np.where(np.arange(len(sites)) % 2 == 0, 1.0, -1.0)
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
    shape = tuple(int(x) for x in sys.argv[1:4])

    orig = geometry.enumerate_ice_states
    geometry.enumerate_ice_states = lambda a, b: np.array([0], dtype=U64)
    cl = geometry.build_cluster("fcc", shape)
    geometry.enumerate_ice_states = orig
    n = int(cl.n_sites)
    positions = np.asarray(cl.positions, dtype=float)
    tets = [tuple(int(s) for s in t) for t in cl.tets]
    hexes = contractible(cl)
    log(f"fcc{shape}: {n} sites, {len(hexes)} hexagons", t0)

    states = enumerate_ice_fast(n, tets, log=lambda m: None)
    ipos = np.rint(positions * 4).astype(np.int64)
    keys = winding_keys(states, ipos)
    uk, cnt = np.unique(keys, return_counts=True)
    hw = hex_words(hexes)
    best = None
    for c in uk[np.argsort(cnt)[::-1][:3]]:
        rows = np.where(keys == c)[0]
        tab = Table128(states[rows])
        H = ring_matrix(tab, hw, t0)
        v = eigsh(H, k=1, which="SA", tol=1e-12, ncv=40)
        if best is None or float(v[0][0]) < best[0] - 1e-9:
            best = (float(v[0][0]), tab, np.ascontiguousarray(v[1][:, 0]))
    e0, tab, gs = best
    del states, keys
    log(f"sector dim {len(tab):,}, E0 = {e0:.8f} K", t0)

    wp = plaquette_w(tab, hw, gs)
    gate = wp.sum() / (-e0)
    log(f"GATE  sum_p w_p / (-E0/K) = {gate:.8f}  "
        f"({'OK' if abs(gate-1) < 1e-6 else 'FAILED'})", t0)

    fam = hex_family(hexes, n)
    subl = np.arange(n) % 4
    momenta, _ = fcc_momenta(cl)
    rows_out = []
    log("\n  momentum                     S        f_1      f_1/S    "
        "per-family geometric factor", t0)
    seen = set()
    for q in momenta:
        qa = np.asarray(q, dtype=float)
        g = geom_factor(hexes, positions, qa, n)
        f1 = float(np.sum(wp * g)) / (2.0 * n)
        # equal-time S(q) from the ground state
        phase = np.exp(-2j * np.pi * (positions @ qa))
        S = 0.0
        for mu in range(4):
            sites = np.where(subl == mu)[0]
            col = np.zeros(len(tab.w), dtype=complex)
            for a in range(0, len(tab.w), CHUNK):
                b = min(a + CHUNK, len(tab.w))
                acc = np.zeros(b - a, dtype=complex)
                for s in sites:
                    acc += (bit_of(tab.w[a:b], int(s)) - 0.5) * phase[s]
                col[a:b] = acc * gs[a:b]
            S += float(np.vdot(col, col).real) / n
        byfam = [float(g[fam == mu].sum()) for mu in range(4)]
        lab = star(q)
        key = (lab, round(S, 6))
        if key in seen:
            continue
        seen.add(key)
        ratio = f1 / S if S > 1e-12 else float("nan")
        log(f"  {lab:<24s} {S:8.5f} {f1:8.5f} {ratio:8.4f}    "
            f"{np.round(byfam, 1)}", t0)
        rows_out.append({"q": [float(x) for x in q], "star": lab,
                         "S": S, "f1": f1, "f1_over_S": ratio,
                         "geom_by_family": byfam})

    tag = "".join(str(s) for s in shape)
    with open(OUT / f"f1_all_fcc{tag}.json", "w") as fh:
        json.dump({"shape": list(shape), "n_sites": n, "E0": e0,
                   "sumrule_ratio": gate, "momenta": rows_out},
                  fh, indent=1, default=float)
    log(f"\nwrote f1_all_fcc{tag}.json", t0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
