"""Census of virtual-path windings of the emergent ring exchanges.

Central point: a ring-exchange process acquires its twist phase from the
BONDS ITS VIRTUAL PATH USES (one perfect matching of the loop into NN
dimer moves), NOT from the homology class of the flipped-site loop.  For a
loop of total winding w the two matchings carry windings N_A and N_B with
N_A + N_B = w, so the coefficient at twist phi is

    c(phi) ~ e^{-i N_A.phi} + e^{-i N_B.phi}
           = e^{-i N_A.phi} (1 + e^{-i w.phi}) / (normalization),

i.e. |c(phi)| ~ |1 + e^{-i w.phi}|: gauge-invariantly, a winding-odd loop's
coupling VANISHES at any corner with w.phi odd (it does not flip sign), and
what survives an OPERATOR-level average is decided by the gauge-dependent
matching windings N_A, N_B -- not by w.

This script tabulates, for the 16-site cubic and 32-site FCC clusters and
each ring channel (4-loops; contractible and wrapping hexagons):
  * the matching windings N of every process (from the exact SW row tables),
  * survival under the 8-corner {0,pi}^3 OPERATOR average (N even),
  * survival under the min-image continuum operator average (N == 0),
  * survival under the zero-net-transport projector delta == 0
    (== smooth-gauge continuum operator average; gauge-invariant).
"""
from __future__ import annotations

import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ice_pt_lib as ipl  # noqa: E402


def census(basis, shape, order=3):
    cl = ipl.build_cluster(basis, shape)
    print(f"\n===== {basis} {shape}: N={cl.n_sites}, ice={cl.n_ice}, "
          f"4-loops={len(cl.loops4)}, hexagons={len(cl.hexes)} =====")
    pt = ipl.sw_effective(cl, 1.0, order=order)
    ice = cl.ice_states

    hex_by_mask = {}
    for path, w in cl.hexes:
        m = 0
        for s in path:
            m |= (1 << s)
        hex_by_mask[m] = tuple(w)
    loop4_by_mask = {}
    for path, w in cl.loops4:
        m = 0
        for s in path:
            m |= (1 << s)
        loop4_by_mask[m] = tuple(w)

    for tag, rows in (("4-loop (order Jpm^2)", pt["H2"]),
                      ("hexagon (order Jpm^3)", pt["H3"])):
        offd = rows["s"] != rows["t"]
        mask = (ice[rows["t"]] ^ ice[rows["s"]]).astype(np.int64)
        delta = ipl.transport_delta(cl, rows)
        stats = defaultdict(lambda: dict(n_loops=0, corner_frac=[], n0_frac=[],
                                         d0_frac=[]))
        seen = set()
        for m in np.unique(mask[offd]):
            m = int(m)
            nbits = bin(m).count("1")
            if tag.startswith("4-loop") and nbits != 4:
                continue
            if tag.startswith("hexagon") and nbits != 6:
                continue
            sel = offd & (mask == m)
            # forward direction only (t > s) to count each element once
            selF = sel & (rows["t"] > rows["s"])
            if not selF.any():
                selF = sel
            c = rows["c"][selF]
            N = rows["N"][selF]
            dl = delta[selF]
            ctot = np.sum(np.abs(c))
            c_even = np.sum(np.abs(c[(N % 2 == 0).all(axis=1)]))
            c_n0 = np.sum(np.abs(c[(N == 0).all(axis=1)]))
            c_d0 = np.sum(np.abs(c[(dl == 0).all(axis=1)]))
            if nbits == 4:
                w = loop4_by_mask.get(m, ("?",))
            else:
                w = hex_by_mask.get(m, None)
                if w is None:
                    continue   # 6-site flip that is not a [111] hexagon
            kind = ("contractible" if w == (0, 0, 0) else "wrapping")
            key = (tag, kind)
            st = stats[key]
            st["n_loops"] += 1 if m not in seen else 0
            seen.add(m)
            st["corner_frac"].append(c_even / ctot if ctot else 0.0)
            st["n0_frac"].append(c_n0 / ctot if ctot else 0.0)
            st["d0_frac"].append(c_d0 / ctot if ctot else 0.0)

        for (t2, kind), st in sorted(stats.items()):
            cf = np.array(st["corner_frac"])
            n0 = np.array(st["n0_frac"])
            d0 = np.array(st["d0_frac"])
            print(f"  {t2:24s} {kind:13s}: masks={st['n_loops']:4d}  "
                  f"|c| kept by 8-corner op-avg: mean={cf.mean():.3f} "
                  f"(min={cf.min():.2f} max={cf.max():.2f})  "
                  f"by n=0: {n0.mean():.3f}  by delta=0: {d0.mean():.3f}")
            hist = Counter(np.round(cf, 3))
            frac_str = ", ".join(f"{k}:{v}" for k, v in sorted(hist.items()))
            print(f"      corner-op-avg survival histogram: {frac_str}")


if __name__ == "__main__":
    census("cubic", (1, 1, 1))
    census("fcc", (2, 2, 2))
