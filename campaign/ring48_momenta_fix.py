"""Ring-exchange DSSF on fcc(2,2,3) with CORRECT momenta, and a check.

`run_wp0b_gamma_rule.allowed_momenta` is valid only for cubic(1,1,1) and
fcc(2,2,2) -- its docstring says so, and it uses span = max(shape) as one
common range for all three directions.  On the anisotropic 2x2x3 box it
over-generates and returns 24 momenta for 12 primitive cells, which destroys
the Fourier transform: the same energies then carry weight at every momentum.

For an fcc cluster with box vectors Lvecs = shape_i * a_i, the allowed momenta
are simply

    q(m) = m @ inv(Lvecs),      m_i in 0 .. shape_i - 1,

giving exactly prod(shape) inequivalent momenta and no deduplication needed.

VALIDATION, which the previous version lacked: each eigenstate carries a single
momentum, so its S^zz weight must concentrate at ONE q.  The script reports the
worst-case concentration over the states carrying real weight.  If that is not
~1, the momentum basis is still wrong and every number below is void.
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
from qsi_campaign.protocol import polarization_sector_labels    # noqa: E402
from run_momentum_resolved_spectrum import (                    # noqa: E402
    channel_weights, reduce_momentum, star, translation_permutations)
from run_ring48_dssf import contractible, ring_matrix           # noqa: E402

OUT = ROOT / "campaign" / "outputs" / "ring_model_dssf"
EDEGEN = 1e-8


def fcc_momenta(cluster):
    """q(m) = m @ inv(Lvecs), m_i in 0..shape_i-1.  Exactly prod(shape) of them."""
    inverse = np.linalg.inv(np.asarray(cluster.Lvecs, dtype=float))
    shape = tuple(int(s) for s in cluster.shape)
    # q must satisfy q.a_k = m_k/shape_k, i.e. q = m @ inv(L)^T -- the
    # transpose is invisible on isotropic boxes (fcc A is symmetric) and
    # essential on 2x2x3, which is how the untransposed version passed fcc222.
    return np.array([np.asarray(m, dtype=float) @ inverse.T
                     for m in np.ndindex(*shape)]), shape


def main() -> int:
    v_pot, shape_arg = 0.0, (2, 2, 3)
    for token in sys.argv[1:]:
        if token.startswith("--v="):
            v_pot = float(token.split("=", 1)[1])
        elif token.startswith("--shape="):
            shape_arg = tuple(int(v) for v in token.split("=", 1)[1].split(","))
    t0 = time.perf_counter()

    cluster = geometry.build_cluster("fcc", shape_arg)
    n_sites = int(cluster.n_sites)
    states = np.sort(np.asarray(cluster.ice_states, dtype=np.uint64))
    hexes = contractible(cluster)
    labels = polarization_sector_labels(cluster)
    momenta, shape = fcc_momenta(cluster)
    translations, _ = translation_permutations(cluster)
    print(f"fcc{shape}: {n_sites} sites, {len(states):,} ice, {len(hexes)} "
          f"contractible hexes, V/K={v_pot}")
    print(f"  {len(momenta)} momenta vs {len(translations)} lattice "
          f"translations vs {int(np.prod(shape))} primitive cells "
          f"-> {'CONSISTENT' if len(momenta)==len(translations) else 'MISMATCH'}",
          flush=True)

    ham, _ = ring_matrix(states, hexes, v_pot)
    e0, v0 = eigsh(ham, k=1, which="SA", tol=1e-11, ncv=80)
    occ = {int(s): float(np.sum(v0[labels == s, 0] ** 2))
           for s in np.unique(labels)}
    sector = max(occ, key=occ.get)
    rows = np.where(labels == sector)[0]
    values, vectors = eigh(np.asarray(ham[np.ix_(rows, rows)].todense()))
    omega = values - values[0]
    print(f"  sector {sector} dim {len(rows)}, E0={float(e0[0]):.6f}, "
          f"bandwidth {omega[-1]:.4f} ({time.perf_counter()-t0:.0f} s)",
          flush=True)

    positions = np.asarray(cluster.positions, dtype=float)
    total = channel_weights(vectors, states[rows], positions, momenta,
                            n_sites).sum(axis=2)

    # ---- VALIDATION: one eigenstate must carry one momentum ----------------
    per_state = total.sum(axis=1)
    live = per_state > 1e-9
    conc = (total[live].max(axis=1) / per_state[live])
    print(f"\n  momentum concentration over {int(live.sum())} states with "
          f"weight: min {conc.min():.4f}, mean {conc.mean():.4f}")
    if conc.min() < 0.99:
        print("  *** FAILED: states carry weight at several momenta; the "
              "momentum basis is still wrong.  Numbers below are void. ***")
    else:
        print("  PASSED: every state carries a single momentum.")

    groups, start = [], 0
    for i in range(1, len(omega) + 1):
        if i == len(omega) or omega[i] - omega[start] > EDEGEN:
            groups.append((float(omega[start]), slice(start, i)))
            start = i
    ge = np.array([e for e, _ in groups])

    rec, seen = [], set()
    for qi, momentum in enumerate(momenta):
        w = total[:, qi]
        tot = float(w.sum())
        if tot < 1e-12:
            continue
        gw = np.array([float(w[sl].sum()) for _, sl in groups])
        top = int(np.argmax(gw))
        qmag = float(np.linalg.norm(reduce_momentum(momentum)))
        key = round(qmag, 4)
        rec.append({"q": [float(v) for v in momentum], "qmag": qmag,
                    "star": star(momentum), "peak_omega": float(ge[top]),
                    "peak_frac": float(gw[top] / tot),
                    "centroid": float(np.sum(gw * ge) / tot), "total": tot,
                    "first": [float(ge[i]) for i in np.argsort(gw)[::-1][:4]],
                    "firstw": [float(gw[i] / tot)
                               for i in np.argsort(gw)[::-1][:4]]})
        seen.add(key)

    rec.sort(key=lambda r: r["qmag"])
    print("\n   |q|      star        peak_w   frac    centroid  total")
    print("  " + "=" * 62)
    for r in rec:
        print(f"   {r['qmag']:<8.4f} {r['star']:<11s} {r['peak_omega']:<8.4f} "
              f"{r['peak_frac']:<7.4f} {r['centroid']:<9.4f} {r['total']:.5f}")
        print("        " + "  ".join(f"{e:.3f}:{w:.3f}"
                                     for e, w in zip(r["first"], r["firstw"])))
    tag = f"fcc{''.join(str(s) for s in shape)}_v{v_pot:+.2f}".replace(".", "p")
    with open(OUT / f"ring_fixed_{tag}.json", "w") as fh:
        json.dump({"shape": list(shape), "v_over_k": v_pot,
                   "concentration_min": float(conc.min()),
                   "momenta": rec}, fh, indent=1, default=float)
    print(f"\n  wrote ring_fixed_{tag}.json ({time.perf_counter()-t0:.0f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
