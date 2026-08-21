"""Ring-exchange DSSF on fcc(2,2,3), analysed properly.

Fixes two errors in the first pass:
  * "the peak" was taken as the LOWEST-energy level with any weight, which
    picked up levels carrying <0.1% while the bulk sat elsewhere.  The peak is
    the level with the MOST weight.
  * degenerate multiplets were treated as separate levels.  eigh returns an
    arbitrary basis inside a degenerate multiplet, so per-level weights there
    are not basis independent.  Summing weight over each degenerate energy
    group is.

Reports, per momentum: the dominant peak (energy + fraction), the weight
centroid (a dispersion measure that does not depend on peak identification),
and the top energy groups.  The question is whether the dominant energy rises
with |q| -- a photon -- and how much weight sits away from it.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.linalg import eigh
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import eigsh

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "notes"))
sys.path.insert(0, str(ROOT / "campaign"))

import recompute_finite_size_artifact as geometry               # noqa: E402
from qsi_campaign.protocol import polarization_sector_labels    # noqa: E402
from run_wp0b_gamma_rule import allowed_momenta                 # noqa: E402
from run_momentum_resolved_spectrum import (                    # noqa: E402
    channel_weights, reduce_momentum, star)
from run_ring48_dssf import contractible, ring_matrix           # noqa: E402

OUT = ROOT / "campaign" / "outputs" / "ring_model_dssf"
EDEGEN = 1e-8


def main() -> int:
    v_pot, shape = 0.0, (2, 2, 3)
    for token in sys.argv[1:]:
        if token.startswith("--v="):
            v_pot = float(token.split("=", 1)[1])
        elif token.startswith("--shape="):
            shape = tuple(int(v) for v in token.split("=", 1)[1].split(","))
    t0 = time.perf_counter()

    cluster = geometry.build_cluster("fcc", shape)
    n_sites = int(cluster.n_sites)
    states = np.sort(np.asarray(cluster.ice_states, dtype=np.uint64))
    hexes = contractible(cluster)
    labels = polarization_sector_labels(cluster)
    ham, n_flip = ring_matrix(states, hexes, v_pot)
    e0, v0 = eigsh(ham, k=1, which="SA", tol=1e-11, ncv=80)
    occ = {int(s): float(np.sum(v0[labels == s, 0] ** 2))
           for s in np.unique(labels)}
    sector = max(occ, key=occ.get)
    rows = np.where(labels == sector)[0]
    values, vectors = eigh(np.asarray(ham[np.ix_(rows, rows)].todense()))
    omega = values - values[0]
    print(f"fcc{shape} V/K={v_pot}: sector {sector} dim {len(rows)}, "
          f"E0={float(e0[0]):.6f}, bandwidth {omega[-1]:.4f} "
          f"({time.perf_counter()-t0:.0f} s)", flush=True)

    momenta = allowed_momenta(cluster)
    positions = np.asarray(cluster.positions, dtype=float)
    total = channel_weights(vectors, states[rows], positions, momenta,
                            n_sites).sum(axis=2)

    # collapse exactly-degenerate levels: only group sums are basis independent
    groups, start = [], 0
    for i in range(1, len(omega) + 1):
        if i == len(omega) or omega[i] - omega[start] > EDEGEN:
            groups.append((float(omega[start]), slice(start, i)))
            start = i
    print(f"  {len(omega)} levels -> {len(groups)} distinct energies")

    seen, rec = {}, []
    print("\n   |q|     star            w_peak  frac    centroid  "
          "off-peak   total")
    print("  " + "=" * 74)
    for qi, momentum in enumerate(momenta):
        w = total[:, qi]
        tot = float(w.sum())
        if tot < 1e-12:
            continue
        gw = np.array([float(w[sl].sum()) for _, sl in groups])
        ge = np.array([e for e, _ in groups])
        top = int(np.argmax(gw))
        centroid = float(np.sum(gw * ge) / tot)
        qmag = float(np.linalg.norm(reduce_momentum(momentum)))
        key = round(qmag, 4)
        if key in seen:                       # star partners are identical
            continue
        seen[key] = True
        print(f"   {qmag:<7.4f} {star(momentum):<15s} {ge[top]:<7.4f} "
              f"{gw[top]/tot:<7.4f} {centroid:<9.4f} "
              f"{100*(1-gw[top]/tot):<9.2f} {tot:.5f}")
        order = np.argsort(gw)[::-1][:5]
        print("        top groups: " + "  ".join(
            f"w={ge[i]:.4f}:{gw[i]/tot:.3f}" for i in order if gw[i] > 1e-9))
        rec.append({"q": [float(v) for v in momentum], "qmag": qmag,
                    "star": star(momentum), "peak_omega": float(ge[top]),
                    "peak_frac": float(gw[top] / tot),
                    "centroid": centroid, "total": tot})

    rec.sort(key=lambda r: r["qmag"])
    print("\n  DISPERSION (does the peak rise with |q|?)")
    print("   |q|      peak_omega   centroid   peak_frac")
    for r in rec:
        print(f"   {r['qmag']:<8.4f} {r['peak_omega']:<12.4f} "
              f"{r['centroid']:<10.4f} {r['peak_frac']:.4f}")
    tag = f"fcc{''.join(str(s) for s in shape)}_v{v_pot:+.2f}".replace(".", "p")
    with open(OUT / f"ring_analysis_{tag}.json", "w") as fh:
        json.dump({"shape": list(shape), "v_over_k": v_pot,
                   "sector_dim": len(rows), "momenta": rec}, fh, indent=1,
                  default=float)
    print(f"\n  wrote ring_analysis_{tag}.json ({time.perf_counter()-t0:.0f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
