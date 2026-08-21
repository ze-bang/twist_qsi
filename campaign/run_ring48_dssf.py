"""Ring-exchange photon and its multi-photon weight on the 48-site cluster.

FCC-32 could not answer this: the connected part of its ground sector is
exactly 8-regular, so the uniform state is an exact eigenvector, V is a
constant, and the spectrum is integer and non-dispersive.  fcc(2,2,3) has
n_flippable spanning 6..14 (mean 10.04) over its 9104-state largest transport
sector, and 12 cluster momenta instead of 8.

The model is the pure ring exchange on the ice manifold,

    H = -K sum_hex (W_hex + h.c.) + V sum_hex n_flippable,

over the 48 CONTRACTIBLE hexagons only.  Spinons do not exist in it, and it has
no torus artifact by construction -- wrapping six-cycles are never included.
So any spectral weight away from the one-photon pole here is photon
self-interaction, which is what Huang et al.'s "virtual spinon pair" reading of
a similar residue cannot explain.

Two questions, in order:
  1. DISPERSION.  Does omega(q) grow with |q|?  On FCC-32 it did not
     (omega(X)/omega(L) = 2 against 1.155 for a linear photon), which is the
     tell that the cluster has no photon.  With 12 momenta there is something
     to fit.
  2. WEIGHT.  What fraction of the S^zz weight at each momentum sits on the
     lowest level, and what is off it?

Strategy: sparse Lanczos on the full 87,546-state manifold to locate the ground
state and its transport sector, then DENSE diagonalisation of that sector so
the level-resolved DSSF is exact rather than a continued fraction.  S^z is
diagonal in the Ising basis and so preserves the transport sector; the whole
structure factor lives in that one block.

Usage: run_ring48_dssf.py [--v=VALUE] [--shape=2,2,3]
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

OUT = ROOT / "campaign" / "outputs" / "ring_model_dssf"
OUT.mkdir(parents=True, exist_ok=True)
DEGEN = 1e-9


def contractible(cluster):
    return [[int(s) for s in e[0]] for e in cluster.hexes
            if tuple(int(w) for w in e[1]) == (0, 0, 0)]


def ring_matrix(states, hexes, v_pot):
    """Sparse -K sum(flip) + V sum n_flippable on the ice basis."""
    index = {int(s): i for i, s in enumerate(states)}
    rows, cols, vals = [], [], []
    n_flip = np.zeros(len(states))
    masks = []
    for path in hexes:
        m = 0
        for site in path:
            m |= 1 << site
        masks.append((m, path))
    for i, state in enumerate(states):
        s = int(state)
        for m, path in masks:
            bits = [(s >> t) & 1 for t in path]
            if any(bits[j] == bits[(j + 1) % 6] for j in range(6)):
                continue
            n_flip[i] += 1.0
            j = index[s ^ m]
            rows.append(j); cols.append(i); vals.append(-1.0)
    rows.extend(range(len(states)))
    cols.extend(range(len(states)))
    vals.extend(v_pot * n_flip)
    mat = coo_matrix((vals, (rows, cols)),
                     shape=(len(states), len(states))).tocsr()
    return mat, n_flip


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
    print(f"fcc{shape}: {n_sites} sites, {len(states):,} ice states, "
          f"{len(hexes)} contractible hexagons, V/K={v_pot}", flush=True)

    ham, n_flip = ring_matrix(states, hexes, v_pot)
    print(f"  H built: {ham.nnz:,} nonzeros, <n_flip>={n_flip.mean():.3f}, "
          f"range {int(n_flip.min())}..{int(n_flip.max())} "
          f"({time.perf_counter()-t0:.0f} s)", flush=True)

    e0, v0 = eigsh(ham, k=1, which="SA", tol=1e-11, ncv=80)
    e0 = float(e0[0])
    occ = {int(s): float(np.sum(v0[labels == s, 0] ** 2))
           for s in np.unique(labels)}
    sector = max(occ, key=occ.get)
    rows = np.where(labels == sector)[0]
    print(f"  ground energy {e0:.9f}; sits {100*occ[sector]:.4f}% in transport "
          f"sector {sector} (dim {len(rows)})", flush=True)
    print(f"  E0/(-K) = {-e0:.6f} vs <n_flip> of that sector "
          f"{n_flip[rows].mean():.3f} -> "
          f"{'UNIFORM/RK-like' if abs(-e0 - n_flip[rows].mean()) < 1e-6 else 'genuinely correlated'}",
          flush=True)

    block = np.asarray(ham[np.ix_(rows, rows)].todense())
    print(f"  dense-diagonalising {len(rows)}x{len(rows)} block...", flush=True)
    values, vectors = eigh(block)
    print(f"  done ({time.perf_counter()-t0:.0f} s); ground degeneracy "
          f"{int(np.sum(values - values[0] < DEGEN))}, bandwidth "
          f"{values[-1]-values[0]:.6f}", flush=True)

    momenta = allowed_momenta(cluster)
    positions = np.asarray(cluster.positions, dtype=float)
    weights = channel_weights(vectors, states[rows], positions, momenta,
                              n_sites)
    total = weights.sum(axis=2)

    print(f"\n  {len(momenta)} momenta.  S^zz weight and dispersion:")
    print("  " + "=" * 72)
    print("   |q|     star     omega_low    weight_low   total    "
          "frac_low  off-pole")
    record = {"shape": list(shape), "n_sites": n_sites,
              "n_ice": len(states), "v_over_k": v_pot, "sector": sector,
              "sector_dim": len(rows), "ground_energy": e0,
              "bandwidth": float(values[-1] - values[0]), "momenta": []}
    for qi, momentum in enumerate(momenta):
        w = total[:, qi]
        if w.sum() < 1e-12:
            continue
        # group exactly-degenerate levels so partners are not counted as off-pole
        bright = np.where(w > 1e-10)[0]
        lowest_e = values[bright].min()
        same = bright[values[bright] - lowest_e < DEGEN]
        w_low = float(w[same].sum())
        tot = float(w.sum())
        qmag = float(np.linalg.norm(reduce_momentum(momentum)))
        print(f"   {qmag:<7.4f} {star(momentum):<8s} "
              f"{lowest_e-values[0]:<12.6f} {w_low:<12.6f} {tot:<8.5f} "
              f"{w_low/tot:<9.4f} {100*(1-w_low/tot):.2f}%")
        record["momenta"].append(
            {"q": [float(v) for v in momentum], "qmag": qmag,
             "star": star(momentum), "omega_low": float(lowest_e - values[0]),
             "n_degenerate": int(len(same)), "weight_low": w_low,
             "total": tot, "off_pole": float(1 - w_low / tot)})

    tag = f"fcc{''.join(str(s) for s in shape)}_v{v_pot:+.2f}".replace(".", "p")
    with open(OUT / f"ring_dssf_{tag}.json", "w") as fh:
        json.dump(record, fh, indent=1, default=float)
    print(f"\n  wrote {OUT / f'ring_dssf_{tag}.json'} "
          f"({time.perf_counter()-t0:.0f} s)")
    print("  Read: does omega_low rise with |q| (a photon), and is off-pole")
    print("  weight comparable to the ~10% the masked XXZ fold showed?")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
