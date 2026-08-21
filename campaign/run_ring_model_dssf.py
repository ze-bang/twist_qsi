"""S^zz of the PURE ring-exchange model: is the off-pole weight photons or spinons?

The masked-fold calculation finds ~10% of the longitudinal weight at L sitting
off the single-photon level.  Two readings:

  (a) photon self-interaction.  The gauge field is compact, so cos(curl A)
      contains a quartic photon-photon vertex; a linear probe (S^z = E) can
      leak weight into three-photon states.  Kwasigroch, Doucot & Castelnovo
      (PRB 95, 134439) argued this vanishes kinematically -- but only in the
      RELATIVISTIC part of the spectrum.  FCC-32 has no small-q momenta at all;
      X and L are both zone boundary, where the band is flat and the argument
      lapses.
  (b) virtual spinon pairs.  Huang, Deng, Wan & Meng (PRL 120, 167202) ascribe
      a residue of the same size in THEIR longitudinal channel to "virtual
      spinon pairs, which temporarily violate the ice rule".  Our masked fold
      is the full XXZ with a depth-1 virtual space: the flux-diagnostic file
      records non_ice_weight = 0.360 for the very state we call the photon at
      Jpm = -0.200.  So (b) is live.

This script settles it.  The pure ring-exchange model

    H = -K sum_hex (W_hex + h.c.)  +  V sum_hex n_flippable

lives entirely inside the 2970-state ice manifold.  Spinons do not exist in it.
Whatever off-pole weight survives here is photon self-interaction, full stop.
If the weight collapses instead, our 10% was spinon admixture and Huang et al.
already explained it.

Only CONTRACTIBLE hexagons enter: 96 of the 128 six-cycles on FCC-32 wrap the
periodic box, and their ring exchange transports polarization between sectors,
which is the torus artifact.  Filtering on winding == (0,0,0) leaves 32.

Usage: run_ring_model_dssf.py [--v=VALUE] [--cluster=fcc32]
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.linalg import eigh

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "notes"))
sys.path.insert(0, str(ROOT / "campaign"))

import recompute_finite_size_artifact as geometry               # noqa: E402
from qsi_campaign.protocol import polarization_sector_labels    # noqa: E402
from run_wp0b_gamma_rule import allowed_momenta                 # noqa: E402
from run_momentum_resolved_spectrum import (                    # noqa: E402
    channel_weights, star)

OUT = ROOT / "campaign" / "outputs" / "ring_model_dssf"
OUT.mkdir(parents=True, exist_ok=True)
CLUSTERS = {"fcc32": ("fcc", (2, 2, 2)), "cubic16": ("cubic", (1, 1, 1))}
DEGEN = 1e-9


def contractible_hexagons(cluster):
    """Paths of the six-cycles that do NOT wrap the periodic box."""
    out = []
    for entry in cluster.hexes:
        path, winding = entry[0], entry[1]
        if tuple(int(w) for w in winding) == (0, 0, 0):
            out.append([int(s) for s in path])
    return out


def build_ring_hamiltonian(states, hexes, n_sites, k_ring=1.0, v_pot=0.0):
    """-K sum (|flipped><ice| + h.c.) + V sum n_flippable, on the ice basis.

    A hexagon is flippable when the spins alternate around its cyclic path;
    the flip is then an XOR with the hexagon mask, and it maps ice -> ice.
    """
    index = {int(s): i for i, s in enumerate(states)}
    dim = len(states)
    ham = np.zeros((dim, dim))
    n_flip = np.zeros(dim)
    moves = 0
    for path in hexes:
        mask = 0
        for site in path:
            mask |= 1 << site
        for i, state in enumerate(states):
            bits = [(int(state) >> s) & 1 for s in path]
            if any(bits[j] == bits[(j + 1) % 6] for j in range(6)):
                continue                       # not alternating: not flippable
            n_flip[i] += 1.0
            j = index.get(int(state) ^ mask)
            if j is None:
                raise SystemExit("hexagon flip left the ice manifold")
            ham[j, i] -= k_ring
            moves += 1
    ham[np.diag_indices(dim)] += v_pot * n_flip
    return ham, n_flip, moves


def main() -> int:
    v_pot, name = 0.0, "fcc32"
    for token in sys.argv[1:]:
        if token.startswith("--v="):
            v_pot = float(token.split("=", 1)[1])
        elif token.startswith("--cluster="):
            name = token.split("=", 1)[1]
    started = time.perf_counter()

    cluster = geometry.build_cluster(*CLUSTERS[name])
    n_sites = int(cluster.n_sites)
    states = np.sort(np.asarray(cluster.ice_states, dtype=np.uint64))
    hexes = contractible_hexagons(cluster)
    labels = polarization_sector_labels(cluster)
    print(f"{name}: {n_sites} sites, {len(states)} ice states, "
          f"{len(cluster.hexes)} six-cycles of which {len(hexes)} contractible",
          flush=True)

    ham, n_flip, moves = build_ring_hamiltonian(states, hexes, n_sites,
                                                v_pot=v_pot)
    print(f"  V/K = {v_pot}; {moves} directed ring moves; "
          f"<n_flippable> = {n_flip.mean():.3f} of {len(hexes)} "
          f"({n_flip.mean()/len(hexes):.3f}); {int((n_flip==0).sum())} frozen "
          f"states", flush=True)

    values, vectors = eigh(ham)
    gs_degen = int(np.sum(values - values[0] < DEGEN))
    print(f"  ground energy {values[0]:.10f}, degeneracy {gs_degen}, "
          f"bandwidth {values[-1]-values[0]:.6f}", flush=True)

    # which transport sector holds the ground state
    weight_by_sector = {}
    for sec in np.unique(labels):
        weight_by_sector[int(sec)] = float(np.sum(np.abs(
            vectors[labels == sec, 0]) ** 2))
    top = max(weight_by_sector, key=weight_by_sector.get)
    print(f"  ground state sits {100*weight_by_sector[top]:.4f}% in transport "
          f"sector {top} (dim {int((labels==top).sum())})", flush=True)

    momenta = allowed_momenta(cluster)
    positions = np.asarray(cluster.positions, dtype=float)
    weights = channel_weights(vectors, states, positions, momenta, n_sites)
    total = weights.sum(axis=2)                    # (n_states, n_momenta)

    print("\n  S^zz weight distribution, pure ring-exchange model")
    print("  " + "=" * 66)
    record = {"cluster": name, "v_over_k": v_pot, "n_ice": len(states),
              "n_hexagons": len(hexes), "ground_degeneracy": gs_degen,
              "bandwidth": float(values[-1] - values[0]), "stars": {}}
    for qi, momentum in enumerate(momenta):
        label = star(momentum)
        w = total[:, qi]
        order = np.argsort(w)[::-1]
        keep = [i for i in order if w[i] > 1e-10][:6]
        if not keep:
            continue
        tot = float(w.sum())
        first = keep[0]
        print(f"\n  q = {[round(float(v),3) for v in momentum]}  ({label})"
              f"   total weight {tot:.6f}")
        print(f"     omega        weight      frac    cumulative")
        cum = 0.0
        rows = []
        for i in keep:
            cum += float(w[i])
            print(f"     {values[i]-values[0]:<11.6f} {w[i]:<11.6f} "
                  f"{w[i]/tot:<7.4f} {cum/tot:.4f}")
            rows.append({"omega": float(values[i] - values[0]),
                         "weight": float(w[i]), "frac": float(w[i] / tot)})
        off = 1.0 - float(w[first]) / tot
        print(f"     -> lowest-level fraction {w[first]/tot:.4f}, "
              f"OFF-POLE {100*off:.2f}%")
        record["stars"].setdefault(label, []).append(
            {"q": [float(v) for v in momentum], "total": tot,
             "off_pole_fraction": off, "levels": rows})

    record["runtime"] = time.perf_counter() - started
    tag = f"v{v_pot:+.2f}".replace(".", "p")
    with open(OUT / f"ring_dssf_{name}_{tag}.json", "w") as fh:
        json.dump(record, fh, indent=1, default=float)
    print(f"\n  wrote {OUT / f'ring_dssf_{name}_{tag}.json'} "
          f"({record['runtime']:.0f} s)")
    print("\n  COMPARE: the masked XXZ fold gives ~10% off-pole at L.")
    print("  If this is also ~10%, it is photon self-interaction.")
    print("  If it collapses toward 0, the fold's 10% was virtual spinons.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
