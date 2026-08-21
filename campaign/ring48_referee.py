"""Referee-response data: three targeted checks.

1. EVEN-CHANNEL V-SCAN AT X (report R6).  The X-edge state at 2.231K is
   claimed to be two quanta of the L intruder.  Prediction: RK slope
   ~ 2 x (+0.34) = +0.68.  Alternative (two L photons): ~ 2 x (-0.05).
   Track the even-channel edge group at X through V = -0.5..+0.5.
2. PURE RING MODEL ON CUBIC-16 (report R10).  90 ice states: diagonalize
   every transport sector, find the true ground sector and whether any
   nontrivial ring dynamics exists.  Facts, not inference from the
   largest sector.
3. DOMINANT-GROUP DEGENERACIES at V=0 on fcc223 (for result (i) at X):
   dimension of each dominant energy group; with star sizes this decides
   whether the X photon line carries both polarizations.

Also writes scale_photon = 0.586840 (photon-only refit, report R1) into
ring48_gaussian_match.json alongside the old dominant-fit value.
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
from run_ring48_dssf import contractible, ring_matrix           # noqa: E402
from ring48_momenta_fix import fcc_momenta                      # noqa: E402
from ring48_channels import unwrap_hexagons                     # noqa: E402

D = ROOT / "campaign" / "outputs" / "ring_model_dssf"
EDEGEN = 1e-8
VLIST = [-0.5, -0.25, 0.0, 0.25, 0.5]


def sector_eig(cluster, states, labels, v):
    ham, _ = ring_matrix(states, contractible(cluster), v)
    e0, v0 = eigsh(ham, k=1, which="SA", tol=1e-11, ncv=80)
    occ = {int(s): float(np.sum(v0[labels == s, 0] ** 2))
           for s in np.unique(labels)}
    sector = max(occ, key=occ.get)
    rows = np.where(labels == sector)[0]
    values, vectors = eigh(np.asarray(ham[np.ix_(rows, rows)].todense()))
    return sector, rows, values, vectors


def even_columns(cluster, sec_states, gs, momenta):
    index = {int(s): i for i, s in enumerate(sec_states)}
    cols = np.zeros((len(sec_states), len(momenta)), dtype=complex)
    for path_sites, pts, signs, center in unwrap_hexagons(
            cluster, contractible(cluster)):
        mask = 0
        for s_ in path_sites:
            mask |= 1 << int(s_)
        flip_to = np.full(len(sec_states), -1, dtype=np.int64)
        for i, st in enumerate(sec_states):
            bits = [(int(st) >> s_) & 1 for s_ in path_sites]
            if all(bits[j] != bits[(j + 1) % 6] for j in range(6)):
                flip_to[i] = index[int(st) ^ mask]
        src = np.where(flip_to >= 0)[0]
        contrib = np.zeros(len(sec_states))
        contrib[flip_to[src]] = gs[src]
        for qi, momentum in enumerate(momenta):
            cols[:, qi] += np.exp(-2j * np.pi *
                                  np.dot(momentum, center)) * contrib
    return cols


def main() -> int:
    t0 = time.perf_counter()

    # ---------- 2. cubic-16 pure ring model, all sectors -----------------
    cl16 = geometry.build_cluster("cubic", (1, 1, 1))
    st16 = np.sort(np.asarray(cl16.ice_states, dtype=np.uint64))
    lb16 = polarization_sector_labels(cl16)
    ham16, nf16 = ring_matrix(st16, contractible(cl16), 0.0)
    dense16 = np.asarray(ham16.todense())
    print("CUBIC-16 pure ring model (90 ice states, 16 contractible hexes):")
    ground_by_sector = {}
    for sec in np.unique(lb16):
        rows = np.where(lb16 == sec)[0]
        vals = np.linalg.eigvalsh(dense16[np.ix_(rows, rows)])
        ground_by_sector[int(sec)] = (float(vals[0]), len(rows),
                                      float(vals[-1] - vals[0]))
    gmin = min(v[0] for v in ground_by_sector.values())
    gsecs = [s for s, v in ground_by_sector.items() if v[0] - gmin < 1e-9]
    print(f"  global ground energy {gmin:.6f}, in sector(s) {gsecs}")
    for s in sorted(ground_by_sector, key=lambda k: ground_by_sector[k][0])[:6]:
        e, d, bw = ground_by_sector[s]
        print(f"    sector {s}: dim {d}, E0 {e:+.6f}, bandwidth {bw:.6f}")
    nz = sum(1 for v in ground_by_sector.values() if v[2] > 1e-9)
    print(f"  sectors with ANY nontrivial spectrum: {nz} of "
          f"{len(ground_by_sector)}")

    # ---------- fcc223: V=0 degeneracies + even V-scan --------------------
    cl = geometry.build_cluster("fcc", (2, 2, 3))
    states = np.sort(np.asarray(cl.ice_states, dtype=np.uint64))
    labels = polarization_sector_labels(cl)
    momenta, shape = fcc_momenta(cl)
    mlist = list(np.ndindex(*shape))
    xq = mlist.index((1, 1, 0))                       # the X momentum

    edge_track = {}
    for v in VLIST:
        sector, rows, values, vectors = sector_eig(cl, states, labels, v)
        omega = values - values[0]
        cols = even_columns(cl, states[rows], vectors[:, 0], momenta)
        w_even = np.abs(vectors.conj().T @ cols[:, xq]) ** 2
        w_even[0] = 0.0
        groups, start = [], 0
        for i in range(1, len(omega) + 1):
            if i == len(omega) or omega[i] - omega[start] > EDEGEN:
                groups.append((float(omega[start]), slice(start, i)))
                start = i
        gw = np.array([float(w_even[sl].sum()) for _, sl in groups])
        ge = np.array([e for e, _ in groups])
        bright = np.where(gw > 1e-3 * gw.sum())[0]
        edge = bright[np.argmin(ge[bright])]
        top = bright[np.argmax(gw[bright])]
        edge_track[v] = (float(ge[edge]), float(gw[edge] / gw.sum()),
                         float(ge[top]), float(gw[top] / gw.sum()))
        print(f"  V={v:+.2f}: even-channel X edge {ge[edge]:.4f} "
              f"(frac {gw[edge]/gw.sum():.3f}); strongest {ge[top]:.4f} "
              f"(frac {gw[top]/gw.sum():.3f})  "
              f"({time.perf_counter()-t0:.0f} s)", flush=True)
        if v == 0.0:
            # 3. degeneracy of dominant ODD groups per momentum class
            bits = ((states[rows][:, None]
                     >> np.arange(cl.n_sites, dtype=np.uint64))
                    & np.uint64(1)).astype(float) - 0.5
            positions = np.asarray(cl.positions, dtype=float)
            sub = np.arange(int(cl.n_sites)) % 4
            print("  V=0 dominant odd-group degeneracies:")
            for mi, qname in ((mlist.index((0, 0, 1)), "(1/3,1/3,1/3)"),
                              (mlist.index((0, 1, 0)), "L"),
                              (xq, "X")):
                phase = np.exp(-2j * np.pi * (positions @ momenta[mi]))
                wq = np.zeros(len(rows))
                for mu in range(4):
                    col = (bits @ (phase * (sub == mu))) * vectors[:, 0]
                    wq += np.abs(vectors.conj().T @ col) ** 2 / cl.n_sites
                gwq = np.array([float(wq[sl].sum()) for _, sl in groups])
                top2 = np.argsort(gwq)[::-1][:2]
                for g in top2:
                    dim = groups[g][1].stop - groups[g][1].start
                    print(f"    {qname}: omega {ge[g]:.4f} share "
                          f"{gwq[g]/gwq.sum():.3f} group-dim {dim}")

    slope = np.polyfit(VLIST, [edge_track[v][0] for v in VLIST], 1)[0]
    print(f"\nEVEN X-EDGE RK SLOPE = {slope:+.4f}")
    print("  two intruder quanta predict ~ +0.68; two L photons ~ -0.10")

    # ---------- R1: photon-only scale into the match json -----------------
    match = json.load(open(D / "ring48_gaussian_match.json"))
    match["scale_bw_dominant_fit"] = match["scale"]
    match["scale"] = 0.586840
    match["scale_note"] = ("scale refit 2026-08-20 to RK-classified photon "
                           "lines only (1.9802/3, 2.0694/sqrt13, "
                           "1.7300/2sqrt3, 2.5327/sqrt15, 2.2354/4); "
                           "previous dominant-line fit kept as "
                           "scale_bw_dominant_fit")
    with open(D / "ring48_gaussian_match.json", "w") as fh:
        json.dump(match, fh, indent=1, default=float)
    print("scale_photon written:", match["scale"])

    with open(D / "referee_checks.json", "w") as fh:
        json.dump({"cubic16_ground_sectors": ground_by_sector,
                   "even_x_edge_track": edge_track,
                   "even_x_edge_slope": float(slope)}, fh, indent=1,
                  default=float)
    print(f"done ({time.perf_counter()-t0:.0f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
