"""Both parity channels of the 48-site ring model + the analytic Gaussian
dispersion along the Gamma-L-X path (which contains ALL five inequivalent
cluster momenta).

Odd channel  (neutron): S^z(q)     -> 1-photon + 3-photon...
Even channel (Raman)  : Phi(q) = sum_p e^{-iq.r_p}(W_p + W_p^+) -> 2-photon...
Parity routing is exact; each eigenstate is bright in at most one channel.

Also: the momentum-space curl C(k) built analytically from one hexagon per
translation class (unwrapped geometry), giving sigma_lambda(k) at ARBITRARY k.
GATE: sigma(k) at the five commensurate momenta must reproduce the cluster
values (3, 2sqrt3, sqrt13, sqrt15, 4) to 1e-6, else everything is void.
Plus Gaussian 2-gamma and 3-gamma kinematic envelopes along the path.

Output: ring48_channels.npz
"""

from __future__ import annotations

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
    channel_weights, reduce_momentum, star)
from run_ring48_dssf import contractible, ring_matrix           # noqa: E402
from ring48_momenta_fix import fcc_momenta                      # noqa: E402

OUT = ROOT / "campaign" / "outputs" / "ring_model_dssf"
EDEGEN = 1e-8


def unwrap_hexagons(cluster, hexes):
    """Unwrapped site positions, alternating signs, and center per hexagon."""
    P = np.asarray(cluster.positions, dtype=float)
    L = np.asarray(cluster.Lvecs, dtype=float)
    Linv = np.linalg.inv(L)
    out = []
    for path in hexes:
        pts = [P[path[0]].copy()]
        for a, b in zip(path, path[1:]):
            d = P[b] - P[a]
            d -= np.round(d @ Linv) @ L          # minimum image step
            pts.append(pts[-1] + d)
        pts = np.array(pts)                       # 6 unwrapped positions
        signs = np.array([1.0 if j % 2 == 0 else -1.0 for j in range(6)])
        out.append((path, pts, signs, pts.mean(axis=0)))
    return out


def analytic_blocks(cluster, unwrapped):
    """One representative hexagon per translation class; returns a function
    C(k) -> (4 x 4 complex) with rows = hexagon classes, cols = sublattices."""
    A = np.diag(1.0 / np.asarray(cluster.shape, float)) @ \
        np.asarray(cluster.Lvecs, float)          # primitive lattice matrix
    Ainv = np.linalg.inv(A)
    reps, seen = [], set()
    for path, pts, signs, center in unwrapped:
        frac = tuple(np.round((center @ Ainv) % 1.0, 4))
        if frac in seen:
            continue
        seen.add(frac)
        reps.append((path, pts, signs, center))
    if len(reps) != 4:
        raise SystemExit(f"expected 4 hexagon classes, got {len(reps)}")

    sub = lambda i: i % 4

    def block(k):
        C = np.zeros((4, 4), dtype=complex)
        for row, (path, pts, signs, center) in enumerate(reps):
            for site, pt, s in zip(path, pts, signs):
                C[row, sub(site)] += s * np.exp(2j * np.pi * np.dot(k, pt - center))
        return C

    return block


def sigmas(block, k):
    s = np.linalg.svd(block(k), compute_uv=False)
    return np.sort(s[s > 1e-9])


def main() -> int:
    t0 = time.perf_counter()
    cluster = geometry.build_cluster("fcc", (2, 2, 3))
    n = int(cluster.n_sites)
    states = np.sort(np.asarray(cluster.ice_states, dtype=np.uint64))
    labels = polarization_sector_labels(cluster)
    hexes = contractible(cluster)
    unwrapped = unwrap_hexagons(cluster, hexes)
    block = analytic_blocks(cluster, unwrapped)

    # ---- GATE: analytic sigma(k) vs the cluster values ------------------
    checks = {(1/3, 1/3, 1/3): 3.0, (0.5, 0.5, 0.5): 2 * np.sqrt(3),
              (1/3, 1/3, 2/3): np.sqrt(13), (1/6, 1/6, 5/6): np.sqrt(15),
              (0.0, 0.0, 1.0): 4.0}
    worst = 0.0
    for k, ref in checks.items():
        s = sigmas(block, np.array(k))
        err = abs(s[-1] - ref)
        worst = max(worst, err)
        print(f"  sigma{k} = {s[-1]:.6f}  (expect {ref:.6f}, "
              f"n_branches={len(s)})")
    print(f"analytic-curl gate: worst |err| = {worst:.2e} -> "
          f"{'PASS' if worst < 1e-6 else 'FAIL'}")
    if worst >= 1e-6:
        return 1

    # ---- Gaussian dispersion + 2g/3g envelopes on the Gamma-L-X path ----
    Lpt, Xpt = np.array([0.5, 0.5, 0.5]), np.array([0.0, 0.0, 1.0])
    path = [t * Lpt for t in np.linspace(0, 1, 41)] + \
           [Lpt + t * (Xpt - Lpt) for t in np.linspace(0, 1, 41)[1:]]
    path = np.array(path)
    disp = np.array([sigmas(block, k)[-1] if len(sigmas(block, k)) else 0.0
                     for k in path])
    # kinematic envelopes on a commensurate grid (Gaussian energies)
    G = 12
    grid = np.array([np.array(m) / G for m in np.ndindex(G, G, G)])
    wgrid = np.array([sigmas(block, k)[-1] if len(sigmas(block, k)) else 0.0
                      for k in grid])
    env2, env3 = [], []
    for q in path:
        # omega(k) + omega(q - k): index arithmetic on the grid
        wq = np.array([wgrid[np.argmin(np.linalg.norm(
            (grid - ((q - k) % 1.0) + 0.5) % 1.0 - 0.5, axis=1))]
            for k in grid[::7]])                  # subsample for speed
        tot2 = wgrid[::7] + wq
        env2.append((tot2.min(), tot2.max()))
        tot3 = (wgrid[::7, None] + wgrid[None, ::7]).ravel()
        env3.append(tot3.min() + wgrid.min())     # crude lower edge only
    env2 = np.array(env2)
    env3 = np.array(env3)
    print(f"path + envelopes done ({time.perf_counter()-t0:.0f} s)", flush=True)

    # ---- exact ED, both channels ----------------------------------------
    ham, _ = ring_matrix(states, hexes, 0.0)
    e0, v0 = eigsh(ham, k=1, which="SA", tol=1e-11, ncv=80)
    occ = {int(s): float(np.sum(v0[labels == s, 0] ** 2))
           for s in np.unique(labels)}
    sector = max(occ, key=occ.get)
    rows = np.where(labels == sector)[0]
    values, vectors = eigh(np.asarray(ham[np.ix_(rows, rows)].todense()))
    omega = values - values[0]
    print(f"sector {sector} dim {len(rows)} diagonalized "
          f"({time.perf_counter()-t0:.0f} s)", flush=True)

    momenta, shape = fcc_momenta(cluster)
    positions = np.asarray(cluster.positions, dtype=float)
    odd = channel_weights(vectors, states[rows], positions, momenta,
                          n).sum(axis=2)          # (n_states, 12)

    # even channel: Phi(q)|0> with W_p site-resolved
    sec_states = states[rows]
    index = {int(s): i for i, s in enumerate(sec_states)}
    gs = vectors[:, 0]
    phi_cols = np.zeros((len(rows), len(momenta)), dtype=complex)
    for path_sites, pts, signs, center in unwrapped:
        mask = 0
        for s_ in path_sites:
            mask |= 1 << int(s_)
        # W_p + W_p^dagger : flip every flippable configuration
        flip_to = np.full(len(rows), -1, dtype=np.int64)
        for i, st in enumerate(sec_states):
            bits = [(int(st) >> s_) & 1 for s_ in path_sites]
            if all(bits[j] != bits[(j + 1) % 6] for j in range(6)):
                flip_to[i] = index[int(st) ^ mask]
        src = np.where(flip_to >= 0)[0]
        contrib = np.zeros(len(rows))
        contrib[flip_to[src]] = gs[src]           # (W+W^+)|0>, this plaquette
        for qi, momentum in enumerate(momenta):
            phi_cols[:, qi] += np.exp(-2j * np.pi *
                                      np.dot(momentum, center)) * contrib
    even = np.abs(vectors.conj().T @ phi_cols) ** 2   # (n_states, 12)
    # remove the elastic (ground-state) term
    even[0, :] = 0.0

    # ---- parity complementarity check -----------------------------------
    both = np.minimum(odd.sum(axis=1), even.sum(axis=1))
    live = (odd.sum(axis=1) + even.sum(axis=1)) > 1e-10
    overlap = float(both[live].max()) if live.any() else 0.0
    print(f"parity complementarity: max min(odd,even) over live states = "
          f"{overlap:.3e}  -> {'PASS' if overlap < 1e-10 else 'CHECK'}")

    # ---- group and save --------------------------------------------------
    groups, start = [], 0
    for i in range(1, len(omega) + 1):
        if i == len(omega) or omega[i] - omega[start] > EDEGEN:
            groups.append((float(omega[start]), slice(start, i)))
            start = i
    ge = np.array([e for e, _ in groups])
    g_odd = np.array([[odd[sl, qi].sum() for qi in range(len(momenta))]
                      for _, sl in groups])
    g_even = np.array([[even[sl, qi].sum() for qi in range(len(momenta))]
                       for _, sl in groups])

    np.savez(OUT / "ring48_channels.npz",
             group_energies=ge, odd=g_odd, even=g_even,
             momenta=np.asarray(momenta),
             m_indices=np.array(list(np.ndindex(*shape))),
             qreduced=np.array([reduce_momentum(m) for m in momenta]),
             stars=np.array([star(m) for m in momenta]),
             path=path, path_sigma=disp, env2=env2, env3=env3,
             parity_overlap=overlap)
    print(f"saved ring48_channels.npz ({time.perf_counter()-t0:.0f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
