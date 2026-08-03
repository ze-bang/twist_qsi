#!/usr/bin/env python3
"""Self-consistent winding-free band on FCC-32 by multi-anchor interpolation.

The failed estimators (fixed anchor: spacings off by ~1/Z; naive
nearest-anchor: unstable near the descended continuum) are replaced by the
validated route: sample the masked map's sorted eigenvalue branches at ~10
anchor energies spanning [floor, continuum edge], interpolate each strictly
decreasing g_n(z) = lambda_n(z) - z with PCHIP, and read off every root.
Validated on cubic-16 truncated blocks against exact self-consistency:
all roots found, rms ~5e-4, ring peak to 0.3-1% at -0.20..-0.30
(scratchpad validate_multi_anchor.py, Aug-02).

Each F(z) evaluation is one conjugate-gradient pass over the ice columns;
this script vectorizes CG across columns (chunked, per-column scalars) and
warm-starts each anchor from the previous one, since the resolvent changes
smoothly along the anchor ladder.

Usage: run_multi_anchor_band.py <cubic16|fcc32> <levels> <jpm> [<jpm> ...]
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq
from scipy.signal import find_peaks
from scipy.sparse.linalg import eigsh

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "notes"))
sys.path.insert(0, str(ROOT / "campaign"))

import recompute_finite_size_artifact as geometry  # noqa: E402
from qsi_campaign.protocol import polarization_sector_labels  # noqa: E402
from run_truncated_feshbach import CLUSTERS, bfs_space, build_hamiltonian  # noqa: E402

OUTPUT = ROOT / "campaign" / "outputs" / "multi_anchor_band"
OUTPUT.mkdir(parents=True, exist_ok=True)

N_ANCHOR = 8
CHUNK = 256
GRID = np.geomspace(1e-5, 0.3, 3000)


def block_cg(matvec, rhs, x0, tol=1e-9, max_iter=25000):
    """Vectorized CG for one SPD operator and many right-hand sides."""
    x = x0.copy()
    r = rhs - matvec(x)
    p = r.copy()
    rr = np.einsum("ij,ij->j", r, r)
    norm = np.sqrt(np.einsum("ij,ij->j", rhs, rhs))
    norm[norm == 0.0] = 1.0
    active = np.sqrt(rr) / norm > tol
    for _ in range(max_iter):
        if not active.any():
            break
        ap = matvec(p)
        alpha = np.where(active, rr / np.einsum("ij,ij->j", p, ap), 0.0)
        x += alpha * p
        r -= alpha * ap
        rr_new = np.einsum("ij,ij->j", r, r)
        beta = np.where(active, rr_new / np.where(rr > 0, rr, 1.0), 0.0)
        p = r + beta * p
        rr = rr_new
        active = np.sqrt(rr) / norm > tol
    else:
        raise RuntimeError("block CG did not converge")
    return x


def band_peaks(levels):
    shifted = np.sort(levels) - np.min(levels)
    beta = 1.0 / GRID[:, None]
    boltzmann = np.exp(-beta * shifted[None, :])
    partition = boltzmann.sum(axis=1)
    first = (boltzmann @ shifted) / partition
    second = (boltzmann @ shifted**2) / partition
    heat = (second - first**2) / GRID**2
    index, _ = find_peaks(heat, prominence=0.005 * heat.max())
    return GRID[index]


def main() -> None:
    name = sys.argv[1]
    levels_arg = int(sys.argv[2])
    couplings = [float(v) for v in sys.argv[3:]]
    basis, shape = CLUSTERS[name]
    cluster = geometry.build_cluster(basis, shape)
    space = bfs_space(cluster, levels_arg)
    ice_states = np.sort(np.asarray(cluster.ice_states, dtype=np.uint64))
    ice_rows = np.searchsorted(space, ice_states)
    complement = np.setdiff1d(np.arange(len(space)), ice_rows)
    order = np.argsort(np.asarray(cluster.ice_states, dtype=np.uint64))
    labels = polarization_sector_labels(cluster)[order]
    same = labels[:, None] == labels[None, :]
    d = len(ice_states)
    print(f"{name} L={levels_arg}: space {len(space):,}, {d} ice", flush=True)

    for jpm in couplings:
        started = time.perf_counter()
        hamiltonian = build_hamiltonian(cluster, space, jpm).tocsr()
        raw_ground = float(eigsh(hamiltonian, k=1, which="SA", tol=1e-9,
                                 return_eigenvectors=False)[0])
        q_block = hamiltonian[complement][:, complement].tocsr()
        edge = float(eigsh(q_block, k=1, which="SA", tol=1e-9,
                           return_eigenvectors=False)[0])
        coupling = hamiltonian[complement][:, ice_rows].tocsc()
        coupling_t = coupling.T.tocsr()
        ice_diag = hamiltonian[ice_rows][:, ice_rows].toarray()

        floor = raw_ground - 0.05 * abs(raw_ground) - 0.02
        top = edge - 0.02 * (edge - floor)
        anchors = list(np.linspace(floor, top, N_ANCHOR))
        samples: dict[float, np.ndarray] = {}

        def f_eigenvalues(z_list):
            """Masked-map eigenvalues at several anchors in one sweep.

            Chunk-outer, anchor-inner (ascending), warm-starting each
            anchor's CG from the previous anchor's solution -- the warm
            block lives only for the current chunk, so memory stays flat
            regardless of anchor count or space size.
            """
            z_sorted = sorted(z_list)
            f_parts = {z: ice_diag.copy() for z in z_sorted}
            for start in range(0, d, CHUNK):
                cols = slice(start, min(start + CHUNK, d))
                rhs = np.asarray(coupling[:, cols].todense())
                solution = np.zeros_like(rhs)
                for z in z_sorted:
                    solution = block_cg(
                        lambda v, shift=z: q_block @ v - shift * v,
                        rhs, solution)
                    f_parts[z][:, cols] -= coupling_t @ solution
            for z in z_sorted:
                f_matrix = 0.5 * (f_parts[z] + f_parts[z].T)
                samples[z] = np.linalg.eigvalsh(np.where(same, f_matrix, 0.0))

        def roots_from(anchor_list):
            zs = np.array(sorted(anchor_list))
            table = np.vstack([samples[z] for z in zs])
            gaps = table - zs[:, None]
            out = np.full(d, np.nan)
            for branch in range(d):
                g = gaps[:, branch]
                if g[-1] > 0 or g[0] <= 0:
                    continue
                interp = PchipInterpolator(zs, g)
                low = zs[np.flatnonzero(g > 0)[-1]]
                high = zs[np.flatnonzero(g <= 0)[0]]
                out[branch] = brentq(interp, low, high, xtol=1e-13)
            return out

        f_eigenvalues(anchors)
        print(f"  {len(anchors)} anchors done "
              f"({time.perf_counter() - started:.0f}s)", flush=True)
        roots = roots_from(anchors)
        finite = roots[np.isfinite(roots)]
        refinement = [float(v) for v in np.percentile(finite, [30.0, 70.0])
                      if all(abs(v - z) > 1e-6 for z in anchors)]
        if refinement:
            f_eigenvalues(refinement)
            anchors.extend(refinement)
        roots = roots_from(anchors)
        found = np.sort(roots[np.isfinite(roots)])

        peak_list = band_peaks(found)
        g6 = 12.0 * abs(jpm) ** 3
        record = {
            "cluster": name, "levels": levels_arg, "jpm": jpm,
            "space": len(space), "n_found": int(len(found)),
            "n_escaped": int(d - len(found)),
            "raw_ground": raw_ground, "edge": edge,
            "band_bottom": float(found[0]) if len(found) else None,
            "band_width": float(found[-1] - found[0]) if len(found) else None,
            "peaks": peak_list.tolist(),
            "ring_peak_over_g6": (float(min(peak_list)) / g6
                                  if len(peak_list) else None),
            "n_anchors": len(anchors),
            "runtime": time.perf_counter() - started,
        }
        tag = f"{name}_L{levels_arg}_{jpm:+.3f}".replace(".", "p")
        with open(OUTPUT / f"band_{tag}.json", "w") as handle:
            json.dump(record, handle, indent=1, default=float)
        np.savez_compressed(OUTPUT / f"band_{tag}.npz", jpm=jpm,
                            roots=found, anchors=np.array(sorted(anchors)))
        print(f"{name} L{levels_arg} jpm={jpm:+.3f} "
              f"found {len(found)}/{d} (+{record['n_escaped']} escaped) "
              f"peaks {np.round(peak_list, 5)} "
              f"ring/g6 {record['ring_peak_over_g6']} "
              f"({record['runtime']:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
