#!/usr/bin/env python3
"""Winding-free C(T) peaks across the (Jpm, Jpmpm) plane on cubic-16.

Motivation: in pure XXZ the winding-free spinon/ring peak separation is
O(100) at every coupling, but the QSI candidates show either one peak
(Ce2Zr2O7, Jpm ~ -0.30) or two peaks a factor ~3 apart (Ce2Hf2O7,
65 mK / 24 mK).  The missing dial is Jpmpm.  The old frame-based scan
(material_scan.solve_point) went ill-defined on most of the strong-mixing
region; this scanner replaces the frame with the two instruments that
survive it:

  raw side   : full 65,536-state spectrum WITH eigenvectors from the eight
               Klein-translation x Sz-parity blocks, giving every level's
               ice weight Z_n (Jpmpm preserves Sz parity, not Sz);
  band side  : masked multi-anchor self-consistent roots on the full
               space (90 ice columns, sparse CG -- exact, no truncation,
               no frame, valid at any coupling);
  fold       : remove the 90 highest-Z_n raw levels, insert the masked
               roots; all local maxima of the full-ensemble C(T).

At Jpmpm = 0 the masked roots must reproduce the masked_band_complete
cache exactly (free oracle, checked when available).  Jpmpm -> -Jpmpm is
a symmetry; scan Jpmpm >= 0.

Usage: run_jpmpm_fold_plane.py <jpm> <jpmpm> [<jpm> <jpmpm> ...]
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.linalg import eigh
from scipy.optimize import brentq
from scipy.signal import find_peaks
from scipy.sparse.linalg import eigsh

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "notes"))
sys.path.insert(0, str(ROOT / "campaign"))

import recompute_finite_size_artifact as geometry  # noqa: E402
from qsi_campaign.exact_band import (  # noqa: E402
    full_spin_basis,
    microscopic_character_hamiltonian,
    translation_sector_bases,
    cubic16_translation_permutations,
)
from qsi_campaign.protocol import polarization_sector_labels  # noqa: E402
from run_multi_anchor_band import block_cg  # noqa: E402

OUTPUT = ROOT / "campaign" / "outputs" / "jpmpm_fold_plane"
OUTPUT.mkdir(parents=True, exist_ok=True)
BAND_DIR = ROOT / "campaign" / "outputs" / "masked_band_complete"

N_ANCHOR = 8
GRID = np.geomspace(1e-5, 1.5, 1200)


def all_peaks(levels: np.ndarray) -> np.ndarray:
    shifted = np.sort(levels) - np.min(levels)
    beta = 1.0 / GRID[:, None]
    boltzmann = np.exp(-beta * shifted[None, :])
    partition = boltzmann.sum(axis=1)
    first = (boltzmann @ shifted) / partition
    second = (boltzmann @ shifted**2) / partition
    heat = (second - first**2) / GRID**2
    index, _ = find_peaks(heat, prominence=0.005 * heat.max())
    return GRID[index]


def raw_side(hamiltonian, states, sectors, ice_indices):
    """All 65,536 levels with ice weights, from the eight symmetry blocks."""
    parity = (np.bitwise_count(states.astype(np.uint64)) % 2).astype(np.int64)
    levels, weights = [], []
    for basis in sectors.values():
        projected = (basis.conj().T @ (hamiltonian @ basis)).toarray()
        if np.abs(projected.imag).max(initial=0.0) > 1e-10:
            raise RuntimeError("translation block is not real")
        basis = basis.tocsc()
        column_parity = np.array(
            [parity[basis.indices[basis.indptr[c]]]
             for c in range(basis.shape[1])])
        ice_basis = basis[ice_indices, :]
        for value in (0, 1):
            mask = column_parity == value
            block = projected.real[np.ix_(mask, mask)]
            vals, vecs = eigh(block)
            amplitude = np.asarray(ice_basis[:, mask] @ vecs)
            levels.append(vals)
            weights.append((np.abs(amplitude) ** 2).sum(axis=0))
    return np.concatenate(levels), np.concatenate(weights)


def main() -> None:
    values = [float(v) for v in sys.argv[1:]]
    points = list(zip(values[0::2], values[1::2]))
    cluster = geometry.build_cluster("cubic", (1, 1, 1))
    states = full_spin_basis(cluster.n_sites)
    state_index = {int(s): i for i, s in enumerate(states)}
    ice_indices = np.asarray(
        [state_index[int(s)] for s in cluster.ice_states], dtype=np.int64)
    labels = polarization_sector_labels(cluster)
    same = labels[:, None] == labels[None, :]
    complement = np.setdiff1d(np.arange(len(states)), ice_indices)
    sectors_cache = None

    for jpm, jpmpm in points:
        started = time.perf_counter()
        hamiltonian = microscopic_character_hamiltonian(
            cluster, states, jpm, np.zeros(3), jpmpm=jpmpm)
        if sectors_cache is None:
            sectors_cache = translation_sector_bases(
                states, cubic16_translation_permutations(cluster))

        raw_levels, raw_ice = raw_side(
            hamiltonian, states, sectors_cache, ice_indices)
        order = np.argsort(raw_levels)
        raw_levels, raw_ice = raw_levels[order], raw_ice[order]
        trace_p = float(raw_ice.sum())

        real = hamiltonian.real.tocsr()
        q_block = real[complement][:, complement].tocsr()
        coupling = np.asarray(real[complement][:, ice_indices].todense())
        ice_diag = real[ice_indices][:, ice_indices].toarray()
        edge = float(eigsh(q_block, k=1, which="SA", tol=1e-9,
                           return_eigenvectors=False)[0])
        floor = raw_levels[0] - 0.05 * abs(raw_levels[0]) - 0.02
        top = edge - 0.02 * (edge - floor)
        anchors = list(np.linspace(floor, top, N_ANCHOR))
        samples = {}

        def f_eigenvalues(z):
            solution = block_cg(
                lambda v, s=z: q_block @ v - s * v, coupling,
                np.zeros_like(coupling))
            f_matrix = ice_diag - coupling.T @ solution
            f_matrix = 0.5 * (f_matrix + f_matrix.T)
            samples[z] = np.linalg.eigvalsh(np.where(same, f_matrix, 0.0))

        def roots_from(anchor_list):
            zs = np.array(sorted(anchor_list))
            table = np.vstack([samples[z] for z in zs])
            gaps = table - zs[:, None]
            out = np.full(len(ice_indices), np.nan)
            for branch in range(len(ice_indices)):
                g = gaps[:, branch]
                if g[-1] > 0 or g[0] <= 0:
                    continue
                interp = PchipInterpolator(zs, g)
                low = zs[np.flatnonzero(g > 0)[-1]]
                high = zs[np.flatnonzero(g <= 0)[0]]
                out[branch] = brentq(interp, low, high, xtol=1e-13)
            return out

        for z in anchors:
            f_eigenvalues(z)
        roots = roots_from(anchors)
        finite = roots[np.isfinite(roots)]
        if len(finite):
            for extra in np.percentile(finite, [30.0, 70.0]):
                extra = float(extra)
                if all(abs(extra - z) > 1e-6 for z in anchors):
                    anchors.append(extra)
                    f_eigenvalues(extra)
            roots = roots_from(anchors)
        masked = np.sort(roots[np.isfinite(roots)])
        n_found = len(masked)

        # oracle at jpmpm = 0: exact masked band from the fixed-Sz cache
        oracle = None
        tag16 = f"{jpm:+.3f}".replace(".", "p")
        if abs(jpmpm) < 1e-12 and (BAND_DIR / f"band_{tag16}.json").exists():
            record = json.loads((BAND_DIR / f"band_{tag16}.json").read_text())
            exact = np.asarray(
                [x for x in record["levels"] if x is not None], float)
            exact = np.sort(exact[np.isfinite(exact)])
            n = min(len(exact), n_found)
            oracle = float(np.max(np.abs(masked[:n] - exact[:n])))

        # n_found = 0 means the whole winding-free band sits inside the
        # continuum (strong pair binding): no replacement is possible and
        # the fold degenerates to the raw spectrum -- recorded, not hidden.
        rank = np.argsort(raw_ice)[::-1]
        kept = rank[n_found:]
        fold_levels = (np.concatenate([raw_levels[kept], masked])
                       if n_found else raw_levels)

        fold_peaks = all_peaks(fold_levels)
        raw_peaks = all_peaks(raw_levels)

        record = {
            "jpm": jpm, "jpmpm": jpmpm,
            "trace_P": trace_p, "n_found": n_found,
            "n_escaped": int(len(ice_indices) - n_found),
            "z_last_removed": (float(raw_ice[rank[n_found - 1]])
                               if n_found else None),
            "z_first_kept": float(raw_ice[rank[n_found]]),
            "fold_peaks": fold_peaks.tolist(),
            "raw_peaks": raw_peaks.tolist(),
            "oracle_max_dev": oracle,
            "runtime": time.perf_counter() - started,
        }
        tag = f"jpm{jpm:+.4f}_pp{jpmpm:+.4f}".replace(".", "p")
        with open(OUTPUT / f"point_{tag}.json", "w") as handle:
            json.dump(record, handle, indent=1, default=float)
        np.savez_compressed(OUTPUT / f"point_{tag}.npz", jpm=jpm, jpmpm=jpmpm,
                            fold_levels=fold_levels, masked=masked,
                            raw_levels=raw_levels, raw_ice=raw_ice)
        print(f"jpm={jpm:+.3f} pp={jpmpm:+.3f} trP={trace_p:.6f} "
              f"band {n_found}/90 | fold peaks {np.round(fold_peaks, 5)} | "
              f"raw peaks {np.round(raw_peaks, 5)} | "
              + (f"oracle {oracle:.1e} " if oracle is not None else "")
              + f"({record['runtime']:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
