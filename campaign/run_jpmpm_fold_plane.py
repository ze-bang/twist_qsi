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

``--bordered`` replaces the multi-anchor solve with the exact bordered
per-sector band from ``masked_band_complete`` at the same (Jpm, Jpmpm).
This matters for more than speed: the anchor solve brackets roots below the
*global* continuum edge, so wherever a branch rises past it the fold removes
only ``n_found`` raw levels and leaves ``90 - n_found`` ice-descended raw
levels in the ensemble alongside their replacements.  At (-0.125, 0.30) that
was 63 contaminating levels against 27 replacements, and the resulting peaks
are not winding-free statements.  The bordered solver uses each transport
sector's own live-pole edge and returns the complete band, so the fold
removes and replaces all 90.

Usage: run_jpmpm_fold_plane.py [--bordered] [--gpu] [--cross-check]
                               <jpm> <jpmpm> [<jpm> <jpmpm> ...]
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
from qsi_campaign.downfold import _dense_eigh  # noqa: E402
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


def raw_side(hamiltonian, states, sectors, ice_indices, gpu=False):
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
            block = np.ascontiguousarray(projected.real[np.ix_(mask, mask)])
            vals, vecs = _dense_eigh(block, gpu)
            amplitude = np.asarray(ice_basis[:, mask] @ vecs)
            levels.append(vals)
            weights.append((np.abs(amplitude) ** 2).sum(axis=0))
    return np.concatenate(levels), np.concatenate(weights)


def bordered_band(jpm: float, jpmpm: float):
    """The exact bordered per-sector band for this cell, if it is cached.

    Returns ``(levels, path)`` with levels sorted ascending and the lost
    entries (recorded as null) dropped, or ``(None, None)``.
    """
    stem = f"{jpm:+.3f}".replace(".", "p")
    candidates = [f"band_{stem}_pp{jpmpm:+.3f}".replace(".", "p") + ".json"] \
        if abs(jpmpm) > 1e-12 else \
        [f"band_{stem}.json", f"band_{stem}_fullbasis_klein.json"]
    for name in candidates:
        path = BAND_DIR / name
        if not path.exists():
            continue
        record = json.loads(path.read_text())
        levels = np.asarray([x for x in record["levels"] if x is not None],
                            float)
        levels = np.sort(levels[np.isfinite(levels)])
        if len(levels):
            return levels, path.name
    return None, None


def main() -> None:
    use_bordered, gpu, cross_check, raw_values = False, False, False, []
    for token in sys.argv[1:]:
        if token == "--bordered":
            use_bordered = True
        elif token == "--gpu":
            gpu = True
        elif token == "--cross-check":
            cross_check = True
        else:
            raw_values.append(float(token))
    points = list(zip(raw_values[0::2], raw_values[1::2]))
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
            hamiltonian, states, sectors_cache, ice_indices, gpu=gpu)
        order = np.argsort(raw_levels)
        raw_levels, raw_ice = raw_levels[order], raw_ice[order]
        trace_p = float(raw_ice.sum())

        exact_band, band_file = (bordered_band(jpm, jpmpm)
                                 if use_bordered else (None, None))
        skip_anchors = exact_band is not None and not cross_check

        # the resolvent machinery exists only to locate the masked roots;
        # with the exact bordered band in hand none of it is assembled
        anchors, samples = [], {}
        if not skip_anchors:
            real = hamiltonian.real.tocsr()
            q_block = real[complement][:, complement].tocsr()
            coupling = np.asarray(real[complement][:, ice_indices].todense())
            ice_diag = real[ice_indices][:, ice_indices].toarray()
            edge = float(eigsh(q_block, k=1, which="SA", tol=1e-9,
                               return_eigenvectors=False)[0])
            floor = raw_levels[0] - 0.05 * abs(raw_levels[0]) - 0.02
            top = edge - 0.02 * (edge - floor)
            anchors = list(np.linspace(floor, top, N_ANCHOR))

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

        anchor_masked = None
        if not skip_anchors:
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
            anchor_masked = np.sort(roots[np.isfinite(roots)])

        # The bordered band is the complete one; the anchor solve brackets
        # only below the global edge and silently truncates above it.
        masked = exact_band if exact_band is not None else anchor_masked
        n_found = len(masked)

        oracle = None
        if anchor_masked is not None and exact_band is not None:
            n = min(len(anchor_masked), len(exact_band))
            oracle = float(np.max(np.abs(anchor_masked[:n] - exact_band[:n])))
        elif anchor_masked is not None:
            fallback, _ = bordered_band(jpm, jpmpm)
            if fallback is not None:
                n = min(len(fallback), len(anchor_masked))
                oracle = float(
                    np.max(np.abs(anchor_masked[:n] - fallback[:n])))

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
            "band_source": band_file or "multi_anchor",
            "n_anchor_found": (None if anchor_masked is None
                               else int(len(anchor_masked))),
            "runtime": time.perf_counter() - started,
        }
        tag = f"jpm{jpm:+.4f}_pp{jpmpm:+.4f}".replace(".", "p")
        if use_bordered:
            tag += "_bordered"
        with open(OUTPUT / f"point_{tag}.json", "w") as handle:
            json.dump(record, handle, indent=1, default=float)
        np.savez_compressed(OUTPUT / f"point_{tag}.npz", jpm=jpm, jpmpm=jpmpm,
                            fold_levels=fold_levels, masked=masked,
                            raw_levels=raw_levels, raw_ice=raw_ice)
        print(f"jpm={jpm:+.3f} pp={jpmpm:+.3f} trP={trace_p:.6f} "
              f"band {n_found}/90 [{record['band_source']}] | "
              f"fold peaks {np.round(fold_peaks, 5)} | "
              f"raw peaks {np.round(raw_peaks, 5)} | "
              + (f"anchor {record['n_anchor_found']}/90 dev {oracle:.1e} "
                 if oracle is not None else "")
              + f"({record['runtime']:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
