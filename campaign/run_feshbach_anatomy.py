#!/usr/bin/env python3
"""Anatomy of the exact Feshbach map on the ice manifold, at any coupling.

The Feshbach (Schur-complement) map

    F(z) = PHP + PHQ (z - QHQ)^{-1} QHP ,   P = the 90 classical ice states,

needs no isolated band, no reference frame, and no Gram matrix: it is defined
at every coupling, and its self-consistent roots are exact eigenvalues of H.
This script uses it as a *measurement device* for the downfolded gauge theory
rather than only as a root finder.  Per coupling it reports

  (i)   the amplitude anatomy of F(z) in the ice basis: every off-diagonal
        element is classified by the microscopic string that connects the two
        ice configurations (contractible hexagon, axial/diagonal winding
        four-cycle, wrapping hexagon, longer strings), giving the dressed
        ring amplitude g(lambda) and the dressed winding amplitude;

  (ii)  the diagonal ("electric") potential: diag F(z) regressed on the number
        of flippable hexagons, i.e. the coefficient V of the ring-exchange
        model H = V * N_flip - g * sum_hex (|o><o'| + h.c.).  The ratio V/g is
        the Rokhsar-Kivelson ratio of the emergent gauge theory;

  (iii) the winding-free (masked) map, its self-consistent band, the ring heat
        capacity peak, the ground-multiplet hexagon flux, and the residue Z;

  (iv)  a fidelity test of the two-parameter (g, V) ring model against the full
        masked band, so that "does a gauge theory survive?" is answered by a
        number rather than by a frame.

Usage: run_feshbach_anatomy.py <jpm> [<jpm> ...]
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

import recompute_finite_size_artifact as geometry  # noqa: E402
from qsi_campaign.exact_band import (  # noqa: E402
    fixed_magnetization_basis,
    microscopic_character_hamiltonian,
    translation_sector_bases,
    cubic16_translation_permutations,
)
from qsi_campaign.downfold import build_blocks, solve_roots  # noqa: E402
from qsi_campaign.protocol import (  # noqa: E402
    polarization_sector_labels,
    sector_project,
)

OUTPUT = ROOT / "campaign" / "outputs" / "feshbach_anatomy"
OUTPUT.mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------------- geometry


def path_mask(path) -> int:
    mask = 0
    for site in path:
        mask |= 1 << int(site)
    return mask


def string_classes(cluster) -> dict[int, str]:
    """Map the site mask of every elementary loop to its class name."""
    table: dict[int, str] = {}
    for sites, wind in cluster.loops4:
        axis = sorted(abs(int(w)) for w in wind)
        table[path_mask(sites)] = "wind4_axial" if axis == [0, 0, 1] \
            else "wind4_diagonal"
    for sites, wind in cluster.hexes:
        table[path_mask(sites)] = "hexagon" if tuple(wind) == (0, 0, 0) \
            else "hexagon_wrapping"
    return table


def flippable_counts(cluster, ice_states: np.ndarray) -> np.ndarray:
    """Number of contractible hexagons that are flippable in each ice state."""
    hexes = [sites for sites, wind in cluster.hexes if tuple(wind) == (0, 0, 0)]
    counts = np.zeros(len(ice_states), dtype=float)
    for sites in hexes:
        mask = path_mask(sites)
        pattern = 0
        for position, site in enumerate(sites):
            if position % 2 == 0:
                pattern |= 1 << int(site)
        other = mask ^ pattern
        occupied = np.asarray([int(s) & mask for s in ice_states])
        counts += ((occupied == pattern) | (occupied == other)).astype(float)
    return counts


def hexagon_flip_operator(cluster, ice_states: np.ndarray) -> np.ndarray:
    """sum_hex (|o><o'| + h.c.) over contractible hexagons, in the ice basis."""
    index = {int(s): i for i, s in enumerate(ice_states)}
    operator = np.zeros((len(ice_states),) * 2)
    for sites, wind in cluster.hexes:
        if tuple(wind) != (0, 0, 0):
            continue
        mask = path_mask(sites)
        pattern = 0
        for position, site in enumerate(sites):
            if position % 2 == 0:
                pattern |= 1 << int(site)
        other = mask ^ pattern
        for row, state in enumerate(ice_states):
            occupied = int(state) & mask
            if occupied == pattern or occupied == other:
                column = index[int(state) ^ mask]
                operator[row, column] += 1.0
    return operator


# ------------------------------------------------------------- thermodynamics


def band_peak(levels, lo=1e-8, hi=0.3, n=4000) -> float:
    shifted = np.asarray(levels, dtype=float) - float(np.min(levels))
    grid = np.geomspace(lo, hi, n)
    beta = 1.0 / grid[:, None]
    weight = np.exp(-beta * shifted[None, :])
    partition = weight.sum(axis=1)
    mean = (weight * shifted[None, :]).sum(axis=1) / partition
    second = (weight * shifted[None, :] ** 2).sum(axis=1) / partition
    heat = (second - mean * mean) / grid**2
    return float(grid[int(np.argmax(heat))])


# ------------------------------------------------------------------ analysis


def classify_operator(matrix, ice_states, table, labels):
    """Split an ice-basis operator into microscopic string classes."""
    states = np.asarray([int(s) for s in ice_states])
    difference = states[:, None] ^ states[None, :]
    length = np.zeros_like(difference)
    for shift in range(64):
        length += (difference >> shift) & 1
    names = np.empty(difference.shape, dtype=object)
    for row in range(len(states)):
        for column in range(len(states)):
            if row == column:
                names[row, column] = "diagonal"
                continue
            key = int(difference[row, column])
            names[row, column] = table.get(
                key, f"string{int(length[row, column])}")
    stats = {}
    same_sector = labels[:, None] == labels[None, :]
    for name in sorted({str(v) for v in names.ravel()}):
        selection = names == name
        values = np.asarray(matrix)[selection].real
        stats[name] = {
            "count": int(selection.sum()),
            "mean": float(values.mean()) if values.size else 0.0,
            "rms": float(np.sqrt((values**2).mean())) if values.size else 0.0,
            "max_abs": float(np.abs(values).max()) if values.size else 0.0,
            "kept_by_mask": int((selection & same_sector).sum()),
        }
    return names, stats


def real_sector_block(basis, hamiltonian) -> np.ndarray:
    """Dense sector block; the theta = 0 Hamiltonian is real in this basis."""
    block = (basis.conj().T @ (hamiltonian @ basis)).toarray()
    block = 0.5 * (block + block.conj().T)
    if np.max(np.abs(block.imag), initial=0.0) > 1e-12:
        raise ValueError("sector block is not real")
    return np.ascontiguousarray(block.real)


def main() -> None:
    arguments = [v for v in sys.argv[1:] if not v.startswith("--")]
    oracle_wanted = "--no-oracle" not in sys.argv
    couplings = [float(v) for v in arguments]
    cluster = geometry.build_cluster("cubic", (1, 1, 1))
    states = fixed_magnetization_basis(cluster.n_sites, cluster.n_sites // 2)
    index = {int(s): i for i, s in enumerate(states)}
    ice_states = np.asarray(cluster.ice_states, dtype=np.uint64)
    ice_indices = np.asarray([index[int(s)] for s in ice_states])
    sectors = translation_sector_bases(
        states, cubic16_translation_permutations(cluster))
    labels = polarization_sector_labels(cluster)
    table = string_classes(cluster)
    n_flip = flippable_counts(cluster, ice_states)
    hex_operator = hexagon_flip_operator(cluster, ice_states)

    for jpm in couplings:
        started = time.perf_counter()
        hamiltonian = microscopic_character_hamiltonian(
            cluster, states, jpm, np.zeros(3))

        # exact low spectrum (oracle for the unmasked roots)
        blocks = build_blocks(cluster, states, ice_indices, hamiltonian)
        masked = solve_roots(blocks, masked=True)
        n_masked = int(masked.converged.sum())

        if oracle_wanted:
            exact = []
            for basis in sectors.values():
                exact.append(eigh(real_sector_block(basis, hamiltonian),
                                  subset_by_index=(0, 119), eigvals_only=True))
            exact = np.sort(np.concatenate(exact))[:120]
            raw = solve_roots(blocks, masked=False)
            n_raw = int(raw.converged.sum())
            raw_energies = np.sort(raw.energies[raw.converged])
            oracle = float(np.max(
                np.abs(raw_energies - exact[:len(raw_energies)]))) \
                if n_raw else np.nan
            z_star = float(exact[0])
        else:
            # only the ground-state energy is needed to place the map
            exact = np.asarray(eigsh(hamiltonian.real.tocsc(), k=1,
                                     which="SA", tol=1e-11,
                                     return_eigenvectors=False))
            n_raw = -1
            oracle = np.nan
            z_star = float(exact[0])

        # ---- anatomy of the map at the exact ground-state energy
        f_raw = blocks.f_matrix(z_star, masked=False).real
        f_masked = sector_project(f_raw, labels).real
        names, stats = classify_operator(f_raw, ice_states, table, labels)

        # dressed ring amplitude and RK potential from the *masked* map
        hexagon = names == "hexagon"
        g_eff = -float(np.asarray(f_masked)[hexagon].mean())
        g_spread = float(np.asarray(f_masked)[hexagon].std())
        design = np.column_stack([np.ones_like(n_flip), n_flip])
        diagonal = np.diag(np.asarray(f_masked)).real
        solution, *_ = np.linalg.lstsq(design, diagonal, rcond=None)
        residual = diagonal - design @ solution
        v_eff = float(solution[1])
        v_r2 = float(1.0 - residual.var() / max(diagonal.var(), 1e-300))
        diagonal_width = float(diagonal.std())

        # longer-string content of the winding-free map
        off = np.asarray(f_masked).copy()
        np.fill_diagonal(off, 0.0)
        hex_norm = float(np.linalg.norm(np.where(hexagon, off, 0.0)))
        long_mask = np.zeros_like(hexagon)
        for name in stats:
            if name.startswith("string") and int(name[6:]) >= 8:
                long_mask |= names == name
        long_norm = float(np.linalg.norm(np.where(long_mask, off, 0.0)))
        total_norm = float(np.linalg.norm(off))

        # ---- winding-free band, its peak, flux, residue
        if n_masked:
            band = np.sort(masked.energies[masked.converged])
            peak = band_peak(band)
            root = float(band[0])
            values, vectors = np.linalg.eigh(
                blocks.f_matrix(root, masked=True))
            multiplet = vectors[:, values - values[0] < 1e-6]
            if multiplet.shape[1] < 1:
                multiplet = vectors[:, :1]
            flux = float(np.real(np.einsum(
                "in,ij,jn->", multiplet.conj(), hex_operator, multiplet))
                / (multiplet.shape[1] * 16.0))
            z_mean = float(np.nanmean(masked.ice_weight))
            z_ground = float(masked.ice_weight[
                int(np.nanargmin(np.where(masked.converged,
                                          masked.energies, np.inf)))])
        else:
            band = np.zeros(0)
            peak = np.nan
            flux = np.nan
            z_mean = np.nan
            z_ground = np.nan

        # ---- two-parameter ring model against the full masked map
        model = v_eff * np.diag(n_flip) - g_eff * hex_operator
        model_levels = np.linalg.eigvalsh(model)
        if n_masked == 90:
            reference = band - band.mean()
            trial = model_levels - model_levels.mean()
            model_error = float(np.linalg.norm(trial - reference)
                                / max(np.linalg.norm(reference), 1e-300))
            model_peak = band_peak(model_levels)
        else:
            model_error = np.nan
            model_peak = band_peak(model_levels)

        record = {
            "jpm": jpm,
            "n_raw_roots": n_raw,
            "n_masked_roots": n_masked,
            "oracle_error": oracle,
            "continuum_edge": float(blocks.poles.min()),
            "gs_energy": z_star,
            "g_eff": g_eff,
            "g_spread": g_spread,
            "g_bare": 12.0 * abs(jpm) ** 3,
            "v_eff": v_eff,
            "v_r2": v_r2,
            "diagonal_width": diagonal_width,
            "width_over_g": diagonal_width / abs(g_eff) if g_eff else np.nan,
            "rk_ratio": v_eff / g_eff if g_eff else np.nan,
            "long_over_hex": long_norm / hex_norm if hex_norm else np.nan,
            "hex_over_total": hex_norm / total_norm if total_norm else np.nan,
            "ring_peak": peak,
            "ring_peak_over_g6": peak / (12.0 * abs(jpm) ** 3),
            "gs_flux": flux,
            "z_mean": z_mean,
            "z_ground": z_ground,
            "model_error": model_error,
            "model_peak": model_peak,
            "runtime": time.perf_counter() - started,
        }
        for name, entry in stats.items():
            record[f"class::{name}"] = entry

        tag = f"{jpm:+.3f}".replace(".", "p")
        with open(OUTPUT / f"anatomy_{tag}.json", "w") as handle:
            json.dump(record, handle, indent=1, default=float)
        np.savez_compressed(
            OUTPUT / f"anatomy_{tag}.npz",
            jpm=jpm, exact=exact, band=band,
            raw_energies=raw.energies if oracle_wanted
            else np.zeros(0), raw_converged=raw.converged if oracle_wanted
            else np.zeros(0, dtype=bool),
            masked_energies=masked.energies,
            masked_converged=masked.converged,
            ice_weight=masked.ice_weight,
            f_masked=f_masked, f_raw=f_raw, poles_min=blocks.poles.min(),
            n_flip=n_flip, model_levels=model_levels,
            diagonal_residual=residual)

        print(f"jpm={jpm:+.3f} roots={n_masked}/90 (raw {n_raw}, oracle "
              f"{oracle:.1e}) W/g={record['width_over_g']:.2f} g={g_eff:+.3e} [{g_eff/(12*abs(jpm)**3):.3f} g6] "
              f"V={v_eff:+.3e} V/g={record['rk_ratio']:+.3f} "
              f"long/hex={record['long_over_hex']:.3f} "
              f"peak/g6={record['ring_peak_over_g6']:.4f} flux={flux:+.4f} "
              f"Z={z_ground:.3f} modelerr={model_error:.3f} "
              f"({record['runtime']:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
