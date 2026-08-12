#!/usr/bin/env python3
"""Paired periodic and transport-masked FCC-32 ice-band thermodynamics.

Both spectra are roots of the same depth-truncated Feshbach map.  At every
anchor the expensive Q-space solve is shared; the periodic spectrum
diagonalizes F(z), while the corrected spectrum diagonalizes Delta[F(z)].
This makes the comparison host-, virtual-space-, anchor-, and state-count
matched.  No cubic-16 data, counterterm, or polar pullback enter.

Usage: run_fcc32_thermodynamics.py <jpm> [--depth=1]
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
from run_multi_anchor_band import block_cg  # noqa: E402
from run_truncated_feshbach import bfs_space, build_hamiltonian  # noqa: E402

OUTPUT = ROOT / "campaign" / "outputs" / "fcc32_thermodynamics"
N_ANCHOR = 10
CHUNK = 256
N_SITES = 32
TEMPERATURE = np.geomspace(5.0e-7, 0.20, 4000)


def thermal_curve(levels: np.ndarray) -> np.ndarray:
    """Heat capacity per spin of one matched 2970-level ice band."""
    shifted = np.sort(np.asarray(levels, dtype=float))
    shifted -= shifted[0]
    beta = 1.0 / TEMPERATURE[:, None]
    weight = np.exp(-beta * shifted[None, :])
    partition = weight.sum(axis=1)
    mean = (weight @ shifted) / partition
    second = (weight @ (shifted * shifted)) / partition
    return (second - mean * mean) / (N_SITES * TEMPERATURE**2)


def peak_positions(curve: np.ndarray) -> np.ndarray:
    """All reproducible band peaks, ordered from low to high temperature."""
    indices, _ = find_peaks(
        curve, prominence=max(1.0e-7, 0.002 * float(curve.max())))
    return TEMPERATURE[indices]


def main() -> int:
    coupling = None
    depth = 1
    for token in sys.argv[1:]:
        if token.startswith("--depth="):
            depth = int(token.split("=", 1)[1])
        else:
            coupling = float(token)
    if coupling is None:
        raise SystemExit("usage: run_fcc32_thermodynamics.py <jpm> [--depth=1]")

    started = time.perf_counter()
    cluster = geometry.build_cluster("fcc", (2, 2, 2))
    space = bfs_space(cluster, depth)
    ice_states = np.sort(np.asarray(cluster.ice_states, dtype=np.uint64))
    ice_rows = np.searchsorted(space, ice_states)
    complement = np.setdiff1d(np.arange(len(space)), ice_rows)
    order = np.argsort(np.asarray(cluster.ice_states, dtype=np.uint64))
    labels = polarization_sector_labels(cluster)[order]
    same_transport = labels[:, None] == labels[None, :]
    n_ice = len(ice_states)

    hamiltonian = build_hamiltonian(cluster, space, coupling).tocsr()
    microscopic_ground = float(eigsh(
        hamiltonian, k=1, which="SA", tol=1.0e-9,
        return_eigenvectors=False)[0])
    q_block = hamiltonian[complement][:, complement].tocsr()
    continuum_edge = float(eigsh(
        q_block, k=1, which="SA", tol=1.0e-9,
        return_eigenvectors=False)[0])
    coupling_qp = hamiltonian[complement][:, ice_rows].tocsc()
    coupling_pq = coupling_qp.T.tocsr()
    ice_block = hamiltonian[ice_rows][:, ice_rows].toarray()

    floor = microscopic_ground - 0.05 * abs(microscopic_ground) - 0.02
    top = continuum_edge - 0.02 * (continuum_edge - floor)
    anchors = list(np.linspace(floor, top, N_ANCHOR))
    samples: dict[str, dict[float, np.ndarray]] = {
        "periodic": {}, "masked": {}}

    def evaluate(new_anchors: list[float]) -> None:
        ordered_anchors = sorted(new_anchors)
        folded = {z: ice_block.copy() for z in ordered_anchors}
        for start in range(0, n_ice, CHUNK):
            columns = slice(start, min(start + CHUNK, n_ice))
            rhs = np.asarray(coupling_qp[:, columns].todense())
            solution = np.zeros_like(rhs)
            for z in ordered_anchors:
                solution = block_cg(
                    lambda vector, shift=z: q_block @ vector - shift * vector,
                    rhs, solution)
                folded[z][:, columns] -= coupling_pq @ solution
            print(f"  columns {columns.stop}/{n_ice}", flush=True)
        for z in ordered_anchors:
            matrix = 0.5 * (folded[z] + folded[z].T)
            samples["periodic"][z] = np.linalg.eigvalsh(matrix)
            samples["masked"][z] = np.linalg.eigvalsh(
                np.where(same_transport, matrix, 0.0))

    def roots_from(method: str, anchor_list: list[float]) -> np.ndarray:
        energies = np.array(sorted(anchor_list))
        table = np.vstack([samples[method][z] for z in energies])
        gap = table - energies[:, None]
        roots = np.full(n_ice, np.nan)
        for branch in range(n_ice):
            values = gap[:, branch]
            positive = np.flatnonzero(values > 0.0)
            nonpositive = np.flatnonzero(values <= 0.0)
            if not len(positive) or not len(nonpositive):
                continue
            low_index = positive[-1]
            high_candidates = nonpositive[nonpositive > low_index]
            if not len(high_candidates):
                continue
            high_index = high_candidates[0]
            interpolator = PchipInterpolator(energies, values)
            roots[branch] = brentq(
                interpolator, energies[low_index], energies[high_index],
                xtol=1.0e-13)
        return np.sort(roots[np.isfinite(roots)])

    print(
        f"FCC-32 d={depth} jpm={coupling:+.3f}: space={len(space):,}, "
        f"ice={n_ice}, Q={len(complement):,}", flush=True)
    evaluate(anchors)

    initial = {name: roots_from(name, anchors) for name in samples}
    refinements: list[float] = []
    for roots in initial.values():
        if len(roots):
            for value in np.percentile(roots, [20.0, 50.0, 80.0]):
                if all(abs(value - old) > 1.0e-6 for old in anchors + refinements):
                    refinements.append(float(value))
    if refinements:
        print(f"  adding {len(refinements)} shared refinement anchors", flush=True)
        evaluate(refinements)
        anchors.extend(refinements)

    spectra = {name: roots_from(name, anchors) for name in samples}
    curves = {
        name: thermal_curve(roots) if len(roots) == n_ice else None
        for name, roots in spectra.items()}
    peaks = {
        name: peak_positions(curve) if curve is not None else np.array([])
        for name, curve in curves.items()}

    record: dict[str, object] = {
        "cluster": "fcc32",
        "virtual_depth": depth,
        "jpm": coupling,
        "space": int(len(space)),
        "n_ice": int(n_ice),
        "microscopic_ground": microscopic_ground,
        "continuum_edge": continuum_edge,
        "n_anchors": int(len(anchors)),
        "runtime": time.perf_counter() - started,
    }
    arrays: dict[str, np.ndarray] = {
        "temperature": TEMPERATURE,
        "anchors": np.asarray(sorted(anchors)),
    }
    for name in ("periodic", "masked"):
        roots = spectra[name]
        record[f"{name}_n_found"] = int(len(roots))
        record[f"{name}_complete"] = bool(len(roots) == n_ice)
        record[f"{name}_band_bottom"] = float(roots[0]) if len(roots) else None
        record[f"{name}_band_top"] = float(roots[-1]) if len(roots) else None
        record[f"{name}_peaks"] = peaks[name].tolist()
        record[f"{name}_low_peak"] = (
            float(peaks[name][0]) if len(peaks[name]) else None)
        arrays[f"{name}_levels"] = roots
        if curves[name] is not None:
            arrays[f"{name}_heat_capacity_per_site"] = curves[name]

    record["comparison_complete"] = bool(
        record["periodic_complete"] and record["masked_complete"])
    tag = f"{coupling:+.3f}".replace(".", "p")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / f"thermo_{tag}.json").write_text(
        json.dumps(record, indent=1, default=float))
    np.savez_compressed(OUTPUT / f"thermo_{tag}.npz", **arrays)
    print(json.dumps(record, indent=1, default=float), flush=True)
    return 0 if record["comparison_complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
