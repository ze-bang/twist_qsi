#!/usr/bin/env python3
"""Brillouin-Wigner extension of the winding-free protocol past band isolation.

For each coupling this driver builds the exact Feshbach map on the zero-spinon
block (one dense QHQ diagonalization, no iterative solver), solves the
self-consistent roots with and without the polarization-block mask, and
records the dissolution diagnostics: converged-root count, minimum pole
distance, and ice weights.

Validation oracle: wherever the ice-descended band is isolated, the UNMASKED
self-consistent roots are exact eigenvalues of H and must reproduce the band
eigenvalues already cached by the scan (``periodic_band``) to solver
precision.  A grey-zone claim is only trustworthy if this oracle passes on
the isolated side with the same code path.

Usage:
    python3 campaign/run_bw_greyzone.py --jpm -0.15          # one point
    python3 campaign/run_bw_greyzone.py --ladder             # -0.19 .. -0.30
    python3 campaign/run_bw_greyzone.py --summary            # collect JSON
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "notes"))

import recompute_finite_size_artifact as geometry  # noqa: E402
from qsi_campaign.downfold import build_blocks, solve_roots  # noqa: E402
from qsi_campaign.exact_band import (  # noqa: E402
    fixed_magnetization_basis,
    microscopic_character_hamiltonian,
)
from qsi_campaign.material_scan import local_maxima  # noqa: E402
from qsi_campaign.thermodynamics import thermal_observables  # noqa: E402

CACHE = ROOT / "campaign" / "cache" / "bw_greyzone"
SCAN_CACHE = ROOT / "campaign" / "cache" / "ce2hf2o7_points"
OUTPUT = ROOT / "campaign" / "outputs"

LADDER = [round(-0.19 - 0.01 * k, 6) for k in range(12)]        # -0.19 .. -0.30
VALIDATION = [-0.05, -0.10, -0.15, -0.18]                        # isolated side


def band_only_ring_peak(energies: np.ndarray) -> float | None:
    """Low-temperature maximum of C(T) from the 90 root energies alone.

    The ring-exchange peak is band-dominated, so the band-only curve locates
    it; the microscopic complement only reshapes the ice crossover above.
    """
    finite = np.sort(energies[np.isfinite(energies)])
    if len(finite) < 2:
        return None
    temperatures = np.geomspace(1e-6, 1.0, 2600)
    heat = thermal_observables(finite, temperatures, n_sites=16)["heat_capacity_per_site"]
    peaks = [p for p in local_maxima(temperatures, heat) if p[0] < 0.1]
    return float(min(peaks)[0]) if peaks else None


def solve_coupling(cluster, jpm: float) -> dict:
    tag = f"jpm{jpm:+.6f}".replace(".", "p")
    cache = CACHE / f"bw_{tag}.npz"
    if cache.exists():
        with np.load(cache) as saved:
            return {key: saved[key] for key in saved.files} | {"cached": True}

    started = time.perf_counter()
    states = fixed_magnetization_basis(cluster.n_sites, cluster.n_sites // 2)
    index = {int(s): k for k, s in enumerate(states)}
    ice_indices = np.asarray([index[int(s)] for s in cluster.ice_states], dtype=np.int64)
    hamiltonian = microscopic_character_hamiltonian(cluster, states, jpm, np.zeros(3))
    blocks = build_blocks(cluster, states, ice_indices, hamiltonian)
    build_seconds = time.perf_counter() - started

    plain = solve_roots(blocks, masked=False)
    masked = solve_roots(blocks, masked=True)

    result = {
        "jpm": np.asarray(jpm),
        "build_seconds": np.asarray(build_seconds),
        "qhq_min": np.asarray(float(blocks.poles.min())),
        "plain_energies": plain.energies,
        "plain_converged": plain.converged,
        "plain_ice_weight": plain.ice_weight,
        "plain_pole_distance": plain.pole_distance,
        "masked_energies": masked.energies,
        "masked_converged": masked.converged,
        "masked_ice_weight": masked.ice_weight,
        "masked_pole_distance": masked.pole_distance,
    }
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, **result)
    return result | {"cached": False}


def oracle_check(jpm: float, plain_energies: np.ndarray) -> dict | None:
    """Compare unmasked roots against the cached exact band, if available."""
    tag = f"jpm{jpm:+.6f}_pp{0.0:+.6f}".replace(".", "p")
    path = SCAN_CACHE / f"cubic16_xyz_{tag}.npz"
    if not path.exists():
        return None
    with np.load(path) as saved:
        if not bool(saved.get("well_defined", True)):
            return {"band_isolated": False}
        band = np.sort(np.asarray(saved["periodic_band"], dtype=float))
    roots = np.sort(plain_energies[np.isfinite(plain_energies)])
    if len(roots) != len(band):
        return {"band_isolated": True, "n_roots": int(len(roots)),
                "max_abs_difference": None}
    return {"band_isolated": True, "n_roots": int(len(roots)),
            "max_abs_difference": float(np.max(np.abs(roots - band)))}


def report(jpm: float, point: dict) -> dict:
    masked_energies = np.asarray(point["masked_energies"], dtype=float)
    entry = {
        "jpm": jpm,
        "plain_converged": int(np.sum(point["plain_converged"])),
        "masked_converged": int(np.sum(point["masked_converged"])),
        "masked_min_pole_distance": float(np.nanmin(point["masked_pole_distance"]))
        if np.any(np.isfinite(point["masked_pole_distance"])) else None,
        "masked_mean_ice_weight": float(np.nanmean(point["masked_ice_weight"]))
        if np.any(np.isfinite(point["masked_ice_weight"])) else None,
        "masked_min_ice_weight": float(np.nanmin(point["masked_ice_weight"]))
        if np.any(np.isfinite(point["masked_ice_weight"])) else None,
        "masked_ring_peak_Jzz": band_only_ring_peak(masked_energies),
        "oracle": oracle_check(jpm, np.asarray(point["plain_energies"], dtype=float)),
    }
    return entry


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jpm", type=float, default=None)
    parser.add_argument("--ladder", action="store_true")
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args()

    cluster = geometry.build_cluster("cubic", (1, 1, 1))

    if args.summary:
        entries = []
        for path in sorted(CACHE.glob("bw_*.npz")):
            with np.load(path) as saved:
                point = {key: saved[key] for key in saved.files}
            entries.append(report(float(point["jpm"]), point))
        entries.sort(key=lambda e: e["jpm"])
        out = OUTPUT / "bw_greyzone_summary.json"
        out.write_text(json.dumps({"n_points": len(entries), "points": entries}, indent=2) + "\n")
        print(f"wrote {out} with {len(entries)} points")
        return

    couplings = ([args.jpm] if args.jpm is not None else []) \
        + (VALIDATION if args.validate else []) \
        + (LADDER if args.ladder else [])
    for jpm in couplings:
        started = time.perf_counter()
        point = solve_coupling(cluster, jpm)
        entry = report(jpm, point)
        entry["seconds"] = round(time.perf_counter() - started, 1)
        print(json.dumps(entry), flush=True)


if __name__ == "__main__":
    main()
