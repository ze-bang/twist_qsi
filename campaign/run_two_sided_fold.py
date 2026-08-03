#!/usr/bin/env python3
"""Two-sided fold: counterterm-free winding-free thermodynamics.

Every eigenstate of H is seen by both Feshbach folds -- the ice-side map
F(z) in proportion to its ice weight Z_n, and the spinon-side mirror map in
proportion to 1 - Z_n -- and the weights are exactly complementary:

    Tr e^{-beta H} = sum_n Z_n e^{-beta E_n} + sum_n (1 - Z_n) e^{-beta E_n}.

The naive winding-free convention -- keep the spinon-side sum raw and
replace the ice-side sum by the masked roots with residues Ztilde_m -- is
computed here as ``fold_naive`` and is WRONG, instructively: the raw band
remnants (weight 1 - Z_n each) sit at raw absolute energies, an artifact
binding of order g4 BELOW the masked band.  At T ~ g6 << g4 the Boltzmann
factor overwhelms any weight, the remnant hijacks the ensemble, the
winding-free bump disappears, and a spurious offset-Schottky appears at the
artifact scale, taller than raw ED's own bump.  No weight convention
repairs a lower-lying level; the naive weighted union has a corrupted
T -> 0 limit by construction.  (Measured at Jpm = -0.05: C(0.84 g6) = 0.000
where splice and H+C put 0.157, and a 0.65-high spur at 7.6 g6.)

The correct two-sided construction uses the mirror side as an IDENTIFIER,
not a weight.  Each raw eigenstate carries its ice weight Z_n = |P psi_n|^2
-- the mirror map's residue, computable with no assignment.  Rank the raw
levels by Z_n, remove the n_band highest (wherever they sit: interleaved
with the continuum is fine, which is exactly the case the positional splice
cannot handle), and insert the masked roots at weight one:

    spectrum_wf = { raw levels, minus the n_band highest-Z_n }
                  union { masked roots }.

Counts are exact by construction, the masked roots are absolute (they share
the QHQ energy axis, so no trace matching), BICs slot in wherever they lie,
and the scheme reduces to the splice at weak coupling.  The classification
sharpness is measured, not assumed: the Z-rank gap between the last removed
and first kept level is bimodal-clean at weak coupling and its closure is
the honest, continuous signature of dissolution -- the obstruction theorem
seen in the data.  The three-way spread (counterterm / splice / fold) is
the measured definitional ambiguity of full-range winding-free
thermodynamics.

Inputs reused from the campaign caches when present:
  outputs/masked_band_complete/band_<tag>.json   masked levels + residues
  outputs/wf_thermodynamics/thermo_<tag>.npz     raw and H+C spectra
Missing masked bands or H+C spectra must be produced first with
run_masked_band_complete.py / run_wf_thermodynamics.py.

The raw-side ice weights Z_n need eigenvectors, which no cache retains, so
this script performs one dense eigh of the full S^z_tot = 0 block per
coupling and validates the levels against the thermo cache.

Usage: run_two_sided_fold.py <jpm> [<jpm> ...]
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

import recompute_finite_size_artifact as geometry  # noqa: E402
from qsi_campaign.exact_band import (  # noqa: E402
    fixed_magnetization_basis,
    microscopic_character_hamiltonian,
)
from qsi_campaign.protocol import full_hilbert_counterterm_spectrum  # noqa: E402

BAND_DIR = ROOT / "campaign" / "outputs" / "masked_band_complete"
THERMO_DIR = ROOT / "campaign" / "outputs" / "wf_thermodynamics"
OUTPUT = ROOT / "campaign" / "outputs" / "two_sided_fold"
OUTPUT.mkdir(parents=True, exist_ok=True)

N_ICE = 90


def tag_of(jpm: float) -> str:
    return f"{jpm:+.3f}".replace(".", "p")


def weighted_observables(levels: np.ndarray, weights: np.ndarray,
                         grid: np.ndarray) -> dict[str, np.ndarray]:
    """Canonical E(T), C(T), S(T) per site from a weighted spectrum."""
    keep = weights > 0.0
    levels = np.asarray(levels, float)[keep]
    weights = np.asarray(weights, float)[keep]
    shifted = levels - levels.min()
    beta = 1.0 / grid[:, None]
    boltzmann = weights[None, :] * np.exp(-beta * shifted[None, :])
    partition = boltzmann.sum(axis=1)
    first = (boltzmann @ shifted) / partition
    second = (boltzmann @ shifted**2) / partition
    return {
        "heat": (second - first**2) / grid**2 / 16.0,
        "entropy": (np.log(partition) + first / grid) / 16.0,
        "log_total_weight": float(np.log(weights.sum())),
    }


def peaks(grid: np.ndarray, curve: np.ndarray, split: float = 0.05):
    low = grid < split
    high = ~low
    return (float(grid[low][np.argmax(curve[low])]) if low.any() else np.nan,
            float(grid[high][np.argmax(curve[high])]) if high.any() else np.nan)


def raw_side(cluster, states, jpm):
    """Dense eigh of the S^z = 0 block: all raw levels with ice weights."""
    hamiltonian = microscopic_character_hamiltonian(
        cluster, states, jpm, np.zeros(3))
    dense = hamiltonian.toarray()
    if np.abs(dense.imag).max(initial=0.0) > 1e-12:
        raise ValueError("expected a real Hamiltonian at theta = 0")
    index = {int(s): i for i, s in enumerate(states)}
    ice_rows = np.asarray([index[int(s)] for s in cluster.ice_states])
    levels, vectors = eigh(np.ascontiguousarray(dense.real))
    ice_weight = (vectors[ice_rows, :] ** 2).sum(axis=0)
    return levels, ice_weight


def main() -> None:
    couplings = [float(v) for v in sys.argv[1:]]
    cluster = geometry.build_cluster("cubic", (1, 1, 1))
    states = fixed_magnetization_basis(cluster.n_sites, cluster.n_sites // 2)

    for jpm in couplings:
        started = time.perf_counter()
        tag = tag_of(jpm)

        band_file = BAND_DIR / f"band_{tag}.json"
        thermo_file = THERMO_DIR / f"thermo_{tag}.npz"
        missing = [str(p) for p in (band_file, thermo_file) if not p.exists()]
        if missing:
            print(f"jpm={jpm:+.3f} SKIPPED, missing cache: {missing}",
                  flush=True)
            continue

        band = json.loads(band_file.read_text())
        band_levels = np.asarray(band["levels"], float)
        band_weights = np.asarray(band["ice_weight"], float)
        found = np.isfinite(band_levels) & np.isfinite(band_weights)
        n_lost = int((~found).sum())

        thermo = np.load(thermo_file)
        grid = thermo["grid"]
        hc_levels = thermo["wf_levels"]

        # the dense eigh is the expensive step; reuse a previous run's raw side
        fold_npz = OUTPUT / f"fold_{tag}.npz"
        raw_levels = raw_ice_weight = None
        if fold_npz.exists():
            previous = np.load(fold_npz)
            if "raw_levels" in previous and "raw_ice_weight" in previous:
                raw_levels = previous["raw_levels"]
                raw_ice_weight = previous["raw_ice_weight"]
        if raw_levels is None:
            raw_levels, raw_ice_weight = raw_side(cluster, states, jpm)
        cache_deviation = float(
            np.max(np.abs(raw_levels - np.sort(thermo["raw_levels"]))))
        trace_p = float(raw_ice_weight.sum())      # must equal N_ICE exactly

        # the fold: remove the n_band highest-Z raw levels (wherever they
        # sit), insert the masked roots at weight one
        n_replace = int(found.sum())
        rank = np.argsort(raw_ice_weight)[::-1]
        removed, kept = rank[:n_replace], rank[n_replace:]
        fold_levels = np.concatenate(
            [raw_levels[kept], band_levels[found]])
        fold_weights = np.ones_like(fold_levels)
        fold = weighted_observables(fold_levels, fold_weights, grid)

        z_last_removed = float(raw_ice_weight[removed[-1]])
        z_first_kept = float(raw_ice_weight[kept[0]])
        # a zero Z-rank gap is harmless iff the cut slices an
        # energy-degenerate multiplet: removal choice is then invisible
        cut_energy_split = float(
            abs(raw_levels[removed[-1]] - raw_levels[kept[0]]))
        removed_vs_masked = float(np.max(np.abs(
            np.sort(raw_levels[removed]) - np.sort(band_levels[found]))))

        # the naive weighted union, kept as the documented failure mode
        naive = weighted_observables(
            np.concatenate([band_levels[found], raw_levels]),
            np.concatenate([band_weights[found], 1.0 - raw_ice_weight]),
            grid)

        # the three references
        raw = weighted_observables(raw_levels, np.ones_like(raw_levels), grid)
        hc = weighted_observables(hc_levels, np.ones_like(hc_levels), grid)
        splice_kwargs = dict(full_spectrum=hc_levels,
                             winding_free_band=np.sort(band_levels[found]))
        curves = {"raw": raw["heat"], "hc": hc["heat"], "fold": fold["heat"],
                  "fold_naive": naive["heat"]}
        splice_peaks = {}
        if n_lost == 0:
            for name, match in (("splice", True), ("splice_abs", False)):
                spec = full_hilbert_counterterm_spectrum(
                    **splice_kwargs, match_trace=match)
                curve = weighted_observables(spec, np.ones_like(spec), grid)
                curves[name] = curve["heat"]
                splice_peaks[name] = peaks(grid, curve["heat"])

        # the weight ledger and the classification-sharpness diagnostics
        sum_z_raw = trace_p
        sum_z_band = float(band_weights[found].sum())
        total_weight = float(fold_weights.sum())
        dimension = len(states)
        z_rank_gap = z_last_removed - z_first_kept

        g6 = 12.0 * abs(jpm) ** 3
        record = {
            "jpm": jpm,
            "g6": g6,
            "n_band_found": int(found.sum()),
            "n_band_lost": n_lost,
            "trace_P_check": sum_z_raw,           # exact value: 90
            "raw_levels_vs_cache": cache_deviation,
            "sum_Z_raw_side": sum_z_raw,
            "sum_Z_band_side": sum_z_band,
            "artifact_weight_removed": sum_z_raw - sum_z_band,
            "total_weight": total_weight,
            "dimension": dimension,
            "z_last_removed": z_last_removed,
            "z_first_kept": z_first_kept,
            "z_rank_gap": z_rank_gap,
            "cut_energy_split": cut_energy_split,
            "removed_vs_masked_levels": removed_vs_masked,
            "entropy_top_fold": float(fold["entropy"][-1]),
            "log_total_weight_over_16": fold["log_total_weight"] / 16.0,
        }
        for name, heats in curves.items():
            low, high = peaks(grid, heats)
            record[f"{name}_low_peak"] = low
            record[f"{name}_high_peak"] = high
            record[f"{name}_low_over_g6"] = low / g6

        definitions = [record["fold_low_peak"], record["hc_low_peak"]]
        if "splice" in curves:
            definitions.append(record["splice_low_peak"])
        record["definition_spread_low_peak"] = (
            float(np.max(definitions) - np.min(definitions)))
        record["definition_spread_over_g6"] = (
            record["definition_spread_low_peak"] / g6)
        record["runtime"] = time.perf_counter() - started

        with open(OUTPUT / f"fold_{tag}.json", "w") as handle:
            json.dump(record, handle, indent=1, default=float)
        np.savez_compressed(
            OUTPUT / f"fold_{tag}.npz", jpm=jpm, grid=grid,
            fold_levels=fold_levels, fold_weights=fold_weights,
            raw_levels=raw_levels, raw_ice_weight=raw_ice_weight,
            band_levels=band_levels[found], band_weights=band_weights[found],
            **{f"curve_{k}": v for k, v in curves.items()})

        spread = record["definition_spread_over_g6"]
        print(f"jpm={jpm:+.3f} trP={sum_z_raw:.10f} "
              f"band {record['n_band_found']}/90 (+{n_lost} lost) | "
              f"Z-rank gap {z_rank_gap:+.3f} "
              f"({z_last_removed:.3f}->{z_first_kept:.3f}) | "
              f"low peaks/g6: raw {record['raw_low_over_g6']:.2f} "
              f"hc {record['hc_low_over_g6']:.2f} "
              + (f"splice {record['splice_low_over_g6']:.2f} "
                 if "splice" in curves else "splice - ")
              + f"fold {record['fold_low_over_g6']:.2f} "
              f"naive {record['fold_naive_low_over_g6']:.2f} | "
              f"spread {spread:.3f} g6 | "
              f"cache-dev {cache_deviation:.1e} "
              f"({record['runtime']:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
