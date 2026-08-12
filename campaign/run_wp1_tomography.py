#!/usr/bin/env python3
"""WP1: loop-amplitude tomography of the winding-free map, and xi_gauge.

The headline this instrument serves: *virtual spinons do not kill the emergent
photon, they delocalize its Hamiltonian.*  The bare theory has exactly one ring
exchange, the contractible hexagon at perimeter 6.  Dressing by virtual spinon
excursions renormalizes it (measured on cubic-16: `|g_eff|/g6` falls from 0.81
to 0.07) and simultaneously *generates* longer processes that the bare theory
does not contain at all (`long/hex` rises from 0.13 to 1.17 over the same
range, while the Rokhsar-Kivelson potential stays below `0.022 |g|`).  The map
is not developing a potential term and is not going unstable: it is spreading.

This script measures the spread as a length.  Every entry of the masked ice
block is classified by the geometry of the string it flips, the connected
single-string classes are pooled into amplitudes `K_L`, and the decay is fit as

```text
K_L = A exp(-(L - 6) / xi_gauge).
```

`xi_gauge` is then plotted against coupling, against `1 - Z`, and against the
inverse continuum-edge gap.

**The truncation gate, which is the critical risk.**  On FCC-32 the complement
is far too large to keep, so `Q` is truncated to states within `levels`
exchange hops of the ice manifold.  The bias this introduces is worst exactly
where the physics is: a long string needs many virtual hops, so the longest
loops converge SLOWEST in `levels` and are biased LOW -- which would make
`xi_gauge` look shorter than it is and could manufacture a spurious trend.  The
gate is therefore run on cubic-16, where `levels` can be pushed until the
answer stops moving and the untruncated map is available from
`run_feshbach_anatomy.py` for comparison; the measured drift in `K_8/K_6` and
`K_10/K_6` transfers as the error bar on FCC-32.

Pre-registered kill criterion (from the plan): if the `levels = 1 -> 2` drift
of `K_10/K_6` exceeds its variation across couplings, no `xi_gauge` trend may
be claimed until `levels = 3` spot checks are in.  `--gate` runs the ladder and
evaluates that criterion directly.

Usage:
  run_wp1_tomography.py <cubic16|fcc32> <levels> <jpm> [<jpm> ...]
  run_wp1_tomography.py --gate <cubic16|fcc32> <jpm> [<jpm> ...]
"""

from __future__ import annotations

import json
import sys
import time
from math import comb
from pathlib import Path

import numpy as np
from scipy.sparse.linalg import eigsh

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "notes"))
sys.path.insert(0, str(ROOT / "campaign"))

import recompute_finite_size_artifact as geometry  # noqa: E402
from qsi_campaign.protocol import polarization_sector_labels  # noqa: E402
from qsi_campaign.tomography import (  # noqa: E402
    connected_class_amplitudes,
    fit_gauge_range,
    graph_distance_matrix,
    loop_tomography,
    minimum_image_distance_matrix,
)
from run_multi_anchor_band import block_cg  # noqa: E402
from run_truncated_feshbach import bfs_space, build_hamiltonian  # noqa: E402

OUTPUT = ROOT / "campaign" / "outputs" / "wp1_tomography"
CLUSTERS = {"cubic16": ("cubic", (1, 1, 1)), "fcc32": ("fcc", (2, 2, 2))}
CHUNK = 256


def masked_map(cluster, space, ice_rows, complement, hamiltonian, labels,
               z: float, chunk: int = CHUNK, checkpoint: Path | None = None,
               limit: int | None = None):
    """`F(z)` on the ice block, one block-CG solve per column chunk.

    Chunked so the dense right-hand side never exceeds `chunk` columns.  The
    sizing is not a detail: on FCC-32 the complement has 140k states at
    levels=1, 2.5M at levels=2 and 21.7M at levels=3, so a 256-column block
    costs 0.27, 4.8 and 41.5 GiB respectively -- and the CG carries about five
    such blocks at once.  Deep runs must drop `chunk`.

    ``checkpoint`` writes the accumulated columns after every chunk and resumes
    from them.  A levels=3 point runs for the better part of a day, the cluster
    does not grant wall-time extensions, and without this a timeout at 90%
    completion would return nothing at all.
    """
    q_block = hamiltonian[complement][:, complement].tocsr()
    coupling = hamiltonian[complement][:, ice_rows].tocsc()
    coupling_t = coupling.T.tocsr()
    f_matrix = hamiltonian[ice_rows][:, ice_rows].toarray()

    d = len(ice_rows) if limit is None else min(limit, len(ice_rows))
    done = 0
    if checkpoint is not None and checkpoint.exists():
        with np.load(checkpoint) as saved:
            # ``d`` distinguishes a sampled run from a full one, so a
            # checkpoint written by one can never be resumed by the other
            if float(saved["z"]) == z and int(saved["d"]) == d:
                f_matrix = saved["f_matrix"]
                done = int(saved["done"])
                print(f"  resumed from {checkpoint.name}: {done}/{d} columns",
                      flush=True)

    for start in range(done, d, chunk):
        stop = min(start + chunk, d)
        columns = slice(start, stop)
        rhs = np.asarray(coupling[:, columns].todense())
        solution = block_cg(lambda v: q_block @ v - z * v, rhs,
                            np.zeros_like(rhs))
        f_matrix[:, columns] -= coupling_t @ solution
        if checkpoint is not None:
            np.savez(checkpoint, f_matrix=f_matrix, done=stop, z=z, d=d)
        print(f"  columns {stop}/{d}", flush=True)

    if limit is None:
        f_matrix = 0.5 * (f_matrix + f_matrix.T)
    else:
        # only the first ``d`` columns were solved; symmetrise just the
        # principal block, and leave the unsolved remainder untouched
        block = f_matrix[:d, :d]
        f_matrix[:d, :d] = 0.5 * (block + block.T)
    same = labels[:, None] == labels[None, :]
    return np.where(same, f_matrix, 0.0), q_block


def run_point(name: str, levels: int, jpm: float, context,
              chunk: int = CHUNK, resumable: bool = False,
              columns: int | None = None) -> dict:
    """One tomography point.

    ``columns`` restricts the fold to the first N ice columns and takes the
    N x N principal block.  `K_L` is an average over thousands of entries per
    class, so a sample is enough to compare truncation depths: at N = 512 the
    L=6 class still holds ~460 surviving entries.  This exists because a full
    levels=3 point on FCC-32 runs for the better part of a day and the cluster
    grants no wall-time extensions -- a sampled answer that lands beats an
    exact one that gets killed at 90%.
    """
    cluster, labels, graph_distance, real_distance = context
    started = time.perf_counter()

    space = bfs_space(cluster, levels)
    ice_states = np.asarray(cluster.ice_states, dtype=np.uint64)
    ice_rows = np.searchsorted(space, ice_states)
    if not np.array_equal(space[ice_rows], ice_states):
        raise RuntimeError("ice manifold is not contained in the BFS space")
    complement = np.setdiff1d(np.arange(len(space)), ice_rows)

    hamiltonian = build_hamiltonian(cluster, space, jpm).tocsr()
    z_star = float(eigsh(hamiltonian, k=1, which="SA", tol=1e-10,
                         return_eigenvectors=False)[0])

    OUTPUT.mkdir(parents=True, exist_ok=True)
    # The sample size MUST be in the tag.  Without it a sampled run overwrites
    # the full result at the same (cluster, levels, coupling) and -- worse --
    # shares a checkpoint file with a full run that may be executing right now,
    # corrupting both.
    suffix = "" if columns is None else f"_n{columns}"
    tag = f"{name}_L{levels}_{jpm:+.3f}{suffix}".replace(".", "p")
    checkpoint = OUTPUT / f"ckpt_{tag}.npz" if resumable else None
    f_masked, q_block = masked_map(cluster, space, ice_rows, complement,
                                   hamiltonian, labels, z_star, chunk=chunk,
                                   checkpoint=checkpoint, limit=columns)
    if columns is not None and columns < len(ice_states):
        # Take the principal block of the columns actually solved.  P and the
        # complement stay the FULL ice manifold above -- truncating the ice
        # states themselves would push 2458 of them into Q and fold them as if
        # they were spinon states, which is a different operator entirely (the
        # continuum edge collapsed from 0.775 to 0.014 when this was done by
        # mistake, and K_8/K_6 moved 30%).
        f_masked = f_masked[:columns, :columns]
        ice_states = ice_states[:columns]
        labels = labels[:columns]
    edge = float(eigsh(q_block, k=1, which="SA", tol=1e-9,
                       return_eigenvectors=False)[0])

    summary = loop_tomography(f_masked, ice_states, cluster, labels,
                              graph_distance, real_distance)
    amplitudes = connected_class_amplitudes(summary)
    fit = fit_gauge_range(amplitudes)

    k6 = amplitudes.get(6, {}).get("K", np.nan)
    record = {
        "cluster": name,
        "levels": levels,
        "jpm": jpm,
        "space_size": int(len(space)),
        "full_space_size": int(
            comb(int(cluster.n_sites), int(cluster.n_sites) // 2)
            if cluster.n_sites <= 20 else -1),
        "n_ice": int(len(ice_states)),
        "z_star": z_star,
        "continuum_edge": edge,
        "edge_gap": edge - z_star,
        "k6": float(k6),
        "xi_gauge": fit.get("xi_gauge"),
        "xi_two_point": fit.get("xi_two_point"),
        "fit": fit,
        "runtime": time.perf_counter() - started,
    }
    for length, entry in amplitudes.items():
        record[f"K::{length}"] = entry
        if length != 6 and k6:
            record[f"k{length}_over_k6"] = float(entry["K"] / k6)
    record["geometry"] = summary

    (OUTPUT / f"tomo_{tag}.json").write_text(
        json.dumps(record, indent=1, default=float))
    np.savez_compressed(OUTPUT / f"tomo_{tag}.npz", f_masked=f_masked,
                        jpm=jpm, levels=levels, z_star=z_star)

    # scientific notation: several classes are killed outright by the mask
    # (every winding four-loop, for one), and a %.4f ratio renders both a true
    # zero and a 1e-6 tail as "0.0000"
    ladder = " ".join(
        f"K{length}/K6={entry['K'] / k6:.3e}"
        for length, entry in amplitudes.items() if length != 6 and k6)
    xi = record["xi_gauge"]
    print(f"{name} L={levels} jpm={jpm:+.3f} space={len(space):,} "
          f"gap={record['edge_gap']:.4f} K6={k6:.4e} {ladder}", flush=True)
    print(f"    xi={'n/a' if xi is None else f'{xi:.3f}'} from "
          f"{fit.get('n_perimeters', 0)} perimeters {fit.get('perimeters')} "
          f"(residual {fit.get('log_residual_rms', float('nan')):.3f}"
          f"{', TWO-POINT: not a fit' if fit.get('n_perimeters', 0) < 3 else ''})"
          f" ({record['runtime']:.0f}s)", flush=True)
    return record


def gate(name: str, couplings, context) -> int:
    """The truncation gate: is the L-drift smaller than the physics?"""
    depths = [1, 2, 3, 4] if name == "cubic16" else [1, 2]
    table: dict[tuple[int, float], dict] = {}
    for levels in depths:
        for jpm in couplings:
            try:
                table[(levels, jpm)] = run_point(name, levels, jpm, context)
            except (MemoryError, RuntimeError) as error:
                print(f"  L={levels} jpm={jpm:+.3f} FAILED: {error}",
                      flush=True)

    report = {"cluster": name, "couplings": list(couplings), "depths": depths}

    # On cubic-16 the deepest level reaches the whole fixed-Sz space, so its
    # answer is the untruncated one and the truncation error is measurable
    # directly rather than inferred from the L1 -> L2 drift.  This is the
    # number that transfers to FCC-32 as the error bar.
    exact_depth = depths[-1]
    exact_available = all(
        table.get((exact_depth, jpm), {}).get("space_size") ==
        table.get((exact_depth, jpm), {}).get("full_space_size")
        for jpm in couplings if (exact_depth, jpm) in table)
    # k6 is an absolute amplitude, not a ratio, and it is the ONE quantity
    # cubic-16 can still calibrate now that composite loops are excluded: the
    # hexagon is the only simple perimeter that survives the mask there.  It is
    # a weak proxy for the long loops -- a hexagon needs fewer virtual hops and
    # so converges faster -- but a measured floor beats no measurement.
    calibration: dict[str, dict] = {}
    for observable in ("k6", "k8_over_k6", "k10_over_k6"):
        per_level = {}
        for levels in depths[:-1]:
            errors = []
            for jpm in couplings:
                truncated = table.get((levels, jpm), {}).get(observable)
                reference = table.get((exact_depth, jpm), {}).get(observable)
                if truncated is None or not reference:
                    continue
                errors.append((truncated - reference) / reference)
            if errors:
                per_level[f"L{levels}"] = {
                    "signed_relative_error": [float(e) for e in errors],
                    "worst_magnitude": float(np.max(np.abs(errors))),
                    "all_biased_low": bool(all(e < 0 for e in errors)),
                }
                print(f"  {observable} at L={levels}: "
                      f"{100 * np.min(errors):+.1f}% to "
                      f"{100 * np.max(errors):+.1f}% vs L={exact_depth}",
                      flush=True)
        calibration[observable] = per_level
    report["calibration_vs_deepest"] = calibration
    report["deepest_is_untruncated"] = exact_available

    for observable in ("k6", "k8_over_k6", "k10_over_k6"):
        drift, values = [], []
        for jpm in couplings:
            first = table.get((1, jpm), {}).get(observable)
            second = table.get((2, jpm), {}).get(observable)
            if first is not None and second is not None:
                drift.append(abs(second - first))
                values.append(second)
        if len(values) < 2:
            report[observable] = {"note": "insufficient data"}
            continue
        spread = float(np.max(values) - np.min(values))
        worst = float(np.max(drift))

        # A ratio that is identically zero on this cluster passes the drift
        # criterion trivially (0 <= 0) while certifying nothing at all.  That
        # is a false pass, and it is the failure mode this gate exists to
        # prevent, so it is reported as NOT CALIBRATABLE rather than clear.
        reference = [table.get((exact_depth, jpm), {}).get(observable)
                     for jpm in couplings]
        degenerate = all(not value for value in reference)
        report[observable] = {
            "max_L1_to_L2_drift": worst,
            "inter_coupling_spread": spread,
            "drift_exceeds_physics": bool(worst > spread) and not degenerate,
            "degenerate_on_this_cluster": bool(degenerate),
            "values_at_L2": values,
        }
        if degenerate:
            print(f"  {observable}: identically zero on {name} -> NOT "
                  f"CALIBRATABLE here (every connected string of this "
                  f"perimeter winds on this cluster, so the mask deletes it)",
                  flush=True)
        else:
            print(f"  {observable}: worst L1->L2 drift {worst:.4f} vs "
                  f"inter-coupling spread {spread:.4f} -> "
                  f"{'BLOCKED' if worst > spread else 'clear'}", flush=True)

    blocked = any(
        isinstance(entry, dict) and entry.get("drift_exceeds_physics")
        for entry in report.values())
    uncalibratable = sorted(
        key for key, entry in report.items()
        if isinstance(entry, dict) and entry.get("degenerate_on_this_cluster"))
    calibrated = sorted(
        key for key, entry in report.items()
        if isinstance(entry, dict) and "max_L1_to_L2_drift" in entry
        and not entry.get("degenerate_on_this_cluster"))
    report["not_calibratable"] = uncalibratable
    report["calibrated"] = calibrated

    # "No observable failed" is not "the gate passed" when no observable could
    # be tested.  Once composite loops are excluded, cubic-16 has exactly one
    # surviving perimeter, so every K_L RATIO is degenerate there and a verdict
    # of "clear" would certify nothing at all -- the same vacuous pass this
    # gate already caught once for k10_over_k6.
    if not calibrated:
        verdict = (
            f"NOT CALIBRATABLE on {name}: every K_L ratio is degenerate here "
            f"({', '.join(uncalibratable) or 'no ratio survives the mask'}), "
            "so this cluster cannot bound the truncation error on xi_gauge at "
            "all.  That bound must come from self-convergence in levels on the "
            "production cluster, and until it exists every xi_gauge carries an "
            "unquantified truncation bias.")
    elif blocked:
        verdict = (
            "BLOCKED: the truncation drift is as large as the coupling trend; "
            "levels=1 is unusable and levels>=3 is required")
    else:
        verdict = (
            f"clear for {', '.join(calibrated)}: the truncation drift is "
            "smaller than the coupling trend; quote the drift as the error bar")
    if uncalibratable and calibrated:
        verdict += (
            f"; but {', '.join(uncalibratable)} cannot be calibrated on "
            f"{name} at all, so its truncation error must be established by "
            "self-convergence in levels on the production cluster instead")
    report["verdict"] = verdict
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / f"gate_{name}.json").write_text(
        json.dumps(report, indent=2, default=float))
    print(f"\nWP1 TRUNCATION GATE ({name}): {report['verdict']}")
    return 1 if blocked else 0


def class_census(name: str, context) -> int:
    """Which loop classes survive the mask at all -- pure combinatorics.

    Run before spending production time: `xi_gauge` needs several perimeters
    with nonzero masked amplitude, and cubic-16 turns out to have only two
    (`L = 6` and `L = 8`).  Every longer connected string on sixteen sites
    winds, so the mask deletes it and the "fit" degenerates to a two-point
    estimate.  This census answers, for free, whether the production cluster
    does better -- i.e. whether WP1 is measurable there at all.

    The amplitudes are irrelevant here, so the class structure is extracted by
    feeding the tomography a matrix of ones and reading `kept_by_mask`.
    """
    cluster, labels, graph_distance, real_distance = context
    ice_states = np.asarray(cluster.ice_states, dtype=np.uint64)
    ones = np.ones((len(ice_states),) * 2)
    summary = loop_tomography(ones, ice_states, cluster, labels,
                              graph_distance, real_distance)

    per_perimeter: dict[int, dict] = {}
    joined: dict[int, int] = {}
    for key, entry in summary.items():
        if key == "long" or entry["components"] != 1:
            continue
        if not entry.get("simple", True):
            joined[int(entry["perimeter"])] = (
                joined.get(int(entry["perimeter"]), 0) + entry["kept_by_mask"])
            continue
        slot = per_perimeter.setdefault(
            int(entry["perimeter"]),
            {"classes": 0, "pairs": 0, "kept": 0, "distinct_masks": 0})
        slot["classes"] += 1
        slot["pairs"] += entry["count"]
        slot["kept"] += entry["kept_by_mask"]
        slot["distinct_masks"] += entry["n_distinct_masks"]

    usable = [length for length, slot in sorted(per_perimeter.items())
              if slot["kept"] > 0 and length >= 6]
    print(f"\nconnected single-string classes on {name}:")
    print(f"{'L':>4} {'classes':>8} {'pairs':>9} {'kept by mask':>13} "
          f"{'survives':>9}")
    for length, slot in sorted(per_perimeter.items()):
        print(f"{length:>4} {slot['classes']:>8} {slot['pairs']:>9} "
              f"{slot['kept']:>13} {'yes' if slot['kept'] else 'NO':>9}")
    print(f"\nperimeters available for a K_L fit: {usable} "
          f"({len(usable)} points)")
    print("VERDICT:", "measurable -- three or more perimeters survive"
          if len(usable) >= 3 else
          "NOT measurable on this cluster: fewer than three perimeters "
          "survive the mask, so xi_gauge would be a two-point estimate, not "
          "a fit")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / f"census_{name}.json").write_text(json.dumps(
        {"cluster": name, "per_perimeter": per_perimeter,
         "usable_perimeters": usable, "measurable": len(usable) >= 3},
        indent=2, default=float))
    return 0 if len(usable) >= 3 else 1


def main() -> int:
    arguments = [v for v in sys.argv[1:] if not v.startswith("--")]
    gate_wanted = "--gate" in sys.argv
    census_wanted = "--classes" in sys.argv
    if not arguments:
        raise SystemExit(__doc__)

    name = arguments[0]
    if name not in CLUSTERS:
        raise SystemExit(f"unknown cluster {name}; choose from {list(CLUSTERS)}")
    basis, shape = CLUSTERS[name]
    cluster = geometry.build_cluster(basis, shape)
    context = (cluster, polarization_sector_labels(cluster),
               graph_distance_matrix(cluster),
               minimum_image_distance_matrix(cluster))
    print(f"{name}: {cluster.n_sites} sites, {cluster.n_ice} ice states",
          flush=True)

    if "--reanalyse" in sys.argv:
        # Recompute the tomography from the stored masked maps.  The expensive
        # part of a point is the block-CG fold, and `f_masked` is saved, so a
        # correction to the K_L statistics costs seconds per point instead of
        # the ~50 min the solve took.
        for path in sorted(OUTPUT.glob(f"tomo_{name}_*.npz")):
            with np.load(path) as data:
                f_masked = data["f_masked"]
                jpm = float(data["jpm"])
                levels = int(data["levels"])
            ice_states = np.asarray(cluster.ice_states, dtype=np.uint64)
            summary = loop_tomography(f_masked, ice_states, cluster,
                                      context[1], context[2], context[3])
            amplitudes = connected_class_amplitudes(summary)
            fit = fit_gauge_range(amplitudes)
            k6 = amplitudes.get(6, {}).get("K", float("nan"))
            record = {"cluster": name, "levels": levels, "jpm": jpm,
                      "k6": float(k6), "xi_gauge": fit.get("xi_gauge"),
                      "fit": fit, "source": path.name}
            for length, entry in amplitudes.items():
                record[f"K::{length}"] = entry
                if length != 6 and k6:
                    record[f"k{length}_over_k6"] = float(entry["K"] / k6)
            tag = f"{name}_L{levels}_{jpm:+.3f}".replace(".", "p")
            (OUTPUT / f"tomo_{tag}.json").write_text(
                json.dumps(record, indent=1, default=float))
            ladder = " ".join(
                f"K{length}/K6={entry['K'] / k6:.3e}"
                for length, entry in amplitudes.items() if length != 6 and k6)
            spreads = " ".join(
                f"L{length}:{entry['spread_over_K']:.1e}"
                for length, entry in amplitudes.items() if entry["K"])
            print(f"{name} L={levels} jpm={jpm:+.3f} K6={k6:.4e} {ladder}",
                  flush=True)
            print(f"    kept-only spread/K  {spreads}", flush=True)
            rungs = "  ".join(
                f"{r['from']}->{r['to']}: xi={r['xi']:.3f} (x{r['ratio']:.3f})"
                for r in fit.get("two_point_ladder", []))
            print(f"    two-point ladder: {rungs}", flush=True)
            xi = fit.get("xi_gauge")
            spread = fit.get("xi_spread_factor")
            print(f"    xi(generated, L>=8) = "
                  f"{'n/a' if xi is None else f'{xi:.3f}'}   "
                  f"spread across rungs "
                  f"{'n/a' if spread is None else f'{spread:.2f}x'}   "
                  f"-> {'exponential' if fit.get('is_exponential') else 'NOT a single exponential'}",
                  flush=True)
        return 0

    if "--sizes" in sys.argv:
        # The gate says levels >= 3 is needed for percent-level K_L; whether
        # that is affordable on FCC-32 is a question about space size and the
        # dense right-hand-side block, so measure it before budgeting.
        depth = int(arguments[1]) if len(arguments) > 1 else 3
        # WP2 needs the same sizing with pair flips switched on: those moves
        # change S^z by +-2, so the reachable space is much larger at the same
        # depth and WP2's budget is not WP1's.
        for jpmpm in (0.0, 1.0):
            label = "XXZ" if jpmpm == 0.0 else "with pair flips"
            for levels in range(1, depth + 1):
                started = time.perf_counter()
                try:
                    space = bfs_space(cluster, levels, jpmpm=jpmpm)
                except MemoryError:
                    print(f"{name} L={levels} ({label}): MemoryError",
                          flush=True)
                    break
                rhs_bytes = len(space) * 256 * 8
                print(f"{name} L={levels} ({label}): {len(space):,} states, "
                      f"256-column dense RHS = {rhs_bytes / 2**30:.2f} GiB "
                      f"({time.perf_counter() - started:.1f}s)", flush=True)
        return 0
    if census_wanted:
        return class_census(name, context)
    if gate_wanted:
        return gate(name, [float(v) for v in arguments[1:]], context)

    chunk = CHUNK
    columns = None
    for token in sys.argv:
        if token.startswith("--chunk="):
            chunk = int(token.split("=", 1)[1])
        if token.startswith("--columns="):
            columns = int(token.split("=", 1)[1])
    resumable = "--resume" in sys.argv

    levels = int(arguments[1])
    for jpm in [float(v) for v in arguments[2:]]:
        run_point(name, levels, jpm, context, chunk=chunk,
                  resumable=resumable, columns=columns)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
