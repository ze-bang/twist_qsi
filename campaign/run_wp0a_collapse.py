#!/usr/bin/env python3
"""WP0a: is the residue Z the organizing variable, or are the bare couplings?

The loop-class anatomy of the winding-free map produces ratios -- long/hex,
K_8/K_6, |V/g| -- that grow as the ice manifold dresses.  Two stories fit the
existing data equally well:

  * *Z organizes*: the ratios are functions of the residue `1 - Z`, i.e. of how
    much of the state has leaked out of the ice manifold, whatever coupling
    put it there.  Then `1 - Z` is the physical scaling coordinate and the
    paper opens with it.
  * *the bare couplings organize*: the ratios are functions of `|Jpm|` (and
    `Jpmpm`) directly, and `Z` is one more downstream observable.

The existing evidence is ambiguous and points slightly the wrong way for the
Z story: at matched `|Jpm| ~ 0.05` the two signs give long/hex 0.317 and 0.321
-- indistinguishable -- while their residues differ, `Z = 0.929` against
`0.890`.  A single matched-|Jpm| ladder across the sign settles it, because
the sign of `Jpm` moves `Z` while holding `|Jpm|` fixed.

The discriminator implemented here is a prediction test, not a correlation.
For each candidate coordinate `x` (either `1 - Z` or `|Jpm|`), fit a monotone
curve through the NEGATIVE-Jpm branch alone, then predict the POSITIVE-Jpm
points from it and measure the error.  A coordinate that genuinely organizes
the physics predicts across the sign it never saw; a coordinate that merely
correlates within one branch does not.  Points whose `x` falls outside the
fitted range are excluded rather than extrapolated.

Two modes:

  --plane <jpm> <pp> ...   compute the masked map and residue at a
                           `(Jpm, Jpmpm)` point on the full 65,536-state
                           basis (pair flips break Sz, so there is no
                           smaller exact basis), caching the tomography;
  --analyse                gather every cached anatomy npz plus any plane
                           caches, run the tomography, and report the verdict.

The analyse mode recomputes the tomography from the `f_masked` arrays already
stored by `run_feshbach_anatomy.py`, so the whole negative ladder costs
nothing beyond reading it back.

The map is evaluated at the exact raw ground-state energy `z_star` in both
modes -- the convention the cached ladder already used -- so the two signs and
the plane points are compared at the same point of the map.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.sparse.linalg import eigsh

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "notes"))
sys.path.insert(0, str(ROOT / "campaign"))

import recompute_finite_size_artifact as geometry  # noqa: E402
from qsi_campaign.exact_band import (  # noqa: E402
    full_spin_basis,
    microscopic_character_hamiltonian,
)
from qsi_campaign.protocol import polarization_sector_labels  # noqa: E402
from qsi_campaign.tomography import (  # noqa: E402
    connected_class_amplitudes,
    graph_distance_matrix,
    loop_tomography,
    minimum_image_distance_matrix,
)
from run_multi_anchor_band import block_cg  # noqa: E402

ANATOMY = ROOT / "campaign" / "outputs" / "feshbach_anatomy"
OUTDIR = ROOT / "campaign" / "outputs" / "wp0_preflight"

OBSERVABLES = ("long_over_hex", "k8_over_k6", "k10_over_k6", "abs_rk_ratio")


# --------------------------------------------------------------- measurement


def measure(f_masked, cluster, labels, graph_distance, real_distance) -> dict:
    """Every WP0a ratio, from one masked map."""
    ice_states = np.asarray(cluster.ice_states, dtype=np.uint64)
    tomography = loop_tomography(f_masked, ice_states, cluster, labels,
                                 graph_distance, real_distance)
    amplitudes = connected_class_amplitudes(tomography)
    k6 = amplitudes.get(6, {}).get("K", np.nan)

    off = np.asarray(f_masked).real.copy()
    np.fill_diagonal(off, 0.0)
    states = np.asarray([int(s) for s in ice_states], dtype=np.uint64)
    perimeter = np.bitwise_count(states[:, None] ^ states[None, :])
    hex_norm = float(np.linalg.norm(np.where(perimeter == 6, off, 0.0)))
    long_norm = float(np.linalg.norm(np.where(perimeter >= 8, off, 0.0)))

    result = {
        "k6": float(k6),
        "long_over_hex": long_norm / hex_norm if hex_norm else np.nan,
        "worst_within_class_spread": float(max(
            (entry["within_class_spread"] for entry in amplitudes.values()),
            default=0.0)),
    }
    for length, entry in amplitudes.items():
        result[f"K::{length}"] = entry
        if length != 6 and k6:
            result[f"k{length}_over_k6"] = float(entry["K"] / k6)
    return result


# ------------------------------------------------------------- plane points


def plane_point(jpm: float, jpmpm: float, cluster, labels,
                graph_distance, real_distance) -> dict:
    """Masked map + residue at one `(Jpm, Jpmpm)` point, full spin basis."""
    started = time.perf_counter()
    states = full_spin_basis(cluster.n_sites)
    state_index = {int(s): i for i, s in enumerate(states)}
    ice_indices = np.asarray(
        [state_index[int(s)] for s in cluster.ice_states], dtype=np.int64)
    complement = np.setdiff1d(np.arange(len(states)), ice_indices)
    same = labels[:, None] == labels[None, :]

    hamiltonian = microscopic_character_hamiltonian(
        cluster, states, jpm, np.zeros(3), jpmpm=jpmpm).real.tocsr()
    z_star = float(eigsh(hamiltonian, k=1, which="SA", tol=1e-11,
                         return_eigenvectors=False)[0])

    q_block = hamiltonian[complement][:, complement].tocsr()
    coupling = np.asarray(hamiltonian[complement][:, ice_indices].todense())
    ice_diag = hamiltonian[ice_indices][:, ice_indices].toarray()
    edge = float(eigsh(q_block, k=1, which="SA", tol=1e-9,
                       return_eigenvectors=False)[0])

    def solve(z: float):
        """(QHQ - z)^{-1} QHP and the masked map there."""
        solution = block_cg(lambda v, s=z: q_block @ v - s * v, coupling,
                            np.zeros_like(coupling))
        f_matrix = ice_diag - coupling.T @ solution
        f_matrix = 0.5 * (f_matrix + f_matrix.T)
        return solution, np.where(same, f_matrix, 0.0)

    # the map at the raw ground-state energy: the ladder's convention
    _, f_masked = solve(z_star)

    # the self-consistent lowest masked root, by secant, for the residue Z
    ceiling = edge - 1e-6 * max(abs(edge), 1.0)
    z_previous, g_previous = z_star, None
    z_current = z_star
    solution, f_current = None, f_masked
    residual = np.inf
    for _ in range(16):
        values, vectors = np.linalg.eigh(f_current)
        g_current = float(values[0]) - z_current
        residual = abs(g_current)
        if residual < 1e-11:
            break
        if g_previous is None or abs(g_current - g_previous) < 1e-16:
            step = g_current
        else:
            step = -g_current * (z_current - z_previous) / (
                g_current - g_previous)
        z_previous, g_previous = z_current, g_current
        z_current = float(np.clip(z_current + step, z_star - 2.0, ceiling))
        solution, f_current = solve(z_current)
    values, vectors = np.linalg.eigh(f_current)
    if solution is None:
        solution, _ = solve(z_current)
    slope = -float(np.sum((solution @ vectors[:, 0]) ** 2))

    record = {
        "jpm": jpm,
        "jpmpm": jpmpm,
        "z_star": z_star,
        "continuum_edge": edge,
        "masked_root": z_current,
        "root_residual": residual,
        "z_ground": 1.0 / (1.0 - slope),
        "runtime": time.perf_counter() - started,
    }
    record.update(measure(f_masked, cluster, labels, graph_distance,
                          real_distance))
    return record, f_masked


# ------------------------------------------------------------------ analysis


def gather(cluster, labels, graph_distance, real_distance) -> list[dict]:
    points = []
    for path in sorted(ANATOMY.glob("anatomy_*.npz")):
        with np.load(path) as data:
            if "f_masked" not in data:
                continue
            f_masked = data["f_masked"]
            jpm = float(data["jpm"])
            weight = data["ice_weight"]
            energies = data["masked_energies"]
            converged = data["masked_converged"]
        if not converged.any():
            continue
        ground = int(np.nanargmin(np.where(converged, energies, np.inf)))
        record = {
            "source": path.name,
            "jpm": jpm,
            "jpmpm": 0.0,
            "z_ground": float(weight[ground]),
            "n_masked_roots": int(converged.sum()),
        }
        record.update(measure(f_masked, cluster, labels, graph_distance,
                              real_distance))
        json_path = path.with_suffix(".json")
        if json_path.exists():
            stored = json.loads(json_path.read_text())
            record["abs_rk_ratio"] = abs(float(stored.get("rk_ratio", np.nan)))
            record["v_eff"] = float(stored.get("v_eff", np.nan))
        points.append(record)

    for path in sorted(OUTDIR.glob("plane_*.json")):
        record = json.loads(path.read_text())
        record["source"] = path.name
        # Recompute the tomography from the stored masked map rather than
        # trusting the stored ratios.  The ladder points are recomputed above,
        # so reading the plane's cached numbers would compare K_L values
        # produced by two different definitions -- and the definition changed
        # when the mask's zeros were removed from the class averages.  Mixing
        # them silently reorders which pair is "worst" and corrupts every
        # cross-family comparison.
        cached = path.with_suffix(".npz")
        if cached.exists():
            with np.load(cached) as data:
                record.update(measure(data["f_masked"], cluster, labels,
                                      graph_distance, real_distance))
        else:
            record["stale_tomography"] = True
        points.append(record)
    return points


def prediction_test(train, test, observable, coordinate) -> dict:
    """Fit `train` in `coordinate`, predict `test`, never extrapolating."""
    train = sorted(
        (p for p in train
         if np.isfinite(p.get(observable, np.nan))
         and np.isfinite(p.get(coordinate, np.nan))),
        key=lambda p: p[coordinate])
    test = [p for p in test
            if np.isfinite(p.get(observable, np.nan))
            and np.isfinite(p.get(coordinate, np.nan))]
    if len(train) < 4 or not test:
        return {"n_train": len(train), "n_test": 0, "rms_relative": None,
                "note": "insufficient data"}

    x_train = np.array([p[coordinate] for p in train])
    y_train = np.array([p[observable] for p in train])
    keep = np.concatenate([[True], np.diff(x_train) > 1e-12])
    x_train, y_train = x_train[keep], y_train[keep]
    if len(x_train) < 2:
        return {"n_train": int(len(x_train)), "n_test": 0,
                "rms_relative": None,
                "note": "coordinate is degenerate on the training set"}
    curve = PchipInterpolator(x_train, y_train)

    errors, tested, outside = [], [], 0
    for p in test:
        x = p[coordinate]
        if not (x_train[0] <= x <= x_train[-1]):
            outside += 1
            continue
        predicted = float(curve(x))
        actual = p[observable]
        errors.append(abs(predicted - actual) / max(abs(actual), 1e-300))
        tested.append({"jpm": p["jpm"], "jpmpm": p.get("jpmpm", 0.0),
                       "x": float(x), "actual": float(actual),
                       "predicted": predicted, "relative_error": errors[-1]})
    if not errors:
        return {"n_train": int(len(x_train)), "n_test": 0,
                "rms_relative": None, "n_outside_range": outside,
                "note": "no test point inside the fitted range"}

    # A coordinate can look excellent by covering only the test points nearest
    # the training set and dropping the rest as extrapolation -- which is
    # exactly what happens here when the Jpm ladder dissolves at |Jpm| >= 0.24
    # and so cannot reach the low-Z end the Jpmpm column occupies.  Report the
    # coverage, and refuse to certify a coordinate that tested less than half
    # of the points offered to it.
    coverage = len(errors) / (len(errors) + outside)
    spread = float(np.max([e["actual"] for e in tested])
                   - np.min([e["actual"] for e in tested]))
    return {
        "n_train": int(len(x_train)),
        "n_test": len(errors),
        "n_outside_range": outside,
        "coverage": float(coverage),
        "tested_range_of_observable": spread,
        "rms_relative": float(np.sqrt(np.mean(np.square(errors)))),
        "max_relative": float(np.max(errors)),
        "certifiable": bool(coverage >= 0.5 and len(errors) >= 3),
        "tested": tested,
    }


COLLAPSE_TOLERANCE = 0.10          # a 1-D collapse must predict to ~10%
NEARLY_EQUAL_Z = 0.05              # how close in 1-Z counts as "the same Z"


def single_valuedness(points, observable, coordinate="one_minus_z",
                      tolerance: float = NEARLY_EQUAL_Z) -> dict:
    """Is the observable a single-valued function of the coordinate?

    The sharpest form of the question, and the one that needs neither a fit nor
    overlapping ranges: find two points from DIFFERENT families (the Jpm
    ladder, and each Jpmpm column) that sit at essentially the same `1 - Z` and
    compare their loop ratios.  If the ratio is a function of `1 - Z` alone
    they must agree; a large disagreement at matched residue falsifies the
    hypothesis outright, with no interpolation and no extrapolation.

    Families are compared only across, never within: two points of the same
    column trivially differ in `1 - Z` and prove nothing.
    """
    # Comparing two quantities that are both essentially zero produces a huge
    # ratio out of nothing: |V/g| stays below 0.022 everywhere, so a 0.0030 vs
    # 0.0004 pair "fails" single-valuedness while saying only that the
    # Rokhsar-Kivelson potential is negligible at both points.  Require the
    # larger member to be a non-negligible fraction of the observable's range.
    magnitudes = [abs(p[observable]) for p in points
                  if np.isfinite(p.get(observable, np.nan))]
    floor = 0.05 * max(magnitudes) if magnitudes else 0.0

    usable = [
        p for p in points
        if np.isfinite(p.get(observable, np.nan))
        and np.isfinite(p.get(coordinate, np.nan))
        and p.get(observable, 0.0) != 0.0
    ]

    def family(point):
        """The pp = 0 ladder is ONE family; each Jpmpm column is another.

        Two ladder points are not an independent check of each other: they are
        the same curve, and a pair of adjacent ones with nearly equal Z says
        only that Z is locally flat along the ladder.  The hypothesis is
        falsified by comparing points reached through DIFFERENT dials.
        """
        return ("ladder" if point.get("jpmpm", 0.0) == 0.0
                else f"column{point['jpm']:+.3f}")

    worst = None
    for index, first in enumerate(usable):
        for second in usable[index + 1:]:
            if family(first) == family(second):
                continue
            if max(abs(first[observable]), abs(second[observable])) < floor:
                continue               # both negligible: ratio is noise
            separation = abs(first[coordinate] - second[coordinate])
            if separation > tolerance:
                continue
            high, low = sorted((first[observable], second[observable]),
                               reverse=True)
            ratio = high / max(abs(low), 1e-300)
            if worst is None or ratio > worst["ratio"]:
                worst = {
                    "ratio": float(ratio),
                    "coordinate_separation": float(separation),
                    "point_a": {"jpm": first["jpm"],
                                "jpmpm": first.get("jpmpm", 0.0),
                                coordinate: float(first[coordinate]),
                                observable: float(first[observable])},
                    "point_b": {"jpm": second["jpm"],
                                "jpmpm": second.get("jpmpm", 0.0),
                                coordinate: float(second[coordinate]),
                                observable: float(second[observable])},
                }
    if worst is None:
        return {"tested": False,
                "note": "no cross-family pair at matched coordinate"}
    worst["tested"] = True
    worst["single_valued"] = bool(worst["ratio"] < 1.0 + COLLAPSE_TOLERANCE)
    return worst


def analyse(points) -> dict:
    """The two legs of the WP0a discriminator.

    *Sign leg* (weak but direct): train on the negative `Jpm` ladder, predict
    the positive one.  Here `1 - Z` and `|Jpm|` are genuine rival coordinates
    and the test is a head-to-head.  It is confined to `|Jpm| <= 0.06`, since
    the positive side is expected to order beyond ~0.05 and a dressed-gauge
    reading of an ordered band would be meaningless.

    *Jpmpm leg* (the strong one): train on the same negative ladder, predict
    the `Jpmpm` column at fixed `Jpm = -0.20`.  This is not a head-to-head:
    `|Jpm|` is constant along the column by construction, so it predicts one
    number for every point and cannot collapse anything.  Its error is quoted
    as the null baseline -- the spread the observables actually have -- and the
    question is whether `1 - Z` collapses that spread onto the ladder's curve.
    A `1 - Z` error far below the null is a genuine one-dimensional collapse;
    an error comparable to it means the loop structure needs both couplings and
    `Z` is a downstream observable, not a scaling variable.

    Only band-complete points enter either fit.  A point that has lost roots
    has no certified winding-free band, so its loop ratios are not statements
    about a gauge theory.
    """
    for p in points:
        p["one_minus_z"] = 1.0 - p["z_ground"]
        p["abs_jpm"] = abs(p["jpm"])

    stale = [p["source"] for p in points if p.get("stale_tomography")]
    if stale:
        raise SystemExit(
            "refusing to compare points whose tomography could not be "
            f"recomputed from a stored masked map: {stale}. Two K_L "
            "definitions in one comparison is not a comparison.")

    complete = [p for p in points if p.get("n_masked_roots", 90) == 90]
    dropped = [
        {"jpm": p["jpm"], "jpmpm": p.get("jpmpm", 0.0),
         "n_masked_roots": p.get("n_masked_roots")}
        for p in points if p.get("n_masked_roots", 90) != 90
    ]
    ladder = [p for p in complete if p.get("jpmpm", 0.0) == 0.0]
    negative = [p for p in ladder if p["jpm"] < 0]
    positive = [p for p in ladder if 0 < p["jpm"] <= 0.06]
    excluded_positive = [p for p in ladder if p["jpm"] > 0.06]
    column = [p for p in complete if p.get("jpmpm", 0.0) != 0.0]

    report = {
        "n_points": len(points),
        "n_band_complete": len(complete),
        "dropped_incomplete_band": dropped,
        "n_train_negative": len(negative),
        "n_test_positive": len(positive),
        "n_test_column": len(column),
        "positive_excluded_as_ordered": [p["jpm"] for p in excluded_positive],
        "tests": {},
    }

    # Column-vs-column: the only leg that stresses the hypothesis without
    # leaving certified data.  Two Jpmpm columns at different fixed Jpm overlap
    # in 1 - Z while differing in the coupling, so a genuine 1-D collapse must
    # predict one from the other.
    columns: dict[float, list] = {}
    for p in column:
        columns.setdefault(p["jpm"], []).append(p)
    column_pair = sorted(columns)[:2] if len(columns) >= 2 else []

    for observable in OBSERVABLES:
        values = [p.get(observable) for p in complete
                  if np.isfinite(p.get(observable, np.nan))]
        # An observable that is identically zero everywhere is predicted
        # perfectly by any coordinate and certifies nothing.  Refuse it
        # outright rather than scoring it 0.0% and calling that a collapse.
        if not values or float(np.ptp(values)) < 1e-12:
            report["tests"][observable] = {
                "verdict": "degenerate: identically constant on this cluster, "
                           "carries no information about any coordinate"}
            continue

        entry = {"sign": {}, "jpmpm": {}, "column_vs_column": {}}
        for coordinate in ("one_minus_z", "abs_jpm"):
            entry["sign"][coordinate] = prediction_test(
                negative, positive, observable, coordinate)
            entry["jpmpm"][coordinate] = prediction_test(
                negative, column, observable, coordinate)
            if column_pair:
                entry["column_vs_column"][coordinate] = prediction_test(
                    columns[column_pair[0]], columns[column_pair[1]],
                    observable, coordinate)

        entry["single_valuedness"] = single_valuedness(complete, observable)

        # rank the legs by what they can actually certify.  The
        # single-valuedness check outranks every fit: it is a direct
        # measurement at matched residue, needs no interpolation, and a
        # failure there cannot be rescued by any amount of curve fitting.
        direct = entry["single_valuedness"]
        if direct.get("tested") and not direct["single_valued"]:
            entry["decided_by"] = "single_valuedness"
            entry["verdict"] = "no collapse: Z is downstream"
            report["tests"][observable] = entry
            continue

        strong = entry["column_vs_column"].get("one_minus_z", {})
        sign_z = entry["sign"].get("one_minus_z", {})
        sign_j = entry["sign"].get("abs_jpm", {})
        if strong.get("certifiable"):
            z_error = strong["rms_relative"]
            entry["decided_by"] = "column_vs_column"
            entry["verdict"] = ("Z collapses"
                                if z_error < COLLAPSE_TOLERANCE
                                else "no collapse: Z is downstream")
        elif sign_z.get("certifiable") and sign_j.get("certifiable"):
            entry["decided_by"] = "sign"
            z_error = sign_z["rms_relative"]
            j_error = sign_j["rms_relative"]
            entry["verdict"] = (
                "bare coupling organizes" if j_error < z_error / 1.5 else
                "Z collapses" if z_error < j_error / 1.5 else "ambiguous")
        else:
            entry["decided_by"] = "none"
            entry["verdict"] = "untested: no leg has adequate coverage"
        report["tests"][observable] = entry

    verdicts = [e["verdict"] for e in report["tests"].values()
                if e["verdict"] in ("Z collapses", "bare coupling organizes",
                                    "no collapse: Z is downstream")]
    if verdicts and all(v == "Z collapses" for v in verdicts):
        report["conclusion"] = (
            "1 - Z is a genuine one-dimensional scaling coordinate: it "
            "predicts the Jpmpm column from the Jpm ladder. Open the paper "
            "with Z and organize WP1 by it.")
    elif verdicts and not any(v == "Z collapses" for v in verdicts):
        report["conclusion"] = (
            "the loop structure does not collapse onto 1 - Z: it needs the "
            "couplings. Organize WP1 by coupling and report Z as a "
            "downstream observable, not a scaling variable.")
    elif not verdicts:
        report["conclusion"] = (
            "UNTESTED: no leg had adequate coverage. Do not claim either way; "
            "organize WP1 by coupling as the safe default.")
    else:
        report["conclusion"] = (
            "collapse is observable-dependent: organize WP1 by coupling (the "
            "safe default) and quote the Z collapse only for the observables "
            "where a certifiable leg actually shows it.")
    report["points"] = points
    return report


def main() -> int:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    cluster = geometry.build_cluster("cubic", (1, 1, 1))
    labels = polarization_sector_labels(cluster)
    graph_distance = graph_distance_matrix(cluster)
    real_distance = minimum_image_distance_matrix(cluster)

    if "--plane" in sys.argv:
        values = [float(v) for v in sys.argv[sys.argv.index("--plane") + 1:]]
        for jpm, jpmpm in zip(values[0::2], values[1::2]):
            record, f_masked = plane_point(jpm, jpmpm, cluster, labels,
                                           graph_distance, real_distance)
            tag = f"jpm{jpm:+.4f}_pp{jpmpm:+.4f}".replace(".", "p")
            (OUTDIR / f"plane_{tag}.json").write_text(
                json.dumps(record, indent=1, default=float))
            np.savez_compressed(OUTDIR / f"plane_{tag}.npz",
                                f_masked=f_masked, jpm=jpm, jpmpm=jpmpm)
            print(f"jpm={jpm:+.3f} pp={jpmpm:+.3f} Z={record['z_ground']:.4f} "
                  f"root={record['masked_root']:.9f} "
                  f"(residual {record['root_residual']:.1e}) "
                  f"long/hex={record['long_over_hex']:.4f} "
                  f"K8/K6={record.get('k8_over_k6', float('nan')):.4f} "
                  f"({record['runtime']:.0f}s)", flush=True)
        return 0

    points = gather(cluster, labels, graph_distance, real_distance)
    report = analyse(points)

    print(f"{len(points)} points "
          f"({sum(1 for p in points if p['jpm'] < 0)} negative, "
          f"{sum(1 for p in points if p['jpm'] > 0)} positive, "
          f"{sum(1 for p in points if p.get('jpmpm', 0.0) != 0.0)} off-axis)")
    print()
    header = (f"{'Jpm':>8} {'pp':>6} {'Z':>7} {'1-Z':>7} {'long/hex':>9} "
              f"{'K8/K6':>8} {'K10/K6':>8} {'|V/g|':>7} {'K6':>11}")
    print(header)
    print("-" * len(header))
    for p in sorted(points, key=lambda p: (p.get("jpmpm", 0.0), p["jpm"])):
        print(f"{p['jpm']:+8.3f} {p.get('jpmpm', 0.0):+6.3f} "
              f"{p['z_ground']:7.4f} {p['one_minus_z']:7.4f} "
              f"{p.get('long_over_hex', float('nan')):9.4f} "
              f"{p.get('k8_over_k6', float('nan')):8.4f} "
              f"{p.get('k10_over_k6', float('nan')):8.4f} "
              f"{p.get('abs_rk_ratio', float('nan')):7.4f} "
              f"{p.get('k6', float('nan')):11.4e}")
    print()
    def rendered(leg, coordinate):
        result = leg.get(coordinate, {})
        value = result.get("rms_relative")
        if value is None:
            return "n/a"
        flag = "" if result.get("certifiable") else "  [NOT CERTIFIABLE]"
        return (f"{value:6.1%} (n={result.get('n_test', 0)}, "
                f"{result.get('n_outside_range', 0)} out of range){flag}")

    for observable, entry in report["tests"].items():
        print(f"{observable}:")
        if "sign" not in entry:
            print(f"    -> {entry['verdict']}")
            continue
        print(f"    sign leg    (+Jpm from -Jpm):  "
              f"via (1-Z) {rendered(entry['sign'], 'one_minus_z')}")
        print(f"                                   "
              f"via |Jpm| {rendered(entry['sign'], 'abs_jpm')}")
        print(f"    Jpmpm leg   (column from ladder): "
              f"via (1-Z) {rendered(entry['jpmpm'], 'one_minus_z')}")
        print(f"    column vs column:              "
              f"via (1-Z) {rendered(entry['column_vs_column'], 'one_minus_z')}")
        print(f"                                   "
              f"via |Jpm| {rendered(entry['column_vs_column'], 'abs_jpm')}")
        direct = entry.get("single_valuedness", {})
        if direct.get("tested"):
            a, b = direct["point_a"], direct["point_b"]
            print(f"    single-valuedness: worst cross-family pair at matched "
                  f"1-Z (within {direct['coordinate_separation']:.3f}):")
            print(f"        (Jpm={a['jpm']:+.3f}, pp={a['jpmpm']:+.3f}) "
                  f"1-Z={a['one_minus_z']:.4f} -> {a[observable]:.4f}")
            print(f"        (Jpm={b['jpm']:+.3f}, pp={b['jpmpm']:+.3f}) "
                  f"1-Z={b['one_minus_z']:.4f} -> {b[observable]:.4f}")
            print(f"        ratio {direct['ratio']:.2f}x -> "
                  f"{'single valued' if direct['single_valued'] else 'NOT a function of 1-Z'}")
        print(f"    -> {entry['verdict']}  (decided by {entry['decided_by']})")
    print()
    print("WP0a CONCLUSION:", report["conclusion"])

    path = OUTDIR / "wp0a_collapse.json"
    path.write_text(json.dumps(report, indent=2, default=float))
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
