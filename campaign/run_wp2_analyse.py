#!/usr/bin/env python3
"""WP2: does any certified NN-XYZ point reproduce the 25 mK anomaly?

The experiment (Smith et al., Ce2Hf2O7) shows two specific-heat features, at
about 65 mK and 25 mK -- a ratio of 0.385.  The 25 mK feature has been read as
the emergent photon.  The campaign's finding is that identifications of that
kind, calibrated on small periodic clusters, inherited the winding artifact:
naive periodic ED puts its ARTIFACT peak at ~29 mK, right on top of the
observed anomaly, while the winding-free gauge peak at the same couplings sits
near 7 mK.

This script asks the question the plan poses, over the whole fit-constrained
region rather than at one seed: **is there ANY certified parameter set whose
winding-free two-peak ratio reaches 0.385?**  A clean null is the expected and
equally publishable outcome -- a rigorous nearest-neighbour exclusion, which is
precisely the open question the experimental paper raises.

Two rules the reading must obey, both learned the hard way in this campaign.

*Certification.*  Only points with a complete 90/90 band carry peak positions.
A partial band leaves ice-descended raw levels in the ensemble beside their
replacements and invents features: NLC_B at 25/90 produced a spurious middle
peak that reads as a perfectly plausible ratio of 0.072.

*Resolved versus merged.*  A local maximum is not a peak.  The campaign's
taxonomy requires the curve to dip below 70% of the smaller of the two maxima
between them; otherwise the feature is a shoulder and is reported as merged,
with no ratio quoted.  Several grid points show two low-lying maxima a factor
of three apart in temperature, and calling the lower one "the photon" without
this test would manufacture exactly the kind of claim this campaign exists to
retract.

Usage: run_wp2_analyse.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PLANE = ROOT / "campaign" / "outputs" / "jpmpm_fold_plane"
OUTDIR = ROOT / "campaign" / "outputs" / "wp2_gate"

GRID = np.geomspace(1e-5, 1.5, 4000)
MEASURED_RATIO = 25.0 / 65.0           # Ce2Hf2O7, Smith et al.
DIP_FRACTION = 0.70                    # the campaign's resolved-peak criterion


def heat_capacity(levels: np.ndarray) -> np.ndarray:
    shifted = np.sort(np.asarray(levels, float))
    shifted = shifted - shifted.min()
    beta = 1.0 / GRID[:, None]
    weight = np.exp(-beta * shifted[None, :])
    partition = weight.sum(axis=1)
    mean = (weight * shifted[None, :]).sum(axis=1) / partition
    second = (weight * shifted[None, :] ** 2).sum(axis=1) / partition
    return (second - mean * mean) / GRID**2


def maxima(curve: np.ndarray) -> list[int]:
    interior = np.flatnonzero(
        (curve[1:-1] > curve[:-2]) & (curve[1:-1] >= curve[2:])) + 1
    return [int(i) for i in interior]


def resolved_pair(curve: np.ndarray, peaks: list[int]):
    """The gauge/spinon pair, if the dip between them clears the criterion.

    The spinon peak is the highest-temperature maximum; the gauge candidate is
    the tallest maximum below it.  They count as two features only if the curve
    between them falls below ``DIP_FRACTION`` of the smaller one.
    """
    if len(peaks) < 2:
        return None, "single maximum"
    spinon = peaks[-1]
    below = peaks[:-1]
    gauge = max(below, key=lambda i: curve[i])
    valley = curve[gauge:spinon].min()
    smaller = min(curve[gauge], curve[spinon])
    if valley >= DIP_FRACTION * smaller:
        return None, (f"merged: dip {valley / smaller:.2f} of the smaller peak, "
                      f"needs < {DIP_FRACTION}")
    return (gauge, spinon), "resolved"


def main() -> int:
    # A point can have both an anchor-solve file and a `_bordered` one.  They
    # disagree, and the bordered solver is the authority -- at (-0.050, 0.250)
    # the anchor file says 0/90 and the bordered one 86/90.  Keeping both
    # double-counts the grid and puts anchor-solve artifacts in the table
    # alongside real results, so prefer the bordered file wherever it exists.
    by_point: dict[tuple[str, str], Path] = {}
    for path in sorted(PLANE.glob("point_*.npz")):
        if not path.with_suffix(".json").exists():
            continue
        stem = path.stem.replace("_bordered", "")
        key = (stem, "")
        if key not in by_point or "_bordered" in path.stem:
            by_point[key] = path

    records = []
    for path in sorted(by_point.values()):
        meta = path.with_suffix(".json")
        stored = json.loads(meta.read_text())
        n_found = stored.get("n_found")
        with np.load(path) as data:
            if "fold_levels" not in data:
                continue
            levels = data["fold_levels"]
            jpm = float(data["jpm"])
            jpmpm = float(data["jpmpm"])

        # A binary 90/90 threshold throws away the distinction between one
        # dissolved branch and sixty-five of them.  At 89/90 a single
        # ice-descended raw level remains in the ensemble beside 89
        # replacements -- a ~1% contamination, not a broken calculation -- and
        # that point happens to carry the largest ratio on the grid, so
        # silently discarding it would hide the strongest evidence against the
        # null.  Report it separately instead of pretending the choice is
        # obvious.
        entry = {"jpm": jpm, "jpmpm": jpmpm, "n_found": n_found,
                 "source": path.name, "certified": n_found == 90,
                 "near_certified": n_found is not None and 85 <= n_found < 90}
        if not (entry["certified"] or entry["near_certified"]):
            entry["status"] = f"uncertified band {n_found}/90"
            records.append(entry)
            continue

        curve = heat_capacity(levels)
        peaks = maxima(curve)
        pair, status = resolved_pair(curve, peaks)
        entry["status"] = status
        entry["n_maxima"] = len(peaks)
        entry["maxima_T"] = [float(GRID[i]) for i in peaks]
        if pair is not None:
            gauge, spinon = pair
            entry["gauge_T"] = float(GRID[gauge])
            entry["spinon_T"] = float(GRID[spinon])
            entry["ratio"] = float(GRID[gauge] / GRID[spinon])
        records.append(entry)

    certified = [r for r in records
                 if r.get("ratio") is not None and r["certified"]]
    certified.sort(key=lambda r: -r["ratio"])
    near = [r for r in records
            if r.get("ratio") is not None and r.get("near_certified")]
    near.sort(key=lambda r: -r["ratio"])

    print(f"{len(records)} plane points, "
          f"{sum(1 for r in records if r['certified'])} with a complete band, "
          f"{len(certified)} with a resolved two-peak structure\n")
    header = (f"{'Jpm':>8} {'pp':>7} {'band':>6} {'gauge T':>9} "
              f"{'spinon T':>9} {'ratio':>7}  status")
    print(header)
    print("-" * (len(header) + 8))
    for r in sorted(records, key=lambda r: (r["jpm"], r["jpmpm"])):
        ratio = r.get("ratio")
        print(f"{r['jpm']:+8.3f} {r['jpmpm']:+7.3f} "
              f"{str(r['n_found']) + '/90':>6} "
              f"{r.get('gauge_T', float('nan')):9.5f} "
              f"{r.get('spinon_T', float('nan')):9.5f} "
              f"{'   --  ' if ratio is None else f'{ratio:7.4f}'}  "
              f"{r['status']}")

    print()
    if certified:
        best = certified[0]
        print(f"BEST certified ratio: {best['ratio']:.4f} at "
              f"(Jpm, Jpmpm) = ({best['jpm']:+.3f}, {best['jpmpm']:+.3f})")
        print(f"measured Ce2Hf2O7 ratio (25/65 mK): {MEASURED_RATIO:.4f}")
        print(f"shortfall: a factor of {MEASURED_RATIO / best['ratio']:.2f}")
        if near:
            print(f"\nbest NEAR-certified ({near[0]['n_found']}/90): "
                  f"{near[0]['ratio']:.4f} at ({near[0]['jpm']:+.3f}, "
                  f"{near[0]['jpmpm']:+.3f}) -- shortfall "
                  f"{MEASURED_RATIO / near[0]['ratio']:.2f}x")
        # Both numbers are cubic-16.  The measured host coefficients are
        # c16 = 1.070 against c32 = 0.521, so FCC-32 pushes ring-scale
        # quantities roughly a factor two COLDER; the null has to be tested
        # against the host-corrected value, not the cubic-16 one.
        host = 0.521 / 1.070
        print(f"\nhost-corrected to FCC-32 (x{host:.3f}): "
              f"certified {best['ratio'] * host:.4f}"
              + (f", near-certified {near[0]['ratio'] * host:.4f}"
                 if near else "")
              + f"  vs measured {MEASURED_RATIO:.4f}")
        verdict = (
            "NULL: no certified nearest-neighbour XYZ point in the "
            "fit-constrained region reaches the measured two-peak ratio. The "
            "25 mK anomaly is not accounted for by dressed gauge dynamics of "
            "the NN model -- which is a rigorous exclusion, and answers the "
            "experiment's own open question about beyond-NN terms."
            if best["ratio"] < MEASURED_RATIO else
            "POSITIVE: at least one certified point reaches the measured "
            "ratio. Build the material case around it, and verify the entropy "
            "split before claiming it.")
    else:
        verdict = ("UNDECIDED: no certified point has a resolved two-peak "
                   "structure yet.")
    print("\nWP2 VERDICT:", verdict)

    OUTDIR.mkdir(parents=True, exist_ok=True)
    (OUTDIR / "wp2_ratios.json").write_text(json.dumps(
        {"measured_ratio": MEASURED_RATIO, "dip_fraction": DIP_FRACTION,
         "verdict": verdict, "points": records}, indent=2, default=float))
    print(f"wrote {OUTDIR / 'wp2_ratios.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
