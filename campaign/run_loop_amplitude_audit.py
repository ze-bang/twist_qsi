#!/usr/bin/env python3
"""Audit loop amplitudes of the produced cubic-16 band operators.

Reads the stored nonperturbative run and measures, directly in the
zero-spinon reference basis, the matrix element that each enumerated loop
carries in the periodic band h(0) and in each character-projected band h_M.
No perturbation theory enters: the operators are polar pullbacks of exact
microscopic eigenspaces.  Backs Table II of the Supplemental Material.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "notes"))

import recompute_finite_size_artifact as geometry  # noqa: E402


def loop_supports(loops, winding: bool) -> list[int]:
    """Spin-flip bit masks of the enumerated loops of the requested topology."""
    supports = set()
    for path, wind in loops:
        if bool(np.any(np.asarray(wind) != 0)) is not winding:
            continue
        mask = 0
        for site in path:
            mask ^= 1 << int(site)
        supports.add(mask)
    return sorted(supports)


def channel_pairs(supports, ice_states, ice_index) -> list[tuple[int, int]]:
    """Zero-spinon pairs (s, t) with t = s xor mu for some support mu."""
    return [
        (ice_index[int(state)], ice_index[int(state) ^ mask])
        for mask in supports
        for state in ice_states
        if (int(state) ^ mask) in ice_index and int(state) < (int(state) ^ mask)
    ]


def channel_amplitudes(stored, pairs) -> dict[str, dict[str, dict[str, float]]]:
    """Mean and max loop amplitude per operator, per channel."""
    # h(0) is the M=1 entry; the projected operators are read before the exact
    # p-block structure is imposed, so that the winding entries measure the
    # character average itself rather than the block projection.
    operators = {"h(0)": stored["M1_operator"]}
    for grid in (2, 3, 4):
        key = f"M{grid}_operators_by_character"
        if key in stored.files:
            operators[f"h_M={grid}"] = stored[key].mean(axis=0)

    amplitudes = {}
    for label, operator in operators.items():
        entry = {}
        for name, channel in pairs.items():
            values = np.abs(np.array([operator[i, j] for i, j in channel]))
            entry[name] = {"mean": float(values.mean()), "max": float(values.max())}
        amplitudes[label] = entry
    return amplitudes


def main() -> None:
    jpm = 0.046
    run = ROOT / "campaign/outputs/nonperturbative_cubic16_p0p046000.npz"
    stored = np.load(run)
    # Optional second coupling: enables the power-law exponent of each channel.
    reference_jpm = 0.030
    reference_run = ROOT / "campaign/outputs/nonperturbative_cubic16_p0p030000.npz"

    cluster = geometry.build_cluster("cubic", (1, 1, 1))
    ice_states = np.asarray(cluster.ice_states)
    ice_index = {int(state): i for i, state in enumerate(ice_states)}

    channels = {
        "hexagon_contractible": loop_supports(cluster.hexes, winding=False),
        "hexagon_wrapping": loop_supports(cluster.hexes, winding=True),
        "four_loop_winding": loop_supports(cluster.loops4, winding=True),
    }
    pairs = {
        name: channel_pairs(supports, ice_states, ice_index)
        for name, supports in channels.items()
    }

    report = {
        "coupling": jpm,
        "g6": 12.0 * abs(jpm) ** 3,
        "g4": 4.0 * jpm**2,
        "n_zero_spinon": int(cluster.n_ice),
        "channel_supports": {name: len(s) for name, s in channels.items()},
        "amplitudes": channel_amplitudes(stored, pairs),
    }

    # Effective exponent nu of each channel, A ~ |Jpm|^nu.  This separates a
    # surviving process from the rounding image of a cancelled one, and dates
    # each residual to a perturbative order without assuming one.
    if reference_run.exists():
        reference = channel_amplitudes(np.load(reference_run), pairs)
        ratio = reference_jpm / jpm
        exponents = {}
        for label, entry in report["amplitudes"].items():
            if label not in reference:
                continue
            exponents[label] = {
                name: float(
                    np.log(reference[label][name]["mean"] / values["mean"])
                    / np.log(ratio)
                )
                for name, values in entry.items()
                if values["mean"] > 0.0 and reference[label][name]["mean"] > 0.0
            }
        report["reference_coupling"] = reference_jpm
        report["exponents"] = exponents

    bare = report["amplitudes"]["h(0)"]
    report["bare_four_loop_over_hexagon"] = (
        bare["four_loop_winding"]["mean"] / bare["hexagon_contractible"]["mean"]
    )

    destination = ROOT / "campaign/outputs/loop_amplitude_audit_cubic16.json"
    destination.write_text(json.dumps(report, indent=2) + "\n")

    print(f"zero-spinon dimension {report['n_zero_spinon']}, "
          f"g6={report['g6']:.4e}, g4={report['g4']:.4e}")
    for label, entry in report["amplitudes"].items():
        print(
            f"{label:8s} "
            f"hex_c={entry['hexagon_contractible']['mean']:.4e} "
            f"hex_w={entry['hexagon_wrapping']['max']:.2e} "
            f"loop4={entry['four_loop_winding']['max']:.2e}"
        )
    print(f"bare four-loop / hexagon = {report['bare_four_loop_over_hexagon']:.3f}")
    for label, entry in report.get("exponents", {}).items():
        print(f"{label:8s} exponents " + " ".join(
            f"{name}={value:.2f}" for name, value in entry.items()
        ))
    print(f"wrote {destination}")


if __name__ == "__main__":
    main()
