#!/usr/bin/env python3
"""Measure how isolated the ice-descended band stays across the source grid.

At the sign-free benchmark the 90-dimensional ice-descended band of cubic-16
sits 0.63 J_zz below the rest of the spectrum, so the spectral projector
P_theta of Eq. (spectral) is well defined at every source point.  At the
Ce2Hf2O7 couplings the zero-source gap is already only 0.023 J_zz, and the
M=3 band extraction fails outright.  This script says why, in numbers: for
each source point it reports the gap between the 90th and 91st levels of
H(theta) and the smallest eigenvalue of Pice P_theta Pice, whose vanishing is
the documented breakdown condition of the construction.

Run cost is one sparse solve of ~120 low levels per source point.
"""

from __future__ import annotations

import json
import sys
import time
from itertools import product
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "notes"))

import recompute_finite_size_artifact as geometry  # noqa: E402
from qsi_campaign.exact_band import (  # noqa: E402
    _lowest_eigenpairs,
    full_spin_basis,
    microscopic_character_hamiltonian,
)
from qsi_campaign.protocol import (  # noqa: E402
    band_operator_from_projected_vectors,
)

JPM = -0.15555742
JPMPM = 0.11532091
BENCHMARK = (0.046, 0.0)
EXTRA_LEVELS = 40
OUTPUT = ROOT / "campaign" / "outputs" / "ce2hf2o7_band_isolation.json"
PRODUCTION_REPORT = {
    "ce2hf2o7_s2": "nonperturbative_cubic16_m0p155557_ppp0p115321.json",
    "benchmark_jpm_0p046": "nonperturbative_cubic16_p0p046000.json",
}


def scan(cluster, states, ice_indices, jpm, jpmpm, grid_size):
    """Band gap and ice-overlap conditioning at every point of an M^3 grid."""
    band_dimension = cluster.n_ice
    records = []
    for index in product(range(grid_size), repeat=3):
        theta = 2.0 * np.pi * np.asarray(index, dtype=float) / grid_size
        hamiltonian = microscopic_character_hamiltonian(
            cluster, states, jpm, theta, jpmpm=jpmpm
        )
        started = time.perf_counter()
        # _lowest_eigenpairs is the production solver: it drops to the real
        # symmetric path wherever the source phases cancel, because ARPACK's
        # complex Hermitian path does not converge on the full 2^N basis here.
        values, vectors = _lowest_eigenpairs(
            hamiltonian, band_dimension + EXTRA_LEVELS, 1.0e-10
        )

        gap = float(values[band_dimension] - values[band_dimension - 1])
        # The diagnostic is exactly the one the production path reports, so the
        # two must agree wherever both are computed.
        _, diagnostics = band_operator_from_projected_vectors(
            values[:band_dimension],
            np.asarray(vectors[:, :band_dimension])[ice_indices, :],
        )
        records.append({
            "index": list(index),
            "band_gap": gap,
            "band_width": float(values[band_dimension - 1] - values[0]),
            "ice_overlap_min": diagnostics["model_overlap_min"],
            "pullback_unitarity_error": diagnostics["pullback_unitarity_error"],
            "seconds": time.perf_counter() - started,
        })
        print(
            f"M={grid_size} index={index} gap={gap:+.6f} "
            f"overlap_min={diagnostics['model_overlap_min']:.3e} "
            f"({records[-1]['seconds']:.1f}s)",
            flush=True,
        )
    return records


def main() -> None:
    cluster = geometry.build_cluster("cubic", (1, 1, 1))
    states = full_spin_basis(cluster.n_sites)
    state_index = {int(state): position for position, state in enumerate(states)}
    ice_indices = np.asarray(
        [state_index[int(state)] for state in cluster.ice_states], dtype=np.int64
    )
    print(f"cubic-16 full basis {len(states)}, band dimension {cluster.n_ice}",
          flush=True)

    report = {}
    for name, (jpm, jpmpm) in (
        ("ce2hf2o7_s2", (JPM, JPMPM)),
        ("benchmark_jpm_0p046", BENCHMARK),
    ):
        print(f"--- {name}: jpm={jpm}, jpmpm={jpmpm} ---", flush=True)
        records = scan(cluster, states, ice_indices, jpm, jpmpm, 3)
        gaps = np.array([item["band_gap"] for item in records])
        overlaps = np.array([item["ice_overlap_min"] for item in records])
        report[name] = {
            "jpm": jpm,
            "jpmpm": jpmpm,
            "grid": 3,
            "minimum_band_gap": float(gaps.min()),
            "maximum_band_gap": float(gaps.max()),
            "n_points_with_closed_gap": int((gaps <= 0.0).sum()),
            "minimum_ice_overlap": float(overlaps.min()),
            "points": records,
        }
        print(
            f"{name}: min gap {gaps.min():+.6f}, min ice overlap "
            f"{overlaps.min():.3e}, {int((gaps <= 0).sum())} closed points",
            flush=True,
        )

        # The zero-source point is computed by the production run through a
        # different route (translation sectors rather than the unblocked
        # basis).  Where that report exists the two must agree, which is what
        # makes the rest of the scan trustworthy.
        production = ROOT / "campaign" / "outputs" / PRODUCTION_REPORT.get(name, "")
        if production.name and production.exists():
            expected = json.loads(production.read_text())["zero_source"]
            zero = records[0]
            deltas = {
                "gap": abs(zero["band_gap"] - expected["gap_above_band"]),
                "overlap": abs(
                    zero["ice_overlap_min"] - expected["model_overlap_min"]
                ),
            }
            report[name]["zero_source_agreement"] = deltas
            print(f"{name}: zero-source agreement vs production {deltas}", flush=True)
            if max(deltas.values()) > 1.0e-6:
                raise RuntimeError(
                    f"{name}: the scan disagrees with the production band at "
                    f"zero source: {deltas}"
                )

    OUTPUT.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
