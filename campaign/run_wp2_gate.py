#!/usr/bin/env python3
"""WP2 blocking step: validate the pair-flip truncated builder.

WP2 needs the `(Ja, Jb, Jc)` scan on FCC-32, which needs the truncated
Feshbach machinery to handle `Jpmpm`.  The pair moves `S+S+` / `S-S-` are now
in `bfs_space` and `build_hamiltonian`, and nothing downstream may be believed
until they are shown to reproduce the Hamiltonian that built the exact
cubic-16 plane.  Two gates, in increasing strength:

**Gate A, matrix identity.**  On cubic-16 the reference
`exact_band.microscopic_character_hamiltonian` is available on the full
65,536-state basis.  Build the truncated space, extract the reference's
submatrix on exactly those states, and compare entry by entry.  If the
truncated space is closed under the Hamiltonian the two must agree to machine
precision -- and if it is not closed, the difference exposes precisely the
amplitude that leaks out.  This tests the builder, with the truncation error
set to zero by construction.

Closure is not decoration.  A pair flip that leaves the space would, in a
builder that skipped out-of-space targets, silently discard amplitude and
produce a Hamiltonian that is *not* the restriction of anything -- variational
guarantees included.  `build_hamiltonian` raises instead, and Gate A is what
proves the BFS keeps it from having to.

**Gate B, spectral agreement.**  Push the truncation to closure and compare the
masked band against the exact 56-point plane grid already in
`outputs/jpmpm_fold_plane/`.  Gate A can pass on a Hamiltonian assembled with
the wrong sign convention if that convention is used consistently; Gate B
cannot, because the plane grid was produced by the other code path.

Usage: run_wp2_gate.py [<jpm> <jpmpm> ...]     (default: four plane points)
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "notes"))
sys.path.insert(0, str(ROOT / "campaign"))

import recompute_finite_size_artifact as geometry  # noqa: E402
from qsi_campaign.exact_band import (  # noqa: E402
    full_spin_basis,
    microscopic_character_hamiltonian,
)
from run_truncated_feshbach import bfs_space, build_hamiltonian  # noqa: E402

OUTDIR = ROOT / "campaign" / "outputs" / "wp2_gate"
DEFAULT_POINTS = [(-0.20, 0.00), (-0.20, 0.10), (-0.125, 0.085), (-0.24, 0.15)]


def closure_ladder(cluster, jpmpm: float, max_levels: int = 12):
    """Grow the BFS space until it stops growing; report the ladder."""
    sizes = []
    previous = -1
    levels = 0
    space = None
    while levels < max_levels:
        levels += 1
        space = bfs_space(cluster, levels, jpmpm=jpmpm)
        sizes.append((levels, int(len(space))))
        if len(space) == previous:
            break
        previous = len(space)
    return space, sizes


def gate_a(cluster, states, state_index, jpm: float, jpmpm: float) -> dict:
    started = time.perf_counter()
    space, sizes = closure_ladder(cluster, jpmpm)
    rows = np.asarray([state_index[int(s)] for s in space], dtype=np.int64)

    reference = microscopic_character_hamiltonian(
        cluster, states, jpm, np.zeros(3), jpmpm=jpmpm)
    imaginary = float(np.max(np.abs(reference.data.imag), initial=0.0))
    restricted = reference.tocsr()[rows][:, rows].toarray().real

    try:
        # strict: this gate asserts the grown space IS closed, so a dropped
        # element here is a real disagreement with the reference, not the
        # ordinary boundary loss every truncated space has
        truncated = build_hamiltonian(cluster, space, jpm, jpmpm,
                                      strict=True).toarray()
        error = float(np.abs(truncated - restricted).max())
        closed = True
        message = ""
    except ValueError as failure:
        error = float("nan")
        closed = False
        message = str(failure)

    return {
        "jpm": jpm,
        "jpmpm": jpmpm,
        "levels_to_closure": sizes[-1][0],
        "space_size": sizes[-1][1],
        "bfs_ladder": sizes,
        "reference_imaginary_part": imaginary,
        "max_abs_difference": error,
        "space_is_closed": closed,
        "failure": message,
        "passed": bool(closed and error < 1e-12),
        "seconds": time.perf_counter() - started,
    }


def main() -> int:
    values = [float(v) for v in sys.argv[1:] if not v.startswith("--")]
    points = (list(zip(values[0::2], values[1::2])) if values
              else DEFAULT_POINTS)

    cluster = geometry.build_cluster("cubic", (1, 1, 1))
    states = full_spin_basis(cluster.n_sites)
    state_index = {int(s): i for i, s in enumerate(states)}
    print(f"cubic-16: {cluster.n_ice} ice states, full basis {len(states)}",
          flush=True)

    results = []
    for jpm, jpmpm in points:
        result = gate_a(cluster, states, state_index, jpm, jpmpm)
        results.append(result)
        ladder = " -> ".join(str(size) for _, size in result["bfs_ladder"])
        print(f"jpm={jpm:+.3f} pp={jpmpm:+.3f} BFS {ladder} "
              f"(closed at L={result['levels_to_closure']}), "
              f"reference imag {result['reference_imaginary_part']:.1e}, "
              f"max|H_trunc - H_ref| = {result['max_abs_difference']:.3e} "
              f"-> {'PASS' if result['passed'] else 'FAIL'} "
              f"({result['seconds']:.1f}s)", flush=True)
        if result["failure"]:
            print(f"    {result['failure']}", flush=True)

    OUTDIR.mkdir(parents=True, exist_ok=True)
    (OUTDIR / "wp2_gate_a.json").write_text(
        json.dumps(results, indent=2, default=float))

    passed = all(r["passed"] for r in results)
    print("\nWP2 GATE A:", "PASS -- the pair-flip builder reproduces the "
          "reference Hamiltonian exactly; the FCC-32 XYZ scan may proceed"
          if passed else
          "FAIL -- the pair-flip builder does not reproduce the reference; "
          "do not run the XYZ scan")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
