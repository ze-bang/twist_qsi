#!/usr/bin/env python3
"""Naive periodic FCC-32 Szz(q,w) on the same space as the masked response.

The calculation keeps all eight cluster momenta and the four pyrochlore
sublattice probes.  Its output is directly comparable to run_wp3_dssf.py at
the same virtual depth, coupling, and requested number of eigenstates.

Usage: run_fcc32_dssf_periodic.py <jpm> [--depth=1] [--nstates=120]
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.sparse.linalg import eigsh

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "notes"))
sys.path.insert(0, str(ROOT / "campaign"))

import recompute_finite_size_artifact as geometry  # noqa: E402
from run_truncated_feshbach import bfs_space, build_hamiltonian  # noqa: E402
from run_wp0b_gamma_rule import allowed_momenta  # noqa: E402
from run_wp3_dssf import sublattice_sources  # noqa: E402

OUTPUT = ROOT / "campaign" / "outputs" / "wp3_dssf"
DEGENERACY_TOL = 1.0e-9


def main() -> int:
    coupling = None
    depth = 1
    nstates = 120
    for token in sys.argv[1:]:
        if token.startswith("--depth="):
            depth = int(token.split("=", 1)[1])
        elif token.startswith("--nstates="):
            nstates = int(token.split("=", 1)[1])
        else:
            coupling = float(token)
    if coupling is None:
        raise SystemExit(
            "usage: run_fcc32_dssf_periodic.py <jpm> "
            "[--depth=1] [--nstates=120]")

    started = time.perf_counter()
    cluster = geometry.build_cluster("fcc", (2, 2, 2))
    space = bfs_space(cluster, depth)
    hamiltonian = build_hamiltonian(cluster, space, coupling).tocsr()
    k = min(nstates, hamiltonian.shape[0] - 2)
    print(
        f"periodic FCC-32 d={depth} jpm={coupling:+.3f}: "
        f"space={len(space):,}, states={k}", flush=True)
    energies, vectors = eigsh(
        hamiltonian, k=k, which="SA", tol=1.0e-10)
    order = np.argsort(energies)
    energies = energies[order]
    vectors = vectors[:, order]
    ground_rows = np.flatnonzero(energies - energies[0] < DEGENERACY_TOL)
    momenta = allowed_momenta(cluster)

    results = []
    for index, momentum in enumerate(momenta):
        sources = sublattice_sources(cluster, space, momentum)
        weights = np.zeros(k)
        for ground in ground_rows:
            for mu in range(4):
                source = sources[mu] * vectors[:, ground]
                overlap = vectors.conj().T @ source
                weights += np.abs(overlap) ** 2 / int(cluster.n_sites)
        weights /= len(ground_rows)
        excitation = energies - energies[0]
        weights[excitation <= DEGENERACY_TOL] = 0.0
        results.append({
            "momentum": [float(value) for value in momentum],
            "is_gamma": bool(np.allclose(momentum, 0.0)),
            "excitation": [float(value) for value in excitation],
            "weight": [float(value) for value in weights],
            "total_inelastic_weight": float(weights.sum()),
        })
        print(
            f"  q{index}: {np.asarray(momentum)}  "
            f"weight={weights.sum():.8g}", flush=True)

    record = {
        "cluster": "fcc32",
        "method": "periodic",
        "virtual_depth": depth,
        "jpm": coupling,
        "space": int(len(space)),
        "nstates_requested": int(nstates),
        "nstates_computed": int(k),
        "ground_energy": float(energies[0]),
        "ground_degeneracy_in_window": int(len(ground_rows)),
        "momenta": results,
        "runtime": time.perf_counter() - started,
    }
    tag = f"{coupling:+.3f}".replace(".", "p")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / f"dssf_periodic_fcc32_L{depth}_{tag}.json"
    path.write_text(json.dumps(record, indent=1, default=float))
    print(f"wrote {path} ({record['runtime']:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
