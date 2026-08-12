#!/usr/bin/env python3
"""Structural checks of the full-65,536-basis Hamiltonian for the Jpmpm campaign.

Verifies, at a merge-line coupling: hermiticity and realness at theta=0; that
the bare ice block vanishes identically (both ring exchange and the winding
artifact are emergent, so there is nothing microscopic to delete -- the mask
acts on the folded map); and at Jpmpm=0 that the full-basis Hamiltonian's
Sz=0 block equals the fixed-Sz Hamiltonian with zero coupling from the ice
manifold to any Sz != 0 state (the structural core of Gate A).
"""

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "notes"))

import recompute_finite_size_artifact as geometry  # noqa: E402
from qsi_campaign.exact_band import (  # noqa: E402
    fixed_magnetization_basis,
    full_spin_basis,
    microscopic_character_hamiltonian,
)
from qsi_campaign.protocol import polarization_sector_labels  # noqa: E402


def main() -> None:
    cluster = geometry.build_cluster("cubic", (1, 1, 1))
    labels = polarization_sector_labels(cluster)
    unique, counts = np.unique(labels, return_counts=True)
    assert len(unique) == 25 and counts.sum() == 90 and counts.max() == 12, \
        f"unexpected transport sectors: {len(unique)}, dims {counts}"
    print(f"transport sectors: {len(unique)}, dims min/max "
          f"{counts.min()}/{counts.max()}", flush=True)

    full = full_spin_basis(cluster.n_sites)
    index = {int(s): i for i, s in enumerate(full)}
    ice_rows = np.asarray([index[int(s)] for s in cluster.ice_states])

    started = time.perf_counter()
    coupled = microscopic_character_hamiltonian(
        cluster, full, -0.125, np.zeros(3), jpmpm=0.30)
    print(f"full-basis H(-0.125, pp=0.30): nnz={coupled.nnz} "
          f"({time.perf_counter() - started:.0f}s)", flush=True)
    assert np.abs(coupled.data.imag).max() < 1e-14, "H not real at theta=0"
    assert abs(coupled - coupled.conj().T).max() < 1e-12, "H not Hermitian"
    php = np.abs(coupled[ice_rows][:, ice_rows].toarray()).max()
    assert php == 0.0, f"bare ice block must vanish exactly, got {php:.3e}"
    print("hermitian, real, PHP == 0 exactly", flush=True)

    plain = microscopic_character_hamiltonian(
        cluster, full, -0.125, np.zeros(3), jpmpm=0.0)
    fixed = fixed_magnetization_basis(cluster.n_sites, cluster.n_sites // 2)
    reference = microscopic_character_hamiltonian(
        cluster, fixed, -0.125, np.zeros(3))
    rows = np.asarray([index[int(s)] for s in fixed])
    deviation = abs(plain[rows][:, rows] - reference).max()
    assert deviation < 1e-14, f"Sz=0 block deviates: {deviation:.3e}"
    outside = np.setdiff1d(np.arange(len(full)), rows)
    leak = abs(plain[outside][:, ice_rows]).max() if len(outside) else 0.0
    assert leak == 0.0, f"pp=0 ice couples to Sz != 0: {leak:.3e}"
    print("pp=0: Sz=0 block == fixed-Sz H, ice -> Sz!=0 coupling == 0",
          flush=True)
    print("FULL-BASIS STRUCTURE OK", flush=True)


if __name__ == "__main__":
    main()
