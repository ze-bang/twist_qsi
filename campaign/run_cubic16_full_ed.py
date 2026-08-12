#!/usr/bin/env python3
"""Exact full-Hilbert cubic-16 spectrum at one XYZ coupling point.

Equation (replacement) needs the microscopic complement of the ice-descended
band, so the winding-free heat capacity at a material parameter set requires
the whole 2^16 spectrum, not just the band.  J_pm_pm != 0 breaks S^z
conservation, so the basis is the full spin basis; the four FCC translations
of the cubic cell still commute with H(0) and block it into four sectors of
about 16384 states each, which dense LAPACK handles directly.

Defaults are the Ce2Hf2O7 "s2" couplings; ``--jpm``/``--jpmpm`` drive the
parameter scan that looks for a set whose nonperturbative low-temperature
structure, rather than its perturbative ring-exchange scale, matches the
measured anomalies.  Results are cached per coupling point.

The spectrum is written in the same convention as
``microscopic_character_hamiltonian``, which is what
``full_hilbert_counterterm_spectrum`` assumes.  Its ``match_trace`` step
re-references the band to the lowest states of this spectrum, so a constant
energy offset in the Ising term is immaterial.
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
from qsi_campaign.exact_band import (  # noqa: E402
    cubic16_translation_permutations,
    full_spin_basis,
    microscopic_character_hamiltonian,
    translation_sector_bases,
)

JPM = -0.15555742
JPMPM = 0.11532091


def coupling_tag(value: float) -> str:
    """Match the tag convention of run_nonperturbative.py."""
    return f"{value:+.6f}".replace("+", "p").replace("-", "m").replace(".", "p")


def cache_path(jpm: float, jpmpm: float) -> Path:
    tag = coupling_tag(jpm)
    if jpmpm != 0.0:
        tag += f"_pp{coupling_tag(jpmpm)}"
    return ROOT / "campaign" / "cache" / f"full_ed_cubic16_{tag}.npz"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jpm", type=float, default=JPM)
    parser.add_argument("--jpmpm", type=float, default=JPMPM)
    args = parser.parse_args()
    jpm, jpmpm = args.jpm, args.jpmpm
    CACHE = cache_path(jpm, jpmpm)
    if CACHE.exists():
        print(f"{CACHE} already exists; nothing to do")
        return
    started = time.perf_counter()
    cluster = geometry.build_cluster("cubic", (1, 1, 1))
    states = full_spin_basis(cluster.n_sites)
    print(f"cubic-16 full basis: {len(states)} states", flush=True)

    hamiltonian = microscopic_character_hamiltonian(
        cluster, states, jpm, np.zeros(3), jpmpm=jpmpm
    )
    sectors = translation_sector_bases(
        states, cubic16_translation_permutations(cluster)
    )

    eigenvalues = []
    for name, basis in sectors.items():
        block = (basis.conj().T @ hamiltonian @ basis).toarray()
        hermiticity = float(np.abs(block - block.conj().T).max())
        if hermiticity > 1.0e-10:
            raise RuntimeError(f"sector {name} is not Hermitian: {hermiticity:.3e}")
        # The Klein characters are all +-1 and the zero-source Hamiltonian is
        # real, so every block is real symmetric.  Handing LAPACK the complex
        # array anyway costs about a factor of four.
        imaginary = float(np.abs(block.imag).max())
        if imaginary < 1.0e-12:
            block = np.ascontiguousarray(block.real)
        else:
            raise RuntimeError(
                f"sector {name} carries an imaginary part {imaginary:.3e}; "
                "the real-symmetric shortcut does not apply"
            )
        block_started = time.perf_counter()
        values = np.linalg.eigvalsh(block)
        eigenvalues.append(values)
        print(
            f"sector {name}: dim={block.shape[0]} "
            f"{time.perf_counter() - block_started:.1f}s "
            f"E0={values[0]:.10f}",
            flush=True,
        )

    spectrum = np.sort(np.concatenate(eigenvalues))
    if len(spectrum) != len(states):
        raise RuntimeError(
            f"sector dimensions sum to {len(spectrum)}, expected {len(states)}"
        )

    # Independent check on the low edge: the sparse solver on the unblocked
    # Hamiltonian must reproduce the lowest eigenvalues of the assembled
    # spectrum, which would fail if a translation sector were mislabelled.
    from scipy.sparse.linalg import eigsh

    reference = np.sort(eigsh(hamiltonian, k=8, which="SA", tol=0)[0])
    low_edge_error = float(np.abs(reference - spectrum[:8]).max())
    print(f"low-edge cross-check vs unblocked eigsh: {low_edge_error:.3e}", flush=True)
    if low_edge_error > 1.0e-8:
        raise RuntimeError("blocked and unblocked low edges disagree")

    elapsed = time.perf_counter() - started
    np.savez_compressed(
        CACHE,
        E_full=spectrum,
        jpm=float(jpm),
        jpmpm=float(jpmpm),
        n_sites=int(cluster.n_sites),
        n_eigenvalues=int(len(spectrum)),
        low_edge_cross_check=low_edge_error,
        elapsed_s=float(elapsed),
    )
    print(json.dumps({
        "cache": str(CACHE),
        "n_eigenvalues": len(spectrum),
        "ground_state": float(spectrum[0]),
        "band_gap_above_ice_block": float(
            spectrum[cluster.n_ice] - spectrum[cluster.n_ice - 1]
        ),
        "elapsed_s": elapsed,
    }, indent=2))


if __name__ == "__main__":
    main()
