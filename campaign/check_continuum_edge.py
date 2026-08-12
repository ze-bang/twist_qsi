#!/usr/bin/env python3
"""Why the full basis loses roots that the even-parity basis keeps.

At two points near Jpm = 0 the parity run finds 25 of 90 masked roots and
the full-basis run finds none.  Parity is supposed to be an exact discard,
so one of them is wrong.

Hypothesis: ``solve_roots`` takes the continuum edge to be ``poles.min()``
over ALL poles of QHQ, and on the full basis that minimum can be an
odd-parity state.  Odd states have exactly zero coupling to the ice
manifold -- H cannot connect them to P at any order -- so they contribute
nothing to the self-energy and an ice-derived level cannot decay into them.
Using them to set the edge rejects roots against a continuum the band can
never reach.

The test prints, for both bases: the lowest pole, the coupling weight
carried by the lowest poles, and the lowest pole whose coupling is
non-negligible.  If the hypothesis holds, the full basis will show a
lowest pole with vanishing coupling and a first *coupled* pole that
matches the even-parity edge.

Usage: check_continuum_edge.py [--gpu]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "notes"))
sys.path.insert(0, str(ROOT / "campaign"))

import recompute_finite_size_artifact as geometry  # noqa: E402
from qsi_campaign.exact_band import (  # noqa: E402
    cubic16_translation_permutations,
    full_spin_basis,
    microscopic_character_hamiltonian,
    translation_sector_bases,
)
from qsi_campaign.downfold import build_blocks_klein, solve_roots  # noqa: E402
from scan_xy_masked_flux import even_parity_basis  # noqa: E402

POINTS = [(0.50, -0.50), (-0.30, 0.10)]
COUPLING_TOL = 1e-8


def report(cluster, states, label, jx, jy, gpu):
    ice_sorted = np.sort(np.asarray(cluster.ice_states, dtype=np.uint64))
    ice_rows = np.searchsorted(states, ice_sorted)
    sector_bases = translation_sector_bases(
        states, cubic16_translation_permutations(cluster))
    jpm, jpmpm = -(jx + jy) / 4.0, (jx - jy) / 4.0
    hamiltonian = microscopic_character_hamiltonian(
        cluster, states, jpm, np.zeros(3), jpmpm=jpmpm)
    blocks = build_blocks_klein(cluster, states, ice_rows, hamiltonian,
                                sector_bases, gpu=gpu)

    order = np.argsort(blocks.poles)
    poles = blocks.poles[order]
    weight = np.linalg.norm(blocks.coupling[order, :], axis=1)
    coupled = np.where(weight > COUPLING_TOL)[0]
    roots = solve_roots(blocks, masked=True)

    print(f"  [{label}] basis {len(states)}, {len(poles)} poles")
    print(f"    lowest five poles      {np.array2string(poles[:5], precision=6)}")
    print(f"    their coupling norms   {np.array2string(weight[:5], precision=3)}")
    print(f"    poles with coupling    {coupled.size} of {poles.size}")
    if coupled.size:
        print(f"    lowest COUPLED pole    {poles[coupled[0]]:+.6f} "
              f"(index {coupled[0]})")
    print(f"    edge used by solver    {poles.min():+.6f}")
    print(f"    masked roots converged {int(roots.converged.sum())}/90")
    return poles, weight, int(roots.converged.sum())


def main() -> None:
    gpu = "--gpu" in sys.argv
    cluster = geometry.build_cluster("cubic", (1, 1, 1))
    bases = {
        "even": even_parity_basis(cluster.n_sites),
        "full": full_spin_basis(cluster.n_sites),
    }
    for jx, jy in POINTS:
        print(f"\n(Jx, Jy) = ({jx:+.2f}, {jy:+.2f})  "
              f"Jpm={-(jx + jy) / 4:+.4f}  Jpmpm={(jx - jy) / 4:+.4f}")
        results = {}
        for label, states in bases.items():
            results[label] = report(cluster, states, label, jx, jy, gpu)
        poles_even, _, _ = results["even"]
        poles_full, weight_full, _ = results["full"]
        coupled_full = np.where(weight_full > COUPLING_TOL)[0]
        if coupled_full.size:
            difference = abs(poles_full[coupled_full[0]] - poles_even.min())
            print(f"    VERDICT: lowest coupled pole (full) vs lowest pole "
                  f"(even) differ by {difference:.3e}")
            print("    -> hypothesis holds" if difference < 1e-8
                  else "    -> hypothesis FAILS, look further")


if __name__ == "__main__":
    main()
