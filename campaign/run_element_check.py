#!/usr/bin/env python3
"""Ice-band matrix elements: naive periodic ED vs transport projection vs PT.

At a deeply perturbative coupling the third-order effective Hamiltonian on
the ice manifold has exactly one kind of off-diagonal element, the hexagon
flip at -12 Jpm^3/Jzz^2.  This script builds the exact periodic band h(0)
and the winding-free band h_wf (M=2 gauge corners + polarization mask) with
dense per-sector diagonalization -- no iterative solver anywhere -- and
tabulates both on the hexagon pairs, on the winding four-cycle pairs (the
weight-4 ice pairs), and on everything else.
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
from qsi_campaign.exact_band import (  # noqa: E402
    basis_character_phases,
    fixed_magnetization_basis,
    microscopic_character_hamiltonian,
    translation_sector_bases,
    cubic16_translation_permutations,
    uniform_character_grid,
)
from qsi_campaign.protocol import (  # noqa: E402
    character_project,
    polarization_sector_labels,
    sector_project,
)

OUTPUT = ROOT / "campaign" / "outputs"


def exact_band(cluster, states, jpm):
    """Lowest-90 eigenpairs by dense diagonalization of the four sectors."""
    hamiltonian = microscopic_character_hamiltonian(
        cluster, states, jpm, np.zeros(3))
    sectors = translation_sector_bases(
        states, cubic16_translation_permutations(cluster))
    values, vectors = [], []
    for basis in sectors.values():
        block = (basis.conj().T @ (hamiltonian @ basis)).toarray()
        eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (block + block.conj().T))
        values.append(eigenvalues)
        vectors.append(np.asarray(basis @ eigenvectors))
    values = np.concatenate(values)
    vectors = np.concatenate(vectors, axis=1)
    order = np.argsort(values)[: cluster.n_ice]
    return values[order], vectors[:, order]


def loewdin(psi, ice_indices, energies):
    overlap = psi[ice_indices, :]
    gram = overlap @ overlap.conj().T
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    if eigenvalues.min() < 1e-10:
        raise RuntimeError("ice Gram near singular")
    inv_sqrt = eigenvectors @ np.diag(eigenvalues**-0.5) @ eigenvectors.conj().T
    frame = overlap.conj().T @ inv_sqrt
    return frame.conj().T * energies @ frame  # W^dag H W in the ice frame


def main() -> None:
    cluster = geometry.build_cluster("cubic", (1, 1, 1))
    states = fixed_magnetization_basis(cluster.n_sites, cluster.n_sites // 2)
    index = {int(s): i for i, s in enumerate(states)}
    ice_indices = np.asarray([index[int(s)] for s in cluster.ice_states])
    ice_bits = [int(s) for s in cluster.ice_states]
    labels = polarization_sector_labels(cluster)
    d = len(ice_bits)

    # pair classification from the validated PT coefficients: hexagon pairs
    # carry the third-order element, weight-4 pairs are the winding cycles.
    e3 = np.load(OUTPUT / "ice_pt_coefficients.npz")["E3"]
    hexagon = np.abs(e3 - np.diag(np.diag(e3))) > 1e-9
    weight = np.array([[bin(a ^ b).count("1") for b in ice_bits] for a in ice_bits])
    winding4 = weight == 4
    other = (~hexagon) & (~winding4) & (weight > 0)

    report = {}
    for jpm in (-0.005, 0.005):
        energies, psi = exact_band(cluster, states, jpm)
        h0 = loewdin(psi, ice_indices, energies)
        ice_states = np.asarray(cluster.ice_states, dtype=np.uint64)
        corners = []
        for theta in uniform_character_grid(2):
            ph = basis_character_phases(cluster, ice_states, np.asarray(theta))
            corners.append(ph[:, None] * h0 * ph.conj()[None, :])
        h_wf = sector_project(character_project(np.asarray(corners)), labels)

        g6 = 12.0 * abs(jpm) ** 3
        g4 = 4.0 * jpm * jpm
        entry = {
            "g6": g6,
            "g4": g4,
            "h0_hexagon_mean": float(np.abs(h0[hexagon]).mean()),
            "h0_winding4_mean": float(np.abs(h0[winding4]).mean()),
            "hwf_hexagon_mean": float(np.abs(h_wf[hexagon]).mean()),
            "hwf_hexagon_spread": float(np.abs(h_wf[hexagon]).std()),
            "hwf_winding4_max": float(np.abs(h_wf[winding4]).max()),
            "hwf_other_max": float(np.abs(h_wf[other]).max()),
        }
        entry["h0_winding4_over_g4"] = entry["h0_winding4_mean"] / g4
        entry["hwf_hexagon_over_g6"] = entry["hwf_hexagon_mean"] / g6
        entry["h0_winding4_over_hex"] = (
            entry["h0_winding4_mean"] / entry["h0_hexagon_mean"])
        report[f"{jpm:+.3f}"] = entry
        print(json.dumps({f"{jpm:+.3f}": entry}, indent=1), flush=True)

    (OUTPUT / "element_check.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {OUTPUT / 'element_check.json'}")


if __name__ == "__main__":
    main()
