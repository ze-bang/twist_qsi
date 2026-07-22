"""Cubic-16 pilot: the FCC-32 production path, run where a gold reference exists.

Exercises exactly the pieces the FCC-32 run needs and that have never been
tested together:

  * the sourced Hamiltonian written to a QED directory (theta sign-flipped),
  * translation-symmetry blocking through ed::make_sector_operators_tagged,
  * the ICE GATHER IN THE SYMMETRY-ADAPTED BASIS -- the previously unbuilt piece,
  * band extraction per momentum sector,
  * the polar pullback, assembled across sectors,

and compares h(theta) against the frozen cubic-16 reference operators.
"""

from __future__ import annotations

import itertools
import json
import math
import os
import sys
import tempfile
import time

import numpy as np
from scipy.sparse.linalg import LinearOperator, eigsh

TW = "/lustre09/project/6003507/zhouzb79/twist_qsi"
sys.path[:0] = [TW + "/notes", TW + "/src"]

import qed
import qed._core as core
from qed.workflow import _write_operator_directory
from recompute_finite_size_artifact import build_cluster
from qsi_campaign.protocol import (
    band_operator_from_projected_vectors,
    centered_relative_error,
    inverse_sqrt_hermitian,
)

JPM = 0.046


def translation_permutations(cluster):
    """The four FCC translations of the cubic cell, as site permutations."""
    shifts = (
        np.array((0.0, 0.0, 0.0)),
        np.array((0.0, 0.5, 0.5)),
        np.array((0.5, 0.0, 0.5)),
        np.array((0.5, 0.5, 0.0)),
    )
    key = lambda p: tuple(np.rint(4.0 * np.mod(p, 1.0)).astype(int) % 4)
    site_of = {key(p): i for i, p in enumerate(cluster.positions)}
    return [
        [site_of[key(p + shift)] for p in cluster.positions] for shift in shifts
    ]


def write_directory(root, cluster, theta, permutations):
    """Trans.dat / InterAll.dat + the Z_2 x Z_2 automorphism metadata."""
    auto = os.path.join(root, "automorphism_results")
    os.makedirs(auto, exist_ok=True)
    with open(os.path.join(auto, "max_clique.json"), "w") as handle:
        json.dump(permutations, handle)
    # Klein group: two commuting order-2 generators.
    with open(os.path.join(auto, "minimal_generators.json"), "w") as handle:
        json.dump(
            {"generators": [{"permutation": permutations[1], "order": 2},
                            {"permutation": permutations[2], "order": 2}]},
            handle,
        )
    sectors = []
    for sid, (c1, c2) in enumerate(itertools.product((0, 1), repeat=2)):
        sectors.append(
            {
                "sector_id": sid,
                "quantum_numbers": [c1, c2],
                "phase_factors": [
                    {"real": float((-1) ** c1), "imag": 0.0},
                    {"real": float((-1) ** c2), "imag": 0.0},
                ],
            }
        )
    with open(os.path.join(auto, "sector_metadata.json"), "w") as handle:
        json.dump({"sectors": sectors}, handle)

    # H0 = 0.5*n_tets*I + sum_<ij> Sz_i Sz_j   (each bond lies in exactly one tet).
    # The constant is NOT representable as a term and is added back below.
    operator = qed.Operator(cluster.n_sites, 0.5)
    for (i, j) in cluster.bonds:
        operator.add_two_body(core.OP_SZ, int(i), core.OP_SZ, int(j), 1.0)
    for (i, j), wrap in zip(cluster.bonds, cluster.bond_wrap):
        d = (cluster.positions[j] - np.asarray(wrap, float) @ cluster.Lvecs
             - cluster.positions[i])
        # Sign convention: verified empirically against the frozen reference.
        # The bit-complemented basis realises H(-theta), but the symmetrised
        # sector construction conjugates once more, so the two cancel and the
        # phase is used as-is here.  (Negating theta gives conj(h) to 1e-10.)
        phase = np.exp(2j * float(theta @ d))
        operator.add_two_body(core.OP_SPLUS, int(i), core.OP_SMINUS, int(j), -JPM * phase)
        operator.add_two_body(core.OP_SPLUS, int(j), core.OP_SMINUS, int(i),
                              -JPM * phase.conjugate())
    _write_operator_directory(operator, root)


def storage_index(M):
    visited, reps, selfc = set(), [], []
    for idx in itertools.product(range(M), repeat=3):
        if idx in visited:
            continue
        neg = tuple((-v) % M for v in idx)
        visited.add(idx)
        visited.add(neg)
        (selfc if idx == neg else reps).append(idx if idx == neg else (idx, neg))
    pos = {idx: k for k, idx in enumerate(selfc)}
    for k, (idx, neg) in enumerate(reps):
        pos[idx] = len(selfc) + 2 * k
        pos[neg] = len(selfc) + 2 * k + 1
    return pos


def run(theta, cluster, permutations, n_up, shift, complement=True, lowdin=True, extra=0):
    # QED's basis is bit-complemented relative to twist_qsi's up-mask, so the
    # internal label of ice state m is (2^n - 1) - m.  The ice set is closed
    # under complement, so passing the wrong one still resolves to valid indices
    # -- just paired with the wrong ice state.  Getting this wrong is silent.
    ice = np.array([int(s) for s in cluster.ice_states], dtype=np.uint64)
    internal = np.array(
        [((1 << cluster.n_sites) - 1 - int(s)) if complement else int(s)
         for s in cluster.ice_states],
        dtype=np.uint64,
    )
    n_band = len(ice)

    with tempfile.TemporaryDirectory() as root:
        write_directory(root, cluster, theta, permutations)
        t0 = time.perf_counter()
        sectors = core.sector_operators(root, cluster.n_sites, 0.5, n_up)
        build_seconds = time.perf_counter() - t0

    dims = [int(s.dimension) for s in sectors]
    print(f"    sector dims {dims} (sum {sum(dims)}), basis build {build_seconds:.2f}s")

    # ---- the ice gather, per sector ------------------------------------
    # X_k[idx, j] = <sector basis idx | ice_j>.  Each ice state lands on at most
    # one index, so the gather is one nonzero per (sector, ice state).
    energies, projected = [], []
    weight = np.zeros(n_band)          # completeness check, see below
    for sector in sectors:
        dim = int(sector.dimension)
        if dim == 0:
            continue
        indices, amplitudes = sector.project_states(internal)
        gather = np.zeros((dim, n_band), dtype=np.complex128)
        hit = indices >= 0
        gather[indices[hit], np.nonzero(hit)[0]] = amplitudes[hit]
        weight += np.abs(amplitudes) ** 2

        # ---- band in this sector ---------------------------------------
        want = min(dim - 2, n_band + extra)
        operator = LinearOperator(
            (dim, dim), dtype=np.complex128,
            matvec=lambda v, s=sector: s.apply(np.ascontiguousarray(v, dtype=np.complex128))
            + shift * v,
        )
        vals, vecs = eigsh(operator, k=want, which="SA", tol=1e-12)
        order = np.argsort(vals.real)
        vals, vecs = vals.real[order], vecs[:, order]
        # Loewdin within the sector.  Note this leaves h unchanged (the
        # downstream Gram normalisation absorbs it) but it IS what makes
        # model_overlap_min agree with the reference, so it stays.
        if lowdin:
            vecs = vecs @ inverse_sqrt_hermitian(vecs.conj().T @ vecs)
        energies.append(vals)
        projected.append(gather.conj().T @ vecs)

    # The sector bases together span the full space, so every ice state must
    # carry total weight 1 across the sectors.  This is convention-independent:
    # it fails loudly if the internal-label mapping or the orbit normalisation
    # is wrong, without needing the reference operator.
    print(f"    ice-gather completeness: max|sum_k |<k|ice>|^2 - 1| = "
          f"{np.abs(weight - 1.0).max():.3e}")

    # ---- select the band across sectors and pull back -------------------
    all_energies = np.concatenate(energies)
    all_projected = np.hstack(projected)
    keep = np.argsort(all_energies)[: n_band]
    gap = np.sort(all_energies)[n_band] - np.sort(all_energies)[n_band - 1]
    operator, diagnostics = band_operator_from_projected_vectors(
        all_energies[keep], all_projected[:, keep]
    )
    return operator, gap, diagnostics


def main():
    cluster = build_cluster("cubic", (1, 1, 1))
    permutations = translation_permutations(cluster)
    n_up = cluster.n_sites // 2
    shift = 0.5 * len(cluster.tets)
    reference = np.load(
        TW + "/campaign/outputs/nonperturbative_cubic16_p0p046000.npz"
    )
    print(f"cubic-16: {cluster.n_sites} sites, ice rank {len(cluster.ice_states)}, "
          f"n_up={n_up}, H0 constant {shift}")

    for M, index in ((3, (0, 1, 1)), (4, (1, 1, 2))):
        theta = 2.0 * np.pi * np.asarray(index, dtype=float) / M
        print(f"\n  M={M} theta index {index}")
        # The driver now writes operators_by_character in LEXICOGRAPHIC order
        # and records M{M}_character_indices alongside; index it that way.
        lex = list(itertools.product(range(M), repeat=3)).index(index)
        assert tuple(reference[f"M{M}_character_indices"][lex]) == index
        frozen = reference[f"M{M}_operators_by_character"][lex]
        for lowdin in (True,):
            label = "lowdin" if lowdin else "no-lowdin"
            t0 = time.perf_counter()
            operator, gap, diagnostics = run(theta, cluster, permutations, n_up,
                                             shift, lowdin=lowdin)
            print(f"    [{label}] solve {time.perf_counter()-t0:.1f}s  gap={gap:.6g}  "
                  f"overlap_min={diagnostics['model_overlap_min']:.6g}")
            print(f"    [{label}] vs h_frozen      : max={np.abs(operator - frozen).max():.3e}"
                  f"  centered={centered_relative_error(operator, frozen):.3e}")
            print(f"    [{label}] vs conj(h_frozen): max={np.abs(operator - frozen.conj()).max():.3e}"
                  f"  centered={centered_relative_error(operator, frozen.conj()):.3e}")
            # The decisive discriminator: if the SPECTRA differ, this is not a
            # basis/gauge issue at all but a genuine error in the pullback.
            sm = np.linalg.eigvalsh(operator)
            sf = np.linalg.eigvalsh(frozen)
            print(f"    [{label}] spec(h_mine) vs spec(h_frozen): max={np.abs(sm-sf).max():.3e}"
                  f"  |spec|max={np.abs(sf).max():.3e}")
            print(f"    [{label}] trace mine={np.trace(operator).real:.9g} "
                  f"frozen={np.trace(frozen).real:.9g}")
            np.save(f"campaign/pilot/h_mine_M{M}_{''.join(map(str,index))}.npy", operator)
            np.save(f"campaign/pilot/h_frozen_M{M}_{''.join(map(str,index))}.npy", frozen)


if __name__ == "__main__":
    main()
