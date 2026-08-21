#!/usr/bin/env python3
"""Momentum-resolved masked spectrum, and the per-sublattice channel weights.

Two measurements the S^zz panel cannot make.

**Is the spectrum at X degenerate with the spectrum at L?**  The structure
factor only shows what the probe couples to, so a mode can exist at a momentum
and be invisible there if the sublattice form factor kills it.  Reading a
dispersion off S^zz therefore cannot test a claimed X/L degeneracy.  This
resolves the eigenvalues themselves by momentum -- project every eigenvector
onto the eight cluster momenta with

    P_q = (1/N_t) sum_t chi_q(t)^* T_t,

`T_t` the lattice translations -- so the spectrum at each momentum is read off
directly, whether or not S^z couples to it.

**Which channel carries which line?**  At L the response splits ~90/10 between
two excitations, and the split is nearly independent of coupling (9.2% at
|Jpm|=0.046 against 9.8% at 0.200).  That is the signature of a geometric form
factor rather than a dynamical residue: the four sublattice channels fall into
different irreps of the little group and couple to different modes.  Resolving
`|A^mu|^2` per channel instead of summing settles it -- if the two lines live
in different channels the split is kinematic, not a multi-photon continuum.

Usage: run_momentum_resolved_spectrum.py <jpm> [--depth=1] [--nstates=200]
                                         [--periodic] [--jpmpm=0.0]
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
from qsi_campaign.protocol import polarization_sector_labels  # noqa: E402
from run_truncated_feshbach import bfs_space, build_hamiltonian  # noqa: E402
from run_wp0b_gamma_rule import allowed_momenta  # noqa: E402

OUTPUT = ROOT / "campaign" / "outputs" / "wp3_dssf"
DEGENERACY_TOL = 1.0e-9
LEVEL_TOL = 1.0e-7
CLUSTERS = {"cubic16": ("cubic", (1, 1, 1)), "fcc32": ("fcc", (2, 2, 2))}


def reduce_momentum(momentum):
    """Shortest vector equivalent to `momentum` modulo the fcc reciprocal set.

    The reciprocal lattice of the fcc lattice of tetrahedron centres (a = 1) is
    bcc: integer triples whose components share a parity.  `allowed_momenta`
    hands back whatever representative the enumeration produced, and on
    cubic-16 those are NOT the canonical ones -- (1,1,0) appears, which is
    (0,0,1) plus the reciprocal vector (1,1,-1).  Classifying the raw
    components therefore mislabels genuine X points as unknown, which is what
    the first cubic-16 run did.
    """
    momentum = np.asarray(momentum, dtype=float)
    best, best_norm = momentum, float(np.linalg.norm(momentum))
    for h in range(-2, 3):
        for k in range(-2, 3):
            for l in range(-2, 3):
                if not (h % 2 == k % 2 == l % 2):
                    continue
                candidate = momentum - np.array([h, k, l], dtype=float)
                norm = float(np.linalg.norm(candidate))
                if norm < best_norm - 1e-9:
                    best, best_norm = candidate, norm
    return best


def star(momentum) -> str:
    magnitudes = sorted(abs(float(v)) for v in reduce_momentum(momentum))
    if np.allclose(magnitudes, [0.0, 0.0, 0.0]):
        return "Gamma"
    if np.allclose(magnitudes, [0.0, 0.0, 1.0]):
        return "X"
    if np.allclose(magnitudes, [0.5, 0.5, 0.5]):
        return "L"
    if np.allclose(magnitudes, [0.0, 0.5, 0.5]):
        return "W/K"
    return "?" + str([round(v, 3) for v in magnitudes])


def translation_permutations(cluster):
    """Site permutation for each lattice translation the cluster admits.

    Found geometrically rather than by index arithmetic, because the two
    cluster shapes lay their sites out differently -- the fcc cluster stacks
    primitive cells, the cubic one stacks conventional cells each holding four
    fcc centres -- and the translation group is the same object in both cases:
    the fcc lattice of tetrahedron centres, reduced modulo the box.  A
    translation must map the site set onto itself preserving sublattice, which
    is checked rather than assumed.
    """
    positions = np.asarray(cluster.positions, dtype=float)
    lvecs = np.asarray(cluster.Lvecs, dtype=float)
    inverse = np.linalg.inv(lvecs)
    n_sites = int(cluster.n_sites)
    sublattice = np.arange(n_sites) % 4

    def locate(shifted):
        """Index of each shifted position, matched modulo the box."""
        fractional = shifted @ inverse
        fractional -= np.round(fractional - 0.5 + 1e-9)
        wrapped = fractional @ lvecs
        distance = np.linalg.norm(
            wrapped[:, None, :] - positions[None, :, :], axis=2)
        best = np.argmin(distance, axis=1)
        return best, float(distance[np.arange(len(best)), best].max())

    span = max(int(v) for v in cluster.shape) + 2
    seen, vectors, permutations = set(), [], []
    for i in range(-span, span + 1):
        for j in range(-span, span + 1):
            for k in range(-span, span + 1):
                shift = (i * geometry.FCC_A[0] + j * geometry.FCC_A[1]
                         + k * geometry.FCC_A[2])
                mapped, residual = locate(positions + shift)
                if residual > 1e-8 or len(set(mapped.tolist())) != n_sites:
                    continue
                if not np.array_equal(sublattice[mapped], sublattice):
                    continue
                key = tuple(mapped.tolist())
                if key in seen:
                    continue
                seen.add(key)
                vectors.append(shift)
                permutations.append(np.asarray(mapped, dtype=np.int64))
    expected = n_sites // 4
    if len(permutations) != expected:
        raise RuntimeError(
            f"found {len(permutations)} translations, expected {expected}")
    return np.asarray(vectors), permutations


def permute_states(states: np.ndarray, permutation) -> np.ndarray:
    """Apply a site permutation to a set of bitmask configurations."""
    out = np.zeros_like(states)
    for source, target in enumerate(permutation):
        bit = (states >> np.uint64(source)) & np.uint64(1)
        out |= bit << np.uint64(int(target))
    return out


def channel_weights(vectors, block_states, positions, momenta, n_sites):
    """`|<n|S^z_mu(q)|0>|^2` per state, per momentum, per sublattice channel.

    No momentum projection is needed: the ground state sits at Gamma and
    `S^z_mu(q)` raises momentum by exactly `q`, so the overlap is automatically
    zero unless `n` carries momentum `q`.  The selection is enforced by the
    operator, not imposed by hand, which is what makes a vanishing entry
    meaningful -- it says the probe cannot reach that level, rather than that
    we declined to look.
    """
    bits = ((block_states[:, None] >> np.arange(n_sites, dtype=np.uint64))
            & np.uint64(1))
    spins = bits.astype(float) - 0.5                    # (dim, n_sites)
    weighted = vectors[:, 0][:, None] * spins           # S^z_i |0>
    amplitudes = vectors.conj().T @ weighted            # (n_states, n_sites)
    sublattice = np.arange(n_sites) % 4
    out = np.zeros((vectors.shape[1], len(momenta), 4))
    for q_index, momentum in enumerate(momenta):
        phase = np.exp(-2j * np.pi * (positions @ momentum))
        for mu in range(4):
            column = amplitudes @ (phase * (sublattice == mu))
            out[:, q_index, mu] = np.abs(column) ** 2 / n_sites
    return out


def momentum_weights(vectors, block_states, permutations, phases):
    """||P_q v||^2 for every state v and every allowed momentum q."""
    order = np.argsort(block_states)
    ordered = block_states[order]
    n_translations = len(permutations)
    # T_t as an index map on the block
    maps = []
    for permutation in permutations:
        moved = permute_states(block_states, permutation)
        slot = np.searchsorted(ordered, moved)
        if not np.array_equal(ordered[slot], moved):
            raise RuntimeError("translation leaves the block: not closed")
        maps.append(order[slot])

    n_states = vectors.shape[1]
    weights = np.zeros((n_states, phases.shape[0]))
    translated = np.empty((n_translations, vectors.shape[0], n_states))
    for index, index_map in enumerate(maps):
        # (T_t v)[index_map[r]] = v[r]
        buffer = np.empty_like(vectors)
        buffer[index_map] = vectors
        translated[index] = buffer
    for q_index in range(phases.shape[0]):
        accumulator = np.tensordot(
            phases[q_index].conj(), translated, axes=(0, 0)) / n_translations
        weights[:, q_index] = np.sum(np.abs(accumulator) ** 2, axis=0)
    return weights


def main() -> int:
    coupling, depth, n_states, jpmpm = None, 1, 200, 0.0
    name = "fcc32"
    periodic = "--periodic" in sys.argv
    for token in sys.argv[1:]:
        if token.startswith("--depth="):
            depth = int(token.split("=", 1)[1])
        elif token.startswith("--nstates="):
            n_states = int(token.split("=", 1)[1])
        elif token.startswith("--jpmpm="):
            jpmpm = float(token.split("=", 1)[1])
        elif token.startswith("--cluster="):
            name = token.split("=", 1)[1]
        elif not token.startswith("--"):
            coupling = float(token)
    if coupling is None:
        raise SystemExit(__doc__)
    if name not in CLUSTERS:
        raise SystemExit(f"unknown cluster {name}; choose from {list(CLUSTERS)}")

    started = time.perf_counter()
    cluster = geometry.build_cluster(*CLUSTERS[name])
    space = bfs_space(cluster, depth, jpmpm=jpmpm)
    n_sites = int(cluster.n_sites)
    hamiltonian = build_hamiltonian(cluster, space, coupling, jpmpm).tocsr()
    method = "periodic" if periodic else "masked"
    print(f"{method} {name} jpm={coupling:+.3f}: space={len(space):,} "
          f"states={n_states}", flush=True)

    if periodic:
        sector_list = [0]
        sector_rows_map = {0: np.arange(len(space))}
    else:
        labels = polarization_sector_labels(cluster)
        ice_states = np.sort(np.asarray(cluster.ice_states, dtype=np.uint64))
        ice_rows = np.searchsorted(space, ice_states)
        non_ice = np.setdiff1d(np.arange(len(space)), ice_rows)
        grounds, sector_rows = {}, {}
        for sector in np.unique(labels):
            candidate = np.concatenate([ice_rows[labels == sector], non_ice])
            sector_rows[int(sector)] = candidate
            grounds[int(sector)] = float(eigsh(
                hamiltonian[np.ix_(candidate, candidate)], k=1, which="SA",
                tol=1e-10, return_eigenvectors=False)[0])
        lowest = min(grounds.values())
        ground_sectors = [s for s, e in grounds.items()
                          if e - lowest < DEGENERACY_TOL]
        print(f"  {len(ground_sectors)} degenerate ground sector(s): "
              f"{ground_sectors}", flush=True)
        # Taking just one is wrong when several are degenerate: the sectors are
        # related by the cubic point group, so a single sector carries only
        # part of each level's momentum content and the star degeneracies come
        # out broken.  On cubic-16 (six ground sectors) that split the three X
        # points 4+2; on FCC-32 there is one sector and the question does not
        # arise.  Pool the whole manifold.
        sector_list = ground_sectors
        sector_rows_map = sector_rows

    translations, permutations = translation_permutations(cluster)
    momenta = allowed_momenta(cluster)
    print(f"  {len(translations)} translations, {len(momenta)} momenta:",
          flush=True)
    for momentum in momenta:
        reduced = reduce_momentum(momentum)
        print(f"    {str([round(float(v), 3) for v in momentum]):>26s}"
              f" -> {str([round(float(v), 3) for v in reduced]):>26s}"
              f"  {star(momentum)}", flush=True)
    phases = np.exp(-2j * np.pi * (translations @ np.asarray(momenta).T)).T

    # Diagonalize every degenerate ground sector and pool.  Their spectra are
    # identical as sets -- the sectors are point-group images of one another --
    # but each labels the momenta differently, so only the union restores the
    # star degeneracies.
    positions = np.asarray(cluster.positions, dtype=float)
    all_values, all_weights, all_channels, completeness = [], [], [], 0.0
    for sector_index, sector in enumerate(sector_list):
        rows = sector_rows_map[sector]
        block = hamiltonian[np.ix_(rows, rows)]
        k = min(n_states, block.shape[0] - 2)
        values, vectors = eigsh(block, k=k, which="SA", tol=1e-10)
        order = np.argsort(values)
        values, vectors = values[order], vectors[:, order]
        piece = momentum_weights(vectors, space[rows], permutations, phases)
        channels = channel_weights(vectors, space[rows], positions,
                                   momenta, n_sites)
        completeness = max(
            completeness, float(np.abs(piece.sum(axis=1) - 1.0).max()))
        all_values.append(values)
        all_weights.append(piece)
        all_channels.append(channels)
        if len(sector_list) > 1:
            print(f"    sector {sector}: ground {values[0]:.9f}", flush=True)
    reference = min(float(v[0]) for v in all_values)
    values = np.concatenate(all_values)
    weights = np.concatenate(all_weights, axis=0) / len(sector_list)
    channel = np.concatenate(all_channels, axis=0) / len(sector_list)
    order = np.argsort(values)
    values, weights, channel = values[order], weights[order], channel[order]
    excitation = values - reference
    print(f"  window covers omega up to {excitation[-1]:.5f}", flush=True)
    print(f"  momentum projector completeness: max |sum_q - 1| = "
          f"{completeness:.2e}", flush=True)

    # group exactly degenerate levels and count states per momentum star
    levels = []
    start = 0
    for index in range(1, len(values) + 1):
        if index == len(values) or values[index] - values[start] > LEVEL_TOL:
            per_star: dict[str, float] = {}
            for q_index, momentum in enumerate(momenta):
                per_star[star(momentum)] = per_star.get(star(momentum), 0.0) \
                    + float(weights[start:index, q_index].sum())
            # S^zz reach of this level, resolved by sublattice channel.  Summed
            # over the momenta the level actually occupies, so a zero here says
            # the probe cannot reach the level at its own momentum.
            occupied = [q for q in range(len(momenta))
                        if weights[start:index, q].sum() > 1e-6]
            per_channel = [
                float(channel[start:index, occupied, mu].sum())
                for mu in range(4)]
            levels.append({
                "excitation": float(excitation[start]),
                "degeneracy": int(index - start),
                "per_star": {s: round(v, 4) for s, v in per_star.items()},
                "channel_weight": per_channel,
                "total_szz_weight": float(sum(per_channel)),
            })
            start = index

    print("\n  omega        deg   star            S^zz total    "
          "per channel mu=0..3")
    for level in levels[:20]:
        occupied = "  ".join(
            f"{s}:{v:5.2f}" for s, v in sorted(level["per_star"].items())
            if v > 1e-3)
        channels = " ".join(f"{v:9.2e}" for v in level["channel_weight"])
        print(f"  {level['excitation']:.6f} {level['degeneracy']:4d}   "
              f"{occupied:14s}  {level['total_szz_weight']:10.3e}  "
              f"{channels}")

    print("\n  (legacy view) omega        deg   states per star")
    for level in levels[:6]:
        breakdown = "  ".join(
            f"{s}:{v:5.2f}" for s, v in sorted(level["per_star"].items())
            if v > 1e-3)
        print(f"  {level['excitation']:.6f}  {level['degeneracy']:3d}   "
              f"{breakdown}")

    lowest_by_star: dict[str, float] = {}
    for level in levels:
        if level["excitation"] <= DEGENERACY_TOL:
            continue
        # NB: not `name` -- that is the cluster name, and shadowing it here
        # silently wrote every output file as momentum_spectrum_*_X_*.json
        for label, count in level["per_star"].items():
            if count > 0.5 and label not in lowest_by_star:
                lowest_by_star[label] = level["excitation"]
    print("\n  lowest excitation per star (probe-independent):")
    for label, value in sorted(lowest_by_star.items()):
        print(f"    {label:6s} {value:.6f}")
    if {"X", "L"} <= set(lowest_by_star):
        ratio = lowest_by_star["L"] / lowest_by_star["X"]
        print(f"    L/X ratio = {ratio:.4f}  "
              f"({'degenerate' if abs(ratio - 1) < 0.02 else 'NOT degenerate'})")

    record = {
        "cluster": name, "method": method, "jpm": coupling,
        "jpmpm": jpmpm, "virtual_depth": depth,
        "nstates_computed": int(len(values)),
        "ground_sectors": [int(s) for s in sector_list],
        "ground_energy": float(reference),
        "projector_completeness": completeness,
        "levels": levels,
        "lowest_by_star": lowest_by_star,
        "runtime": time.perf_counter() - started,
    }
    tag = f"{coupling:+.3f}".replace(".", "p")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    destination = OUTPUT / f"momentum_spectrum_{method}_{name}_{tag}.json"
    destination.write_text(json.dumps(record, indent=1, default=float))
    print(f"\nwrote {destination} ({record['runtime']:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
