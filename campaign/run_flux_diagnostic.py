#!/usr/bin/env python3
"""What the non-photon levels are: flux structure, level by level.

One-point functions cannot answer this.  Every level is a momentum eigenstate,
so `<n|Phi_h|n>` is translationally invariant and identical on every hexagon
whatever the level is -- a localized monopole pair looks exactly as uniform as
a photon once symmetry is respected.  The discriminator has to be the
CONNECTED two-point function

    C(h, h') = <n|Phi_h Phi_h'|n> - <n|Phi_h|n><n|Phi_h'|n>,

binned by the distance between hexagon centres.  A single photon is an extended
plane wave and its flux correlations decay slowly across the cluster; a
monopole-antimonopole pair is two localized defects and correlates over a short
range with a pronounced near-neighbour peak.

`Phi_h` is the ring-exchange operator on hexagon `h`: it flips all six spins
when they alternate around the loop, which is exactly the lattice magnetic
flux term of the emergent gauge theory.

Usage: run_flux_diagnostic.py <jpm> [--cluster=fcc32] [--depth=1]
                              [--nstates=60] [--levels=6]
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
from run_momentum_resolved_spectrum import (  # noqa: E402
    CLUSTERS, DEGENERACY_TOL, LEVEL_TOL, channel_weights, momentum_weights,
    star, translation_permutations)

OUTPUT = ROOT / "campaign" / "outputs" / "wp3_dssf"


def hexagon_data(cluster):
    """(mask, ordered sites, centre) for every CONTRACTIBLE hexagon.

    `simple_cycles` returns every 6-cycle, and on these clusters three quarters
    of them wrap the periodic box -- 96 of 128 on FCC-32, 48 of 64 on cubic-16,
    leaving exactly the 4-per-primitive-cell contractible count.  A wrapping
    loop is not a plaquette of the gauge theory: its ring exchange transports
    polarization by a lattice vector and changes transport sector, which is the
    winding artifact the mask exists to remove.  Including them in a magnetic
    flux measurement mixes the artifact back in, so they are dropped here on
    the winding number, which `simple_cycles` already records.
    """
    masks, orders, centres = [], [], []
    positions = np.asarray(cluster.positions, dtype=float)
    for path, winding in cluster.hexes:
        if any(int(v) != 0 for v in winding):
            continue
        sites = [int(v) for v in path]
        mask = 0
        for site in sites:
            mask |= 1 << site
        masks.append(np.uint64(mask))
        orders.append(np.asarray(sites, dtype=np.int64))
        centres.append(positions[sites].mean(axis=0))
    return np.asarray(masks), orders, np.asarray(centres)


def magnetic_amplitudes(vectors, flux_of_ground, centres, momenta):
    """`|<n|Phi(q)|0>|^2` -- the B-channel counterpart of S^zz.

    `Phi(q) = sum_h exp(-2 pi i q.r_h) Phi_h` over plaquette centres, exactly
    parallel to `S^z_mu(q)` over sites.  The point of computing it: a photon is
    a gauge mode carrying BOTH an electric and a magnetic component, so it must
    appear in this channel whatever the E-channel form factor does.  The X
    level that `S^zz` cannot see is either bright here -- confirming it is the
    photon and the E-zero is a form-factor effect -- or dark in both, in which
    case it is not the photon and the assignment inverts.
    """
    out = np.zeros((vectors.shape[1], len(momenta)))
    overlaps = vectors.conj().T @ flux_of_ground        # (n_states, n_hex)
    for q_index, momentum in enumerate(momenta):
        phase = np.exp(-2j * np.pi * (centres @ momentum))
        out[:, q_index] = np.abs(overlaps @ phase) ** 2
    return out


def dual_sites(centres, lvecs):
    """Group the plaquettes into dual-diamond sites, with orientation signs.

    A hexagonal plaquette of the pyrochlore is a BOND of the dual diamond
    lattice, so it has two endpoints and every dual site carries four of them
    -- diamond is 4-coordinate.  Four bonds meeting at a diamond site have
    midpoints forming a regular tetrahedron, so the four plaquettes around a
    dual site are mutually equidistant at the minimum plaquette separation:
    the sites are exactly the 4-cliques of the minimum-distance graph.

    Everything here is checked rather than trusted, because this is the part I
    am most likely to get wrong: the clique count must be n_plaquettes / 2,
    every plaquette must land in exactly two cliques, and the signs must make
    `sum_d Q_d` vanish identically.
    """
    count = len(centres)
    separation = np.full((count, count), np.inf)
    for shift in np.ndindex(3, 3, 3):
        offset = (np.asarray(shift, dtype=float) - 1.0) @ lvecs
        candidate = np.linalg.norm(
            centres[:, None, :] - centres[None, :, :] - offset, axis=2)
        separation = np.minimum(separation, candidate)
    off = separation[~np.eye(count, dtype=bool)]
    step = float(off.min())
    adjacent = np.isclose(separation, step) & ~np.eye(count, dtype=bool)

    cliques = set()
    for a in range(count):
        neighbours = np.flatnonzero(adjacent[a])
        for b in neighbours[neighbours > a]:
            for c in neighbours[neighbours > b]:
                if not adjacent[b, c]:
                    continue
                for d in neighbours[neighbours > c]:
                    if adjacent[b, d] and adjacent[c, d]:
                        cliques.add((int(a), int(b), int(c), int(d)))
    cliques = sorted(cliques)

    membership = np.zeros(count, dtype=int)
    for clique in cliques:
        for h in clique:
            membership[h] += 1
    ok = (len(cliques) == count // 2) and bool(np.all(membership == 2))
    print(f"  dual lattice: {len(cliques)} sites (expected {count // 2}), "
          f"plaquettes per site 4, sites per plaquette "
          f"{membership.min()}-{membership.max()} (expected 2) -> "
          f"{'OK' if ok else 'FAILED'}", flush=True)
    if not ok:
        raise RuntimeError("dual-diamond incidence check failed")

    # Orientation: each plaquette is a bond joining two dual sites; give it +1
    # at the first and -1 at the second so that sum_d Q_d telescopes to zero.
    signs = np.zeros((len(cliques), count))
    seen: dict[int, int] = {}
    for index, clique in enumerate(cliques):
        for h in clique:
            signs[index, h] = 1.0 if h not in seen else -1.0
            seen[h] = index
    residual = float(np.abs(signs.sum(axis=0)).max())
    print(f"  orientation check: max |sum_d eps(h,d)| = {residual:.1e} "
          f"(must be 0)", flush=True)
    if residual > 1e-12:
        raise RuntimeError("orientation signs do not telescope")
    return cliques, signs


def flux_apply(vector, block_states, lookup_sorted, order_index, mask, sites,
               n_ice):
    """`Phi_h |v>`: flip the six spins wherever they alternate around h.

    Applied on the ICE rows only.  The emergent magnetic flux is defined on the
    gauge-invariant manifold, and the ring exchange is closed there; applied to
    the spinon rows of a depth-1 space it walks straight out of the truncation,
    which is what the first version hit.  The caller is told how much of each
    eigenvector lives outside the ice block so the omission is visible rather
    than silent.
    """
    bits = ((block_states[:, None] >> sites.astype(np.uint64))
            & np.uint64(1)).astype(np.int8)
    alternating = np.ones(len(block_states), dtype=bool)
    for step in range(len(sites)):
        alternating &= bits[:, step] != bits[:, (step + 1) % len(sites)]
    # ICE rows only.  The block is laid out as [ice of this sector | non-ice],
    # so the ice rows are the first n_ice entries.  Applying the ring exchange
    # to a spinon row walks out of the depth-1 truncation.
    alternating[n_ice:] = False
    out = np.zeros_like(vector)
    if not alternating.any():
        return out, 0.0
    source = np.flatnonzero(alternating)
    flipped = block_states[source] ^ mask
    slot = np.searchsorted(lookup_sorted, flipped)
    slot = np.minimum(slot, len(lookup_sorted) - 1)
    inside = lookup_sorted[slot] == flipped
    dropped = float(np.sum(vector[source[~inside]] ** 2))
    target = order_index[slot[inside]]
    out[target] = vector[source[inside]]
    return out, dropped


def main() -> int:
    coupling, name, depth, n_states, n_levels = None, "fcc32", 1, 60, 6
    for token in sys.argv[1:]:
        if token.startswith("--cluster="):
            name = token.split("=", 1)[1]
        elif token.startswith("--depth="):
            depth = int(token.split("=", 1)[1])
        elif token.startswith("--nstates="):
            n_states = int(token.split("=", 1)[1])
        elif token.startswith("--levels="):
            n_levels = int(token.split("=", 1)[1])
        elif not token.startswith("--"):
            coupling = float(token)
    if coupling is None:
        raise SystemExit(__doc__)

    started = time.perf_counter()
    cluster = geometry.build_cluster(*CLUSTERS[name])
    space = bfs_space(cluster, depth)
    hamiltonian = build_hamiltonian(cluster, space, coupling, 0.0).tocsr()
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
    sector = [s for s, e in grounds.items() if e - lowest < DEGENERACY_TOL][0]
    rows = sector_rows[sector]
    n_ice = int(np.sum(labels == sector))
    block = hamiltonian[np.ix_(rows, rows)]
    values, vectors = eigsh(block, k=min(n_states, block.shape[0] - 2),
                            which="SA", tol=1e-10)
    order = np.argsort(values)
    values, vectors = values[order], vectors[:, order]
    excitation = values - values[0]
    print(f"{name} jpm={coupling:+.3f} sector {sector}: "
          f"{len(values)} states to omega={excitation[-1]:.5f}", flush=True)

    block_states = space[rows]
    sort_order = np.argsort(block_states)
    lookup_sorted = block_states[sort_order]
    translations, permutations = translation_permutations(cluster)
    momenta = allowed_momenta(cluster)
    phases = np.exp(-2j * np.pi * (translations @ np.asarray(momenta).T)).T
    weights = momentum_weights(vectors, block_states, permutations, phases)

    masks, orders, centres = hexagon_data(cluster)
    print(f"  {len(masks)} contractible hexagons "
          f"(of {len(cluster.hexes)} 6-cycles; the rest wrap the box)",
          flush=True)
    lvecs = np.asarray(cluster.Lvecs, dtype=float)
    # B channel: Phi_h applied to the sector ground state, once per plaquette.
    flux_of_ground = np.empty((len(vectors), len(masks)))
    for h in range(len(masks)):
        flux_of_ground[:, h], _ = flux_apply(
            vectors[:, 0], block_states, lookup_sorted, sort_order,
            masks[h], orders[h], n_ice)
    b_weight = magnetic_amplitudes(vectors, flux_of_ground, centres, momenta)
    # E channel, for the side-by-side comparison
    e_weight = channel_weights(vectors, block_states,
                               np.asarray(cluster.positions, dtype=float),
                               momenta, int(cluster.n_sites)).sum(axis=2)
    # `not shift` is False for EVERY tuple including (0,0,0) -- a non-empty
    # tuple is truthy -- so the previous form min'd against a zero matrix and
    # collapsed every separation to 0.  Start from +inf instead.
    separation = np.full((len(masks),) * 2, np.inf)
    for shift in np.ndindex(3, 3, 3):
        offset = (np.asarray(shift, dtype=float) - 1.0) @ lvecs
        candidate = np.linalg.norm(
            centres[:, None, :] - centres[None, :, :] - offset, axis=2)
        separation = np.minimum(separation, candidate)

    # levels
    boundaries, start = [], 0
    for index in range(1, len(values) + 1):
        if index == len(values) or values[index] - values[start] > LEVEL_TOL:
            boundaries.append((start, index))
            start = index

    report = []
    print("\n  omega        deg  star    <Phi>     connected C(r) by shell")
    for start, stop in boundaries[:n_levels + 1]:
        occupied = [q for q in range(len(momenta))
                    if weights[start:stop, q].sum() > 1e-6]
        name_star = star(momenta[occupied[0]]) if occupied else "?"
        mean = np.zeros(len(masks))
        correlator = np.zeros((len(masks), len(masks)))
        dropped_total = 0.0
        non_ice_weight = float(np.mean(np.sum(
            vectors[n_ice:, start:stop] ** 2, axis=0)))
        per_state_flux = []
        for column in range(start, stop):
            vector = vectors[:, column]
            applied = np.empty((len(masks), len(vector)))
            for h in range(len(masks)):
                applied[h], dropped = flux_apply(
                    vector, block_states, lookup_sorted, sort_order,
                    masks[h], orders[h], n_ice)
                dropped_total = max(dropped_total, dropped)
                mean[h] += float(vector @ applied[h])
            correlator += applied @ applied.T
            per_state_flux.append(applied)
        degeneracy = stop - start
        mean /= degeneracy
        correlator /= degeneracy
        connected = correlator - np.outer(mean, mean)
        # bin by minimum-image separation
        shells = np.unique(np.round(separation, 6))
        binned = [(float(r), float(np.mean(np.abs(connected[
            np.isclose(separation, r)])))) for r in shells[:5]]
        summary = "  ".join(f"r={r:.2f}:{c:.3e}" for r, c in binned)
        onsite = binned[0][1] if binned else 0.0
        tail = binned[-1][1] if len(binned) > 1 else 0.0
        # Correlation length from the second moment of the OFF-DIAGONAL
        # connected correlator.  This is the discriminator: an extended mode
        # spreads its flux correlations over the whole cluster and returns
        # xi comparable to the largest available separation, a localized
        # vison pair returns a small xi.  Taken as a second moment rather than
        # an exponential fit because on 32 plaquettes there are too few shells
        # to fit a decay, and a ratio of two shells (the earlier
        # tail/onsite) throws away everything in between.
        off = ~np.eye(len(masks), dtype=bool)
        magnitude = np.abs(connected)[off]
        radius = separation[off]
        total_weight = float(magnitude.sum())
        xi = (float(np.sqrt((magnitude * radius ** 2).sum() / total_weight))
              if total_weight > 0 else 0.0)
        # Magnetic charge density: Q_d = sum_{h in d} eps(h,d) Phi_h, and the
        # observable is <Q_d^2> summed over dual sites.  This is LOCAL, so
        # unlike a correlation length it does not need room for a decay and
        # stays meaningful on a 32-plaquette cluster.  A photon has div B = 0
        # up to compactness; a vison pair carries real magnetic charge.
        # E and B channel weight for this level, at its own momenta.  The dual
        # -diamond magnetic-charge construction is not used: its incidence
        # check failed (1 clique of an expected 16) and no candidate relation
        # -- nearest shell, shared-site counts -- reproduced the required
        # structure, so the plaquette/dual-site mapping is not established.
        # These two channels answer the same question without it.
        b_total = float(b_weight[start:stop, occupied].sum())
        e_total = float(e_weight[start:stop, occupied].sum())
        report.append({
            "excitation": float(excitation[start]),
            "degeneracy": int(degeneracy),
            "star": name_star,
            "mean_flux": float(mean.mean()),
            "shells": binned,
            "tail_over_onsite": float(tail / onsite) if onsite else 0.0,
            "xi_flux": xi,
            "b_channel": b_total,
            "e_channel": e_total,
            "max_dropped_weight": dropped_total,
            "non_ice_weight": non_ice_weight,
        })
        print(f"  {excitation[start]:.6f} {degeneracy:4d}  {name_star:6s} "
              f"{mean.mean():8.4f}  B={b_total:10.3e}  E={e_total:10.3e}   "
              f"drop={dropped_total:.1e} spinon={non_ice_weight:.3f}",
              flush=True)

    print("\n  E vs B channel per level -- a photon must be BRIGHT in B:")
    for entry in report:
        print(f"    omega={entry['excitation']:.6f} {entry['star']:6s} "
              f"B={entry['b_channel']:10.3e}  E={entry['e_channel']:10.3e}")
    largest = float(separation.max())
    print(f"\n  flux correlation length xi (largest separation on this "
          f"cluster is {largest:.4f});")
    print("  extended/photon-like -> xi near that value, "
          "localized/vison-like -> xi well below it:")
    for entry in report:
        print(f"    omega={entry['excitation']:.6f} {entry['star']:6s} "
              f"xi={entry['xi_flux']:.4f}  "
              f"xi/max={entry['xi_flux'] / largest:.3f}  "
              f"(tail/onsite {entry['tail_over_onsite']:.4f})")

    record = {"cluster": name, "jpm": coupling, "sector": int(sector),
              "n_hexagons": int(len(masks)), "levels": report,
              "runtime": time.perf_counter() - started}
    tag = f"{coupling:+.3f}".replace(".", "p")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    destination = OUTPUT / f"flux_diagnostic_{name}_{tag}.json"
    destination.write_text(json.dumps(record, indent=1, default=float))
    print(f"\nwrote {destination} ({record['runtime']:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
