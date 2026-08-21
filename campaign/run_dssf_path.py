#!/usr/bin/env python3
"""S^zz(q,w) along a connected path through the fcc Brillouin zone.

Two things this answers that the eight-column panel could not.

**A connected path.**  FCC-32 has only eight commensurate momenta, so the
existing figure is eight discrete columns.  But `S^z_mu(q)` is diagonal in the
Ising basis for ANY q, so the amplitude

    A_n^mu(q) = sum_{i in mu} e^{-2 pi i q.r_i} <0|S^z_i|n>

is computable on a dense path.  Only the eight commensurate points are free of
a convention: at incommensurate q the phase e^{-2 pi i q.r_i} depends on which
periodic image represents site i, so the interpolation between them is a
smooth finite-cluster continuation, NOT new information.  Path points that
coincide with a genuine cluster momentum are flagged `exact` in the output and
should be the only ones quoted as numbers.

**Why the periodic calculation has weight at Gamma.**  U(1) conservation kills
the TOTAL longitudinal probe: sum_i S^z_i commutes with H, so
`|<n|S^z_total(0)|0>|^2 = 0` for every excited n.  It does not kill the
sublattice-resolved probe.  The four pyrochlore sublattice magnetisations are
not separately conserved -- the J_pm term S^+_i S^-_j moves S^z between
sublattices while preserving the sum -- so three of the four q=0 channels are
free to carry inelastic weight, and the existing figure plots their incoherent
sum.  This script reports both decompositions at every q:

    uniform  |sum_mu A_n^mu|^2         <- U(1)-protected, must vanish at Gamma
    trace    sum_mu |A_n^mu|^2         <- what the eight-column panel plots

so the Gamma column separates into an exact zero and a finite remainder, and
the "periodic has spurious Gamma weight" claim can be stated precisely: the
spurious weight is entirely in the sublattice-staggered channels.

Usage: run_dssf_path.py <jpm> [--depth=1] [--nstates=144] [--per-segment=24]
                        [--masked] [--jpmpm=0.0]
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

# Canonical fcc circuit, in the cubic r.l.u. that `allowed_momenta` uses
# (phase = exp(-2 pi i q.r) with r in conventional-cell units, so X sits at
# (0,0,1) and L at (1/2,1/2,1/2)).
CORNERS = (
    ("$\\Gamma$", (0.0, 0.0, 0.0)),
    ("X",         (0.0, 0.0, 1.0)),
    ("W",         (0.5, 0.0, 1.0)),
    ("L",         (0.5, 0.5, 0.5)),
    ("$\\Gamma$", (0.0, 0.0, 0.0)),
)


def build_path(per_segment: int):
    """Dense q along the circuit, with cumulative distance for the x axis."""
    points, labels, distance = [], [], []
    travelled = 0.0
    for index in range(len(CORNERS) - 1):
        (name, start), (_, stop) = CORNERS[index], CORNERS[index + 1]
        start = np.asarray(start, dtype=float)
        stop = np.asarray(stop, dtype=float)
        span = float(np.linalg.norm(stop - start))
        # include the endpoint only on the final segment, so corners are not
        # duplicated where two segments meet
        count = per_segment + 1 if index == len(CORNERS) - 2 else per_segment
        for step in range(count):
            fraction = step / per_segment
            points.append(start + fraction * (stop - start))
            distance.append(travelled + fraction * span)
            labels.append(name if step == 0 else "")
        travelled += span
    labels[-1] = CORNERS[-1][0]
    return np.asarray(points), np.asarray(distance), labels


def site_amplitudes(vectors, ground_row, spins):
    """`<0|S^z_i|n>` for every site i and every computed state n.

    `S^z_i` is diagonal, so this is one weighted contraction per site and needs
    no operator application.  Returns shape (n_states, n_sites).
    """
    weighted = vectors[:, ground_row][:, None] * spins      # (dim, n_sites)
    return vectors.conj().T @ weighted                      # (n_states, sites)


def spins_of(space, n_sites):
    bits = (space[:, None] >> np.arange(n_sites, dtype=np.uint64)) & np.uint64(1)
    return bits.astype(float) - 0.5


def channels(amplitudes, positions, sublattice, path, n_sites):
    """Uniform and trace weight per (path point, state).

    `amplitudes` is (n_states, n_sites) of `<0|S^z_i|n>`.
    """
    n_path = len(path)
    n_states = amplitudes.shape[0]
    uniform = np.zeros((n_path, n_states))
    trace = np.zeros((n_path, n_states))
    for index, momentum in enumerate(path):
        phase = np.exp(-2j * np.pi * (positions @ momentum))    # (n_sites,)
        per_sublattice = np.stack(
            [amplitudes @ (phase * (sublattice == mu)) for mu in range(4)],
            axis=0)                                             # (4, n_states)
        uniform[index] = np.abs(per_sublattice.sum(axis=0)) ** 2 / n_sites
        trace[index] = (np.abs(per_sublattice) ** 2).sum(axis=0) / n_sites
    return uniform, trace


def exact_flags(path, cluster):
    """Which path points are genuine cluster momenta (mod reciprocal lattice)."""
    allowed = allowed_momenta(cluster)
    flags = []
    for momentum in path:
        # a path point is exact when its phase pattern over the sites matches
        # some allowed momentum's, which is the only statement that survives
        # the periodic-image ambiguity
        reference = np.exp(-2j * np.pi * (cluster.positions @ momentum))
        hit = any(
            np.allclose(reference,
                        np.exp(-2j * np.pi * (cluster.positions @ candidate)))
            for candidate in allowed)
        flags.append(bool(hit))
    return flags


def periodic_spectrum(cluster, space, jpm, jpmpm, n_states):
    hamiltonian = build_hamiltonian(cluster, space, jpm, jpmpm).tocsr()
    k = min(n_states, hamiltonian.shape[0] - 2)
    values, vectors = eigsh(hamiltonian, k=k, which="SA", tol=1.0e-10)
    order = np.argsort(values)
    return values[order], vectors[:, order], np.arange(len(space))


def masked_spectrum(cluster, space, jpm, jpmpm, n_states):
    """Ground-sector block of the transport-masked theory."""
    labels = polarization_sector_labels(cluster)
    ice_states = np.sort(np.asarray(cluster.ice_states, dtype=np.uint64))
    ice_rows = np.searchsorted(space, ice_states)
    non_ice = np.setdiff1d(np.arange(len(space)), ice_rows)
    hamiltonian = build_hamiltonian(cluster, space, jpm, jpmpm).tocsr()

    grounds, sector_rows = {}, {}
    for sector in np.unique(labels):
        rows = np.concatenate([ice_rows[labels == sector], non_ice])
        sector_rows[int(sector)] = rows
        block = hamiltonian[np.ix_(rows, rows)]
        grounds[int(sector)] = float(eigsh(
            block, k=1, which="SA", tol=1e-10, return_eigenvectors=False)[0])
    lowest = min(grounds.values())
    ground_sectors = [s for s, e in grounds.items()
                      if e - lowest < DEGENERACY_TOL]
    if len(ground_sectors) != 1:
        print(f"  note: {len(ground_sectors)} degenerate ground sectors; "
              f"using {ground_sectors[0]}", flush=True)
    sector = ground_sectors[0]
    rows = sector_rows[sector]
    block = hamiltonian[np.ix_(rows, rows)]
    k = min(n_states, block.shape[0] - 2)
    values, vectors = eigsh(block, k=k, which="SA", tol=1e-10)
    order = np.argsort(values)
    return values[order], vectors[:, order], rows


def main() -> int:
    coupling = None
    depth, n_states, per_segment, jpmpm = 1, 144, 24, 0.0
    masked = "--masked" in sys.argv
    for token in sys.argv[1:]:
        if token.startswith("--depth="):
            depth = int(token.split("=", 1)[1])
        elif token.startswith("--nstates="):
            n_states = int(token.split("=", 1)[1])
        elif token.startswith("--per-segment="):
            per_segment = int(token.split("=", 1)[1])
        elif token.startswith("--jpmpm="):
            jpmpm = float(token.split("=", 1)[1])
        elif not token.startswith("--"):
            coupling = float(token)
    if coupling is None:
        raise SystemExit(__doc__)

    started = time.perf_counter()
    cluster = geometry.build_cluster("fcc", (2, 2, 2))
    space = bfs_space(cluster, depth, jpmpm=jpmpm)
    n_sites = int(cluster.n_sites)
    method = "masked" if masked else "periodic"
    print(f"{method} FCC-32 d={depth} jpm={coupling:+.3f}: "
          f"space={len(space):,} states={n_states}", flush=True)

    solver = masked_spectrum if masked else periodic_spectrum
    values, vectors, rows = solver(cluster, space, coupling, jpmpm, n_states)
    excitation = values - values[0]
    ground_rows = np.flatnonzero(excitation < DEGENERACY_TOL)
    print(f"  ground energy {values[0]:.9f}, "
          f"{len(ground_rows)} state(s) in the ground manifold", flush=True)

    spins = spins_of(space[rows], n_sites)
    positions = np.asarray(cluster.positions, dtype=float)
    sublattice = np.arange(n_sites) % 4
    path, distance, labels = build_path(per_segment)

    uniform = np.zeros((len(path), len(values)))
    trace = np.zeros((len(path), len(values)))
    for ground in ground_rows:
        amplitudes = site_amplitudes(vectors, ground, spins)
        one, two = channels(amplitudes, positions, sublattice, path, n_sites)
        uniform += one
        trace += two
    uniform /= len(ground_rows)
    trace /= len(ground_rows)
    # elastic line carries no information about the excitations
    uniform[:, excitation <= DEGENERACY_TOL] = 0.0
    trace[:, excitation <= DEGENERACY_TOL] = 0.0

    flags = exact_flags(path, cluster)
    gamma = int(np.argmin(np.linalg.norm(path, axis=1)))
    print(f"  Gamma: uniform={uniform[gamma].sum():.6e}  "
          f"trace={trace[gamma].sum():.6e}", flush=True)
    peak = int(np.argmax(trace.sum(axis=1)))
    print(f"  max trace weight {trace[peak].sum():.6f} at q={path[peak]}",
          flush=True)

    record = {
        "cluster": "fcc32", "method": method, "virtual_depth": depth,
        "jpm": coupling, "jpmpm": jpmpm, "space": int(len(space)),
        "nstates_computed": int(len(values)),
        "ground_energy": float(values[0]),
        "ground_degeneracy": int(len(ground_rows)),
        "corners": [name for name, _ in CORNERS],
        "per_segment": per_segment,
        "path": [[float(v) for v in q] for q in path],
        "distance": [float(d) for d in distance],
        "corner_label": labels,
        "is_exact_cluster_momentum": flags,
        "excitation": [float(v) for v in excitation],
        "uniform_weight": [[float(v) for v in row] for row in uniform],
        "trace_weight": [[float(v) for v in row] for row in trace],
        "runtime": time.perf_counter() - started,
    }
    tag = f"{coupling:+.3f}".replace(".", "p")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path_out = OUTPUT / f"dssfpath_{method}_fcc32_L{depth}_{tag}.json"
    path_out.write_text(json.dumps(record, default=float))
    print(f"wrote {path_out} ({record['runtime']:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
