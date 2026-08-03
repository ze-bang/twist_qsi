#!/usr/bin/env python3
"""Every eigenvalue of H from the Feshbach map -- and why the masked map has none.

For the UNMASKED map this works and is exact.  By Haynsworth inertia
additivity, for any z that is not a pole of the self-energy,

    #{eigenvalues of H below z} = #{poles below z}
                                  + #{negative eigenvalues of F(z) - z},

so bisection on that counting function locates every eigenvalue.  Validated at
J_pm/J_zz = -0.20: 4748 roots in the open intervals plus 8122 eigenvalues
sitting exactly ON degenerate poles (the combinations of degenerate spinon
states that decouple from the ice manifold) = 12870, agreeing with the exact
spectrum to 8.4e-10.

For the MASKED map it does NOT work, and the attempt is recorded here so that
it is not repeated.  Haynsworth applies to the Schur complement of a Hermitian
matrix; the masked map is the Schur complement of nothing, so the identity
fails and the counting function is not the counting function of any operator.
Worse, the mask makes the map block diagonal in the 25 transport sectors while
every block keeps poles at all 12780 spinon eigenvalues, so each block carries
its own near-complete root set and the union is overcomplete: the run at
J_pm/J_zz = -0.30 returned 31233 "roots" for a 12870-dimensional space.

The masked map's roots BELOW the lowest pole -- the gauge band -- remain exact
and unambiguous; they are what ``downfold.solve_roots`` computes.  Above the
continuum edge the masked map does not define a spectrum, which is precisely
why the counterterm Hamiltonian H + C of Proposition 1 exists, and why it pays
for a genuine spectrum with an O(1-Z) anchoring residue.

Usage: run_masked_full_spectrum.py <jpm> [--unmasked]
       (--unmasked is the validated mode; the masked mode is kept only to
        reproduce the negative result above)
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.linalg import eigh, eigvalsh

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "notes"))
sys.path.insert(0, str(ROOT / "campaign"))

import recompute_finite_size_artifact as geometry  # noqa: E402
from qsi_campaign.exact_band import (  # noqa: E402
    fixed_magnetization_basis,
    microscopic_character_hamiltonian,
    translation_sector_bases,
    cubic16_translation_permutations,
)
from qsi_campaign.downfold import build_blocks  # noqa: E402
from qsi_campaign.protocol import sector_project  # noqa: E402

OUTPUT = ROOT / "campaign" / "outputs" / "masked_full_spectrum"
OUTPUT.mkdir(parents=True, exist_ok=True)


def counting_function(blocks, z, masked, labels):
    """Number of winding-free eigenvalues below z (Haynsworth inertia).

    n_-(H - z) = n_-(QHQ - z) + n_-(F(z) - z): the number of eigenvalues below
    z is the number of self-energy poles below z plus the number of negative
    eigenvalues of F(z) - z.  Exact for any z that is not a pole.
    """
    gap = z - blocks.poles
    if np.min(np.abs(gap)) < 1e-13:                  # never evaluate on a pole
        gap = np.where(np.abs(gap) < 1e-13, 1e-13, gap)
    weights = 1.0 / gap
    sigma = (blocks.coupling * weights[:, None]).T @ blocks.coupling
    matrix = blocks.php + sigma
    if masked:
        # ``labels`` arrives as the precomputed boolean same-sector mask, so
        # the mask stays in real arithmetic; sector_project would promote to
        # complex and dominate the cost of this inner loop.
        matrix = np.where(labels, matrix, 0.0)
    values = np.linalg.eigvalsh(matrix)
    # roots below z are the branches that have already crossed, plus every
    # pole passed (each pole carries one root with it)
    return int(np.sum(values < z)) + int(np.sum(blocks.poles < z))


def find_roots(blocks, masked, labels, tolerance=1e-10, verbose=True):
    """Every root, by bisection on the counting function.

    Degenerate poles need care: a pole of multiplicity m to which only r
    combinations couple carries m - r exact eigenvalues sitting ON it (the
    decoupled combinations), and those are not roots of anything.  They are
    recovered from the jump of the counting function across the pole, and are
    identical in the masked and unmasked problems because the mask does not
    touch Q.
    """
    poles = np.sort(blocks.poles)
    distinct = poles[np.concatenate(([True], np.diff(poles) > 1e-11))]
    dimension = blocks.php.shape[0] + len(poles)
    floor = float(min(poles.min(), np.linalg.eigvalsh(blocks.php).min())) - 5.0
    edges = np.concatenate(([floor], distinct, [poles.max() + 20.0]))
    roots = []
    started = time.perf_counter()
    for k in range(len(edges) - 1):
        lo, hi = float(edges[k]), float(edges[k + 1])
        margin = min(1e-9, 0.001 * (hi - lo))
        if hi - lo < 4e-9:
            continue
        lo, hi = lo + margin, hi - margin
        n_lo = counting_function(blocks, lo, masked, labels)
        n_hi = counting_function(blocks, hi, masked, labels)
        for target in range(n_lo, n_hi):
            a, b = lo, hi
            for _ in range(50):
                mid = 0.5 * (a + b)
                if counting_function(blocks, mid, masked, labels) > target:
                    b = mid
                else:
                    a = mid
                if b - a < tolerance:
                    break
            roots.append(0.5 * (a + b))
        if verbose and k % 1000 == 0:
            print(f"    interval {k}/{len(edges)-1}, {len(roots)} roots, "
                  f"{time.perf_counter()-started:.0f}s", flush=True)
    # eigenvalues sitting exactly on degenerate poles
    at_poles = []
    for value in distinct:
        below = counting_function(blocks, value - 1e-9, masked, labels)
        above = counting_function(blocks, value + 1e-9, masked, labels)
        multiplicity = above - below - sum(
            1 for r in roots if value - 1e-9 < r < value + 1e-9)
        at_poles.extend([float(value)] * max(multiplicity, 0))
    if verbose:
        print(f"    {len(roots)} roots + {len(at_poles)} on poles "
              f"= {len(roots)+len(at_poles)} of {dimension}", flush=True)
    return np.sort(np.asarray(roots + at_poles))


def main() -> None:
    arguments = [v for v in sys.argv[1:] if not v.startswith("--")]
    masked = "--unmasked" not in sys.argv
    jpm = float(arguments[0])

    cluster = geometry.build_cluster("cubic", (1, 1, 1))
    states = fixed_magnetization_basis(cluster.n_sites, cluster.n_sites // 2)
    index = {int(s): i for i, s in enumerate(states)}
    ice_rows = np.asarray([index[int(s)] for s in cluster.ice_states])
    sectors = translation_sector_bases(
        states, cubic16_translation_permutations(cluster))
    from qsi_campaign.protocol import polarization_sector_labels
    sector = polarization_sector_labels(cluster)
    labels = sector[:, None] == sector[None, :]      # boolean same-sector mask

    started = time.perf_counter()
    hamiltonian = microscopic_character_hamiltonian(
        cluster, states, jpm, np.zeros(3))
    blocks = build_blocks(cluster, states, ice_rows, hamiltonian)
    print(f"blocks built ({time.perf_counter()-started:.0f}s)", flush=True)

    exact = np.sort(np.concatenate([
        eigvalsh(np.ascontiguousarray(
            (0.5 * ((basis.conj().T @ (hamiltonian @ basis)).toarray()
                    + (basis.conj().T @ (hamiltonian @ basis)
                       ).toarray().conj().T)).real))
        for basis in sectors.values()]))

    roots = find_roots(blocks, masked, labels)
    tag = f"{jpm:+.3f}".replace(".", "p") + ("" if masked else "_unmasked")
    record = {
        "jpm": jpm, "masked": masked, "n_roots": int(len(roots)),
        "dimension": int(len(states)),
        "runtime": time.perf_counter() - started,
    }
    if len(roots) == len(exact):
        record["max_deviation_from_exact"] = float(
            np.max(np.abs(roots - exact)))
    np.savez_compressed(OUTPUT / f"spectrum_{tag}.npz", jpm=jpm, roots=roots,
                        exact=exact, masked=masked)
    with open(OUTPUT / f"spectrum_{tag}.json", "w") as handle:
        json.dump(record, handle, indent=1, default=float)
    print(f"jpm={jpm:+.3f} masked={masked} roots={len(roots)}/{len(states)} "
          f"maxdev={record.get('max_deviation_from_exact', float('nan')):.3e} "
          f"({record['runtime']:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
