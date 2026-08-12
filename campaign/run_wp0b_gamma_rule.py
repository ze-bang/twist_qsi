"""WP0b: is the Gamma-point longitudinal selection rule a theorem or a cubic-16 accident?

The winding-free longitudinal response at `q = 0` vanishes identically on
cubic-16.  The reason recorded in `run_dssf_counterterm.py` is a combinatoric
one: the four `q = 0` sublattice magnetisations

```text
m_mu(C) = sum_{i in sublattice mu} S_i^z
```

are *exactly constant* within each transport sector, so `P S^z_{q=0} P` --
which is diagonal in the Ising basis -- is a c-number on every mask block.
It therefore commutes with the whole masked band operator and cannot connect
the winding-free ground multiplet to anything.

That argument was verified only on cubic-16, and the docstring's stated reason
("the transport label is a linear function of them") proves the implication in
the wrong direction: `m -> p` linear does NOT imply `p -> m` single valued.
The constancy is an empirical fact about which `(p, m)` pairs the ice manifold
actually realises, and it must be re-checked on a second cluster before any
selection-rule claim enters WP3.

This script decides it on the 2970 FCC-32 ice states, with cubic-16 run
alongside as the control that reproduces the known result.  Pure
combinatorics: no Hamiltonian, no coupling, no eigensolver.

Four tests per cluster:

1. *constancy* -- the within-sector spread of each `2 m_mu` (an integer), which
   must be exactly zero for the rule to hold;
2. *end-to-end* -- the commutator `[D_Gamma, A]` for a random Hermitian `A`
   that is block diagonal in the transport sectors (i.e. any masked band
   operator whatsoever).  Constancy makes this exactly zero; this test is what
   the physics claim actually needs, and it does not depend on trusting the
   argument above;
3. *contrast* -- the same spread at the cluster's nonzero allowed momenta,
   which must NOT vanish, or the "rule" would be a triviality of the ice
   manifold rather than a statement about `q = 0`;
4. *unmasked control* -- the same commutator against a random Hermitian
   operator with no mask, which must be large: the rule has to be a property
   of the mask, not of `S^z_{q=0}`.

Usage: run_wp0b_gamma_rule.py [cubic16] [fcc32] ...   (default: both)
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "notes"))

from qsi_campaign.protocol import polarization_sector_labels  # noqa: E402
from recompute_finite_size_artifact import FCC_A, build_cluster  # noqa: E402

OUTDIR = ROOT / "campaign" / "outputs" / "wp0_preflight"

CLUSTERS = {
    "cubic16": ("cubic", (1, 1, 1)),
    "fcc32": ("fcc", (2, 2, 2)),
}


def sublattice_magnetisations(cluster) -> np.ndarray:
    """`2 m_mu` per ice configuration: an exact integer array, n_ice x 4."""
    states = np.asarray(cluster.ice_states, dtype=np.uint64)
    n_sites = int(cluster.n_sites)
    bits = (states[:, None] >> np.arange(n_sites, dtype=np.uint64)) & np.uint64(1)
    sigma = 2 * bits.astype(np.int64) - 1              # 2 S^z_i, exactly +-1
    sublattice = np.arange(n_sites) % 4
    return np.stack(
        [sigma[:, sublattice == mu].sum(axis=1) for mu in range(4)], axis=1
    )


def allowed_momenta(cluster) -> np.ndarray:
    """The cluster's allowed momenta, in the units of `positions @ q`.

    `q` is allowed when `exp(-2 pi i q.L_a) = 1` for every box vector, i.e.
    `q = n @ inv(Lvecs)`.  Distinct momenta are those differing by a reciprocal
    vector of the underlying Bravais lattice, which for both cluster shapes is
    the FCC lattice of tetrahedron centres -- NOT `Lvecs / shape`.  On the
    cubic cluster those differ by a factor of four: one conventional cubic cell
    holds four FCC primitive cells, and reducing modulo the cubic cell instead
    would fold every momentum onto Gamma and silently empty the contrast test.
    The count returned equals the number of primitive cells in the cluster:
    four for cubic-16 (Gamma and the three X points), eight for FCC-32.
    """
    lvecs = np.asarray(cluster.Lvecs, dtype=float)
    inverse = np.linalg.inv(lvecs)
    span = max(cluster.shape) * (2 if cluster.basis == "cubic" else 1)
    kept: list[np.ndarray] = []
    primitive = np.asarray(FCC_A, dtype=float)
    for n in np.ndindex(*(span + 1,) * 3):
        q = np.asarray(n, dtype=float) @ inverse
        # reduce modulo the primitive reciprocal lattice: q is a duplicate of
        # q' when (q - q') . a is an integer for every primitive vector a
        duplicate = False
        for other in kept:
            residual = (q - other) @ primitive.T
            if np.allclose(residual, np.round(residual), atol=1e-9):
                duplicate = True
                break
        if not duplicate:
            kept.append(q)
    return np.array(kept)


def structure_factor_sources(cluster, momentum) -> np.ndarray:
    """The four sublattice-resolved diagonal sources `P S^z_{q,mu} P`."""
    states = np.asarray(cluster.ice_states, dtype=np.uint64)
    n_sites = int(cluster.n_sites)
    bits = (states[:, None] >> np.arange(n_sites, dtype=np.uint64)) & np.uint64(1)
    spins = bits.astype(float) - 0.5
    sublattice = np.arange(n_sites) % 4
    phase = np.exp(-2j * np.pi * (np.asarray(cluster.positions) @ momentum))
    return np.stack(
        [spins @ (phase * (sublattice == mu)) for mu in range(4)], axis=0
    )


def masked_random_operator(labels: np.ndarray, seed: int) -> np.ndarray:
    """A random Hermitian operator with the mask's exact block structure.

    Standing in for *any* winding-free band operator: the selection rule must
    hold for all of them, not for one particular coupling's fold.
    """
    rng = np.random.default_rng(seed)
    n = len(labels)
    full = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    full = full + full.conj().T
    return full * (labels[:, None] == labels[None, :])


def analyse(name: str) -> dict:
    basis, shape = CLUSTERS[name]
    started = time.time()
    cluster = build_cluster(basis, shape)
    labels = polarization_sector_labels(cluster)
    n_ice = int(cluster.n_ice)
    sectors, dims = np.unique(labels, return_counts=True)
    build_seconds = time.time() - started

    # --- test 1: within-sector spread of the integer 2 m_mu at q = 0 --------
    two_m = sublattice_magnetisations(cluster)
    spread = np.zeros(4, dtype=np.int64)
    for sector in sectors:
        rows = two_m[labels == sector]
        spread = np.maximum(spread, rows.max(axis=0) - rows.min(axis=0))
    constant = bool(spread.max() == 0)

    # how much of the sector label the magnetisations see: if the map
    # sector -> m is also injective the two labels are equivalent, which is a
    # stronger statement than the rule needs but worth recording
    per_sector = np.array(
        [two_m[labels == sector][0] for sector in sectors]
    ) if constant else np.empty((0, 4), dtype=np.int64)
    distinct_m = int(len(np.unique(per_sector, axis=0))) if constant else -1

    # --- test 2: the commutator the physics claim actually needs -----------
    # Per sublattice, NOT summed over them: the total S^z at q = 0 is the same
    # number on every ice configuration (N_up is fixed by the ice rule), so the
    # summed operator is a multiple of the identity and commutes with
    # everything.  Testing it would pass unconditionally and prove nothing --
    # and its unmasked control would also come out zero, which is how this test
    # announced the mistake.  The DSSF sums |matrix element|^2 sublattice by
    # sublattice, so the sublattice-resolved operators are the physical ones.
    sources = structure_factor_sources(cluster, np.zeros(3)).real
    operator = masked_random_operator(labels, seed=20260809)
    scale = float(np.abs(operator).max())
    commutator_norm = 0.0
    for diagonal in sources:
        commutator = diagonal[:, None] * operator - operator * diagonal[None, :]
        commutator_norm = max(commutator_norm, float(np.abs(commutator).max()))

    # --- test 4: the same thing with the mask switched off ----------------
    rng = np.random.default_rng(11)
    unmasked = rng.standard_normal((n_ice, n_ice)) + 1j * rng.standard_normal(
        (n_ice, n_ice)
    )
    unmasked = unmasked + unmasked.conj().T
    raw_commutator = 0.0
    for diagonal in sources:
        raw_commutator = max(raw_commutator, float(np.abs(
            diagonal[:, None] * unmasked - unmasked * diagonal[None, :]).max()))
    total_is_constant = bool(
        np.ptp(sources.sum(axis=0)) < 1e-12)          # recorded, not asserted

    # --- test 3: contrast at the nonzero allowed momenta ------------------
    contrast = []
    for momentum in allowed_momenta(cluster):
        if np.allclose(momentum, 0.0):
            continue
        sources = structure_factor_sources(cluster, momentum)
        worst = 0.0
        for sector in sectors:
            rows = sources[:, labels == sector]
            worst = max(worst, float(np.abs(rows - rows[:, :1]).max()))
        contrast.append(
            {"momentum": [float(x) for x in momentum], "max_spread": worst}
        )

    verdict = (
        constant
        and commutator_norm < 1e-10 * max(scale, 1.0)
        and raw_commutator > 1e-3
        and all(entry["max_spread"] > 1e-9 for entry in contrast)
    )

    result = {
        "cluster": name,
        "n_sites": int(cluster.n_sites),
        "n_ice": n_ice,
        "n_sectors": int(len(sectors)),
        "sector_dims": [int(d) for d in dims],
        "two_m_within_sector_spread": [int(x) for x in spread],
        "gamma_rule_holds": constant,
        "distinct_magnetisation_tuples": distinct_m,
        "label_equivalent_to_magnetisation": bool(
            constant and distinct_m == len(sectors)
        ),
        "masked_commutator_max": commutator_norm,
        "masked_operator_scale": scale,
        "unmasked_commutator_max": raw_commutator,
        "total_magnetisation_is_constant": total_is_constant,
        "nonzero_momentum_contrast": contrast,
        "verdict_pass": bool(verdict),
        "build_seconds": build_seconds,
        "seconds": time.time() - started,
    }
    return result


def main() -> int:
    names = sys.argv[1:] or ["cubic16", "fcc32"]
    unknown = [n for n in names if n not in CLUSTERS]
    if unknown:
        raise SystemExit(f"unknown cluster(s): {unknown}; choose from {list(CLUSTERS)}")

    OUTDIR.mkdir(parents=True, exist_ok=True)
    results = []
    for name in names:
        print(f"=== {name} ===", flush=True)
        result = analyse(name)
        results.append(result)
        print(
            f"  {result['n_ice']} ice states, {result['n_sectors']} transport "
            f"sectors, dims {min(result['sector_dims'])}-{max(result['sector_dims'])}"
        )
        print(
            f"  test 1 constancy: max within-sector spread of 2 m_mu = "
            f"{result['two_m_within_sector_spread']} "
            f"-> {'CONSTANT' if result['gamma_rule_holds'] else 'NOT CONSTANT'}"
        )
        print(
            f"  test 2 masked commutator max_mu |[D_mu, A]| = "
            f"{result['masked_commutator_max']:.3e} "
            f"(operator scale {result['masked_operator_scale']:.3f})"
        )
        print(
            f"  test 3 contrast: nonzero-q spreads "
            + ", ".join(
                f"{e['max_spread']:.3f}" for e in result["nonzero_momentum_contrast"]
            )
        )
        print(
            f"  test 4 unmasked control |[D_Gamma, A_raw]| = "
            f"{result['unmasked_commutator_max']:.3e}"
        )
        print(
            f"  distinct magnetisation tuples {result['distinct_magnetisation_tuples']}"
            f" vs {result['n_sectors']} sectors -> label equivalent: "
            f"{result['label_equivalent_to_magnetisation']}"
        )
        print(f"  VERDICT: {'PASS' if result['verdict_pass'] else 'FAIL'} "
              f"({result['seconds']:.1f} s)", flush=True)

    path = OUTDIR / "wp0b_gamma_rule.json"
    path.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {path}")

    if all(r["verdict_pass"] for r in results):
        print("WP0b PASS: the Gamma selection rule is a property of the "
              "construction, not a cubic-16 accident. WP3 may claim it.")
        return 0
    print("WP0b FAIL: the rule does not generalise; WP3 must not claim it.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
