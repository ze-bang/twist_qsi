"""Why does the Gaussian theory have exactly two transverse modes, and why
are they degenerate?

The counting argument, which we then verify:

  The emergent gauge field A_l lives on the links of the DIAMOND lattice --
  equivalently on the pyrochlore sites, which are the diamond bond midpoints.
  Per cubic cell the diamond lattice has 2 sites and 4 links, and the pyrochlore
  has 4 hexagons.  So at each momentum the curl C is a 4x4 matrix (4 plaquettes
  by 4 links).

    - Gauge invariance kills 2 of the 4 link modes: C @ d0 = 0, where d0 is the
      lattice gradient, and d0 has one column per diamond site, i.e. 2 per cell.
      Hence rank(C) <= 2.
    - Generically the rank is exactly 2, leaving TWO physical transverse
      polarizations -- the same count as Maxwell in three dimensions, for the
      same reason (4 link d.o.f. - 2 gauge = 2).

  That explains "two".  It does NOT explain why the two are DEGENERATE.  In
  continuum Maxwell the degeneracy follows from rotational symmetry, which the
  lattice breaks.  So the question this script settles is whether sigma_1 =
  sigma_2 survives at GENERIC momenta or only at the high-symmetry points we
  happened to resolve.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for sub in ("src", "notes", "campaign"):
    sys.path.insert(0, str(ROOT / sub))

import recompute_finite_size_artifact as geometry           # noqa: E402
from run_ring48_dssf import contractible                    # noqa: E402
from ring48_gaussian import build_curl, build_gradient      # noqa: E402


def curl_block(C, positions, q):
    """C restricted to the momentum-q link subspace, as a 4x4 matrix."""
    ph = np.exp(-2j * np.pi * (positions @ q))
    n = len(positions)
    sub = np.arange(n) % 4
    # one Bloch column per sublattice
    V = np.zeros((n, 4), dtype=complex)
    for mu in range(4):
        sel = sub == mu
        V[sel, mu] = ph[sel]
        V[:, mu] /= np.linalg.norm(V[:, mu])
    return C @ V


def main() -> int:
    orig = geometry.enumerate_ice_states
    geometry.enumerate_ice_states = lambda a, b: np.array([0], dtype=np.uint64)
    cl = geometry.build_cluster("fcc", (2, 2, 3))
    geometry.enumerate_ice_states = orig

    pos = np.asarray(cl.positions)
    hexes = contractible(cl)
    C = build_curl(cl, hexes)
    d0 = build_gradient(cl)

    print("=" * 74)
    print("The counting: 4 links/cell - 2 gauge modes = 2 transverse modes")
    print("=" * 74)
    print(f"  pyrochlore sites (= diamond links) : {len(pos)}")
    print(f"  contractible hexagons (plaquettes) : {len(hexes)}")
    print(f"  gauge (gradient) columns           : {d0.shape[1]}"
          f"   [= diamond sites]")
    print(f"  ||C @ d0||_max                     : "
          f"{np.abs(C @ d0).max():.3e}   [gauge invariance]")
    print(f"  rank(C)                            : "
          f"{np.linalg.matrix_rank(C)}")
    print(f"  {len(pos)} links - {d0.shape[1]} gauge = "
          f"{len(pos) - d0.shape[1]} physical modes over "
          f"{len(pos)//4} cells = 2 per cell")

    print()
    print("=" * 74)
    print("Is sigma_1 = sigma_2 at GENERIC momenta, or only at symmetric ones?")
    print("=" * 74)

    # Random vectors are NOT allowed momenta of the torus, so their Bloch
    # blocks are meaningless.  Use a larger cluster whose allowed set contains
    # genuinely low-symmetry momenta, and test every one of them.
    from ring48_momenta_fix import fcc_momenta

    geometry.enumerate_ice_states = lambda a, b: np.array([0], dtype=np.uint64)
    SHAPE = tuple(int(x) for x in (sys.argv[1:4] or (4, 4, 4)))
    big = geometry.build_cluster("fcc", SHAPE)
    geometry.enumerate_ice_states = orig
    bpos = np.asarray(big.positions)
    bhex = contractible(big)
    Cb = build_curl(big, bhex)
    qs, shape = fcc_momenta(big)
    print(f"  testing every allowed momentum of fcc{shape} "
          f"({len(bpos)} sites, {len(qs)} momenta)\n")

    def star_class(q):
        a = np.sort(np.abs(np.round(q, 9)))[::-1]
        if np.allclose(a, 0):
            return "Gamma"
        if np.allclose(a[0], a[1]) and np.allclose(a[1], a[2]):
            return "<111> line"
        if np.allclose(a[1], 0) and np.allclose(a[2], 0):
            return "<100> line"
        if np.allclose(a[2], 0):
            return "(hk0) plane"
        if np.allclose(a[0], a[1]) or np.allclose(a[1], a[2]):
            return "(hhl) plane"
        return "GENERIC"

    buckets = {}
    for q in qs:
        blk = curl_block(Cb, bpos, q)
        sv = np.sort(np.linalg.svd(blk, compute_uv=False))[::-1]
        if sv[0] < 1e-9:
            continue
        buckets.setdefault(star_class(q), []).append(
            (sv[1] / sv[0], sv[2] / sv[0]))

    print(f"  {'class':<14s} {'count':>6s} {'min s2/s1':>11s} "
          f"{'max s2/s1':>11s} {'max s3/s1':>11s} {'degenerate?':>12s}")
    print("  " + "-" * 70)
    all_deg = True
    for k in sorted(buckets, key=lambda k: -len(buckets[k])):
        v = np.array(buckets[k])
        deg = bool(np.all(np.abs(v[:, 0] - 1) < 1e-9))
        all_deg &= deg
        print(f"  {k:<14s} {len(v):>6d} {v[:,0].min():>11.8f} "
              f"{v[:,0].max():>11.8f} {v[:,1].max():>11.2e} "
              f"{'YES' if deg else 'NO':>12s}")

    print()
    if all_deg:
        print("  ==> The transverse doublet is degenerate at EVERY allowed")
        print("      momentum tested, generic ones included.  The")
        print("      degeneracy is a property of the pyrochlore curl, not an")
        print("      accident of high-symmetry points.")
    else:
        print("  ==> The degeneracy FAILS somewhere.  It is NOT a general")
        print("      property of the pyrochlore curl, and the Letter must")
        print("      restrict the claim to the classes where it holds.")

    print()
    print("=" * 74)
    print("Consequence: sqrt(2UK) multiplies both members identically, so no")
    print("choice of U or K can split them.  Any observed splitting is outside")
    print("Gaussian lattice electrodynamics entirely.")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
