"""Which pyrochlore cluster is the next usable one after 64 sites?

Two requirements, both hard:
  (1) the cluster must CONTAIN an L-type momentum (otherwise it cannot
      test the roton at all);
  (2) the ground transport sector must be diagonalizable.

This scout answers (1) exactly and cheaply -- momenta follow from the
supercell alone, no ice enumeration -- and estimates (2) from the
measured growth of the ice manifold (48 -> 64 sites), so we learn
whether 96 sites is reachable before spending a single core-hour on it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "notes"))
sys.path.insert(0, str(ROOT / "campaign"))

import recompute_finite_size_artifact as geometry               # noqa: E402

# measured on our own clusters
REF = {48: (87_546, 9_104), 64: (2_756_394, 249_180), 72: (12_846_186, 973_008)}
RATE = np.log(REF[72][0] / REF[48][0]) / 24.0
FRAC = REF[72][1] / REF[72][0]

SHAPES = [(2, 2, 3), (2, 2, 4), (2, 3, 3), (2, 2, 5), (1, 3, 5), (3, 3, 3),
          (2, 2, 6), (2, 3, 4), (3, 3, 4), (2, 4, 4), (2, 3, 5), (3, 3, 5)]


def momenta_of(Lvecs, shape):
    """Allowed momenta of the supercell, in cubic reciprocal units."""
    n1, n2, n3 = shape
    M = np.array([[i, j, k] for i in range(n1) for j in range(n2)
                  for k in range(n3)], dtype=float)
    return M @ np.linalg.inv(Lvecs).T


def has_L(qs):
    """Is any allowed momentum an L point, i.e. equivalent to
    (1/2,1/2,1/2) in cubic reciprocal units up to a reciprocal vector?"""
    best = None
    for q in qs:
        r = q - np.round(q * 2) / 2.0          # fold onto the half-grid
        f = np.abs(np.round(q * 2) / 2.0)
        f = np.sort(f % 1.0)
        if np.allclose(np.abs(r), 0, atol=1e-9) \
                and np.allclose(np.sort(np.abs(q % 1.0)), [0.5, 0.5, 0.5],
                                atol=1e-6):
            best = q
            break
    return best


def main() -> int:
    print(f"ice-manifold growth {RATE:.4f} /site, "
          f"sector fraction {FRAC:.3f}  (calibrated 48 -> 72)\n")
    print("shape      sites   L point?   est. ice states   est. sector   "
          "vector (complex128)")
    for shape in SHAPES:
        try:
            positions, Lvecs = geometry.make_positions("fcc", shape)
        except Exception as exc:                       # pragma: no cover
            print(f"  {shape}  build failed: {exc}")
            continue
        nsite = len(positions)
        qs = momenta_of(np.asarray(Lvecs, dtype=float), shape)
        L = has_L(qs)
        ice = REF[48][0] * np.exp(RATE * (nsite - 48))
        sec = ice * FRAC
        gb = sec * 16 / 2**30
        flag = "yes" if L is not None else "NO"
        print(f"  {str(shape):<9s} {nsite:4d}    {flag:<4s}    "
              f"{ice:14.3e}   {sec:11.3e}   {gb:8.2f} GB")
    print("\nreference: our 64-site run diagonalized a 249,180-dim sector "
          "by Lanczos.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
