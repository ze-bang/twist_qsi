"""Aggregate the L-point response across sublattice channels.

The continued-fraction pipeline seeds separately with each of the four
sublattice-resolved probes S^z_mu(q)|0>, so one physical level appears
as up to four separate lines at the same energy.  Only the SUM over
channels is basis-independent, so the lines must be grouped by energy
before branch energies and weights are quoted.  Comparison across
cluster sizes then uses the same quantities at 48, 64 and 72 sites.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "campaign" / "outputs" / "ring_model_dssf"
TOL = 1e-6


def group(lines):
    out = []
    for e, w in sorted(lines):
        if out and abs(out[-1][0] - e) < TOL:
            out[-1][1] += w
        else:
            out.append([e, w])
    return out


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "233"
    p = OUT / f"ring128_fcc{tag}.json"
    if not p.exists():
        raise SystemExit(f"missing {p}")
    d = json.load(open(p))
    print(f"fcc{tag}: {d['n_sites']} sites, {d['n_ice']:,} ice states, "
          f"sector {d['sector']} dim {d['sector_dim']:,}")
    print(f"E0 = {d['E0']:.8f} K   E0/site = {d['E0_per_site']:.8f}")
    print(f"<n_flip> = {d['nflip_ground']:.4f} of {d['n_hex']} hexagons "
          f"= {d['nflip_per_hex']:.4f} per hexagon")
    for m in d["momenta"]:
        g = group([list(x) for x in m["lines"]])
        g.sort(key=lambda r: -r[1])
        S = m["S"]
        print(f"\n  q = {[round(x,3) for x in m['q']]}   S(L) = {S:.5f}")
        print("     omega/K     weight     share")
        for e, w in sorted(g[:6]):
            print(f"    {e:8.4f}   {w:.6f}   {w/S:.4f}")
        two = sorted(g[:2])
        if len(two) == 2:
            split = abs(two[0][0] - two[1][0]) / (0.5 * (two[0][0]
                                                         + two[1][0]))
            print(f"    two brightest: {two[0][0]:.4f} "
                  f"(share {two[0][1]/S:.4f}) and {two[1][0]:.4f} "
                  f"(share {two[1][1]/S:.4f});  splitting {split:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
