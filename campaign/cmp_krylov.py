"""Krylov convergence at 96 sites: m = 120 against m = 200.

The referee asks that the large-cluster continued fractions be shown
converged rather than asserted.  The two independent runs differ only
in Krylov dimension, so every pole energy, residue, and flippability
census can be compared directly.
"""
import json
from pathlib import Path

D = Path(__file__).resolve().parent / "outputs" / "ring_model_dssf"
a = json.load(open(D / "ring96_fcc234_n120.json"))
b = json.load(open(D / "ring96_fcc234.json"))
print(f"E0        : {a['E0']:.10f}  vs {b['E0']:.10f}   "
      f"|diff| {abs(a['E0']-b['E0']):.2e}")
print(f"<n_flip>  : {a['nflip_ground']:.8f}  vs {b['nflip_ground']:.8f}")
worst_w = worst_z = 0.0
for ma, mb in zip(a["momenta"], b["momenta"]):
    print(f"\nq = {[round(x,3) for x in ma['q']]}    "
          f"S: {ma['S']:.8f} vs {mb['S']:.8f}  "
          f"|diff| {abs(ma['S']-mb['S']):.2e}")
    print("      m=120: omega      Z        census |     m=200: omega      Z"
          "        census |   |dw|     |dZ|")
    for la, lb in zip(ma["lines"][:6], mb["lines"][:6]):
        dw, dz = abs(la[0]-lb[0]), abs(la[1]-lb[1])
        worst_w, worst_z = max(worst_w, dw), max(worst_z, dz)
        print(f"      {la[0]:10.6f} {la[1]:9.6f} {la[2]:+8.5f} | "
              f"     {lb[0]:10.6f} {lb[1]:9.6f} {lb[2]:+8.5f} | "
              f"{dw:.1e}  {dz:.1e}")
print(f"\nlargest pole-energy difference over all reported lines: {worst_w:.2e} K")
print(f"largest residue    difference over all reported lines: {worst_z:.2e}")
