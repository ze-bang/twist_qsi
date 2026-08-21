"""Does one parameter per coupling collapse all four loop perimeters?

Hypothesis.  Perturbation theory gives K_L^(0) = 2 C(L-2,n-1) |Jpm|^n / Jzz^(n-1)
with n = L/2, built from n-1 energy denominators.  If the whole nonperturbative
effect is a UNIVERSAL renormalization mu of each denominator, then

    K_L / K_L^(0) = mu^(n-1),        n = L/2,

with a single mu per coupling and no per-perimeter freedom.  If that holds, the
effective gauge theory at large Jpm is still the ring-exchange theory with the
same loop content -- only the virtual propagator is dressed.

Decisive follow-up encoded here: is 1/mu intensive or extensive?  Brillouin-
Wigner denominators are (z_0 - E_Q) with z_0 the EXTENSIVE ground energy, so a
BW size-inconsistency artifact would make 1/mu grow with the cluster.  A
physical renormalization would not.  cubic-16 (exact fold) and FCC-32 are
compared at matched coupling.
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TOMO = ROOT / "campaign" / "outputs" / "wp1_tomography"
NORM = ROOT / "campaign" / "outputs" / "map_normalization" / "normalization.json"

PERIMS = (6, 8, 10, 12)
PERT = {6: 12.0, 8: 40.0, 10: 140.0, 12: 504.0, 14: 1716.0, 16: 6435.0}


def k0(L, x):
    return PERT[L] * abs(x) ** (L // 2)


def load(depth, sampled):
    pat = "tomo_fcc32_L%d_*_n512.json" if sampled else "tomo_fcc32_L%d_*.json"
    rows = []
    for p in sorted(glob.glob(str(TOMO / (pat % depth)))):
        if not sampled and p.endswith("_n512.json"):
            continue
        d = json.load(open(p))
        rows.append((abs(float(d["jpm"])),
                     {L: d.get("K::%d" % L, {}).get("K") for L in PERIMS}))
    return sorted(rows)


def analyse(rows, label):
    print("\n" + "=" * 74)
    print(label)
    print("=" * 74)
    print("   x      " + "".join("mu(L=%-2d) " % L for L in PERIMS)
          + "  mu_fit   max dev")
    out = []
    for x, ks in rows:
        mus = {}
        for L in PERIMS:
            k = ks.get(L)
            if not k:
                continue
            n = L // 2
            mus[L] = (k / k0(L, x)) ** (1.0 / (n - 1))
        if len(mus) < 2:
            continue
        # single-parameter fit: minimise squared error in log K
        lognum = sum((L // 2 - 1) * np.log(ks[L] / k0(L, x)) for L in mus)
        logden = sum((L // 2 - 1) ** 2 for L in mus)
        mu = float(np.exp(lognum / logden))
        dev = max(abs(ks[L] / (k0(L, x) * mu ** (L // 2 - 1)) - 1.0)
                  for L in mus)
        print("  %.3f   " % x
              + "".join("%7.4f  " % mus[L] for L in PERIMS)
              + " %7.4f  %6.1f%%" % (mu, 100 * dev))
        out.append((x, mu))
    return out


def main():
    d2 = load(2, False)
    d3 = load(3, True)
    fit2 = analyse(d2, "FCC-32, virtual depth 2   [mu_L = (K_L/K_L^(0))^(1/(n-1))]")
    analyse(d3, "FCC-32, virtual depth 3 (512-column sample)")

    print("\n" + "=" * 74)
    print("Is 1/mu intensive (physical) or extensive (BW size artifact)?")
    print("=" * 74)
    cub = {}
    for r in json.load(open(NORM)):
        if r["jpm"] > 0:
            continue
        x = abs(r["jpm"])
        s_bw = r["k6_bw_over_g6"] if r["k6_bw_over_g6"] > 0 else -r["k6_bw_over_g6"]
        s_h = r["k6_herm_over_g6"]
        cub[round(x, 4)] = (abs(s_bw), abs(s_h) if s_h else None)
    print("  cubic-16 uses L=6 only (mu = sqrt(K_6/K_6^(0))): it is the only")
    print("  perimeter that survives masking on 16 sites.")
    print("\n    x      mu cubic-16 (BW)  mu cubic-16 (Herm)  mu FCC-32 (d2)  ratio")
    lookup = dict(fit2)
    for x in sorted(cub):
        bw, h = cub[x]
        mu_c = np.sqrt(bw)
        mu_ch = np.sqrt(h) if h else None
        mu_f = lookup.get(round(x, 3))
        if mu_f is None:
            near = [k for k in lookup if abs(k - x) < 0.011]
            mu_f = lookup[near[0]] if near else None
        print("  %.3f      %8.4f          %s          %s      %s"
              % (x, mu_c,
                 "%8.4f" % mu_ch if mu_ch else "    --  ",
                 "%8.4f" % mu_f if mu_f else "    --  ",
                 "%.3f" % (mu_f / mu_c) if mu_f else " -- "))
    print("\n  If the suppression were the BW extensive-z_0 artifact, doubling the")
    print("  cluster (16 -> 32 sites) would move mu substantially.  Read the last")
    print("  column: values near 1.0 mean the renormalization is INTENSIVE.")

    print("\n" + "=" * 74)
    print("Prediction from the collapse: unmeasured perimeters at FCC-32 depth 2")
    print("=" * 74)
    print("    x       K_14 predicted   K_16 predicted   K_14/K_6   K_16/K_6")
    for x, mu in fit2:
        k6 = k0(6, x) * mu ** 2
        p14 = k0(14, x) * mu ** 6
        p16 = k0(16, x) * mu ** 7
        print("  %.3f      %.4e       %.4e     %.5f    %.5f"
              % (x, p14, p16, p14 / k6, p16 / k6))

    print("\n" + "=" * 74)
    print("Why panel (c) plateaus: K_(L+2)/K_L = (4 - 4/L) * x * mu(x)")
    print("=" * 74)
    print("    x       mu      x*mu     (4-4/8) x mu = K_10/K_8 predicted")
    for x, mu in fit2:
        print("  %.3f   %6.4f   %6.4f    %6.4f" % (x, mu, x * mu, 3.5 * x * mu))
    print("\n  x*mu saturating is exactly why the K_L/K_6 curves flatten:")
    print("  the perturbative ratio grows like x, mu falls like ~1/x.")


if __name__ == "__main__":
    main()
