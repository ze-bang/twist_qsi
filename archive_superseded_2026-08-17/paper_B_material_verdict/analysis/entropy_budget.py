"""Pauling-entropy budget for the Ce2Hf2O7 two-peak heat capacity.

Evaluates  dS_low = int_0^{T_dip} C_mag/T dT  from the digitized trace of
Smith et al., PRL 135, 086702 (2025), and compares it with the residual ice
entropy (R/2) ln(3/2) = 1.6871 J/(mol_Ce K).

The digitized data starts at T_min = 20.6 mK, ABOVE the low-temperature flank
of the peak, so the integral is reported in three parts:

    measured   : trapezoid in log T from T_min to the inter-peak minimum
    tail_T3    : photon-like extrapolation C = C(T_min) (T/T_min)^3 below T_min
                 -> contributes C(T_min)/3
    tail_lin   : disorder-like extrapolation C = C(T_min) (T/T_min)
                 -> contributes C(T_min)

The spread between the two tail models is the systematic that the missing
sub-20 mK data leaves behind; report it, do not hide it.

Run through the scheduler, never on a login node:
    sbatch analysis/run_entropy.sbatch
"""

import csv
import math
import os

R_GAS = 8.31446261815324               # J / (mol K)
S_ICE = 0.5 * R_GAS * math.log(1.5)    # 1.6871 J / (mol_Ce K)
S_FULL = R_GAS * math.log(2.0)         # 5.7632

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.normpath(os.path.join(
    HERE, os.pardir, os.pardir,
    "campaign", "data", "ce2hf2o7_smith2025_digitized.csv"))


def load(series="experiment"):
    """Return sorted [(T_K, C_mag)] for one series, skipping the # header."""
    rows = []
    with open(CSV) as fh:
        lines = [ln for ln in fh if not ln.startswith("#")]
    reader = csv.DictReader(lines)
    for row in reader:
        if row.get("series") != series:
            continue
        try:
            t = float(row["temperature_K"])
            c = float(row["Cmag_J_molCe_K"])
        except (TypeError, ValueError):
            continue
        if t > 0 and math.isfinite(c):
            rows.append((t, c))
    rows.sort()
    return rows


def find_dip(rows, t_lo=0.030, t_hi=0.060):
    """Inter-peak minimum of C, searched between the two published peaks."""
    window = [(t, c) for t, c in rows if t_lo <= t <= t_hi]
    if not window:
        raise SystemExit("no data in the inter-peak window")
    return min(window, key=lambda tc: tc[1])


def integrate_c_over_t(rows, t_max):
    """int C/T dT by trapezoid in log T, i.e. int C dlnT."""
    pts = [(t, c) for t, c in rows if t <= t_max]
    total = 0.0
    for (t0, c0), (t1, c1) in zip(pts, pts[1:]):
        total += 0.5 * (c0 + c1) * (math.log(t1) - math.log(t0))
    return total, pts


def main():
    rows = load("experiment")
    if not rows:
        raise SystemExit("no experimental rows parsed from " + CSV)
    t_min, c_at_tmin = rows[0]
    t_dip, c_dip = find_dip(rows)
    measured, pts = integrate_c_over_t(rows, t_dip)

    tail_t3 = c_at_tmin / 3.0      # int_0^Tmin (C/T)(T/Tmin)^3 dT
    tail_lin = c_at_tmin           # int_0^Tmin (C/T)(T/Tmin)   dT

    print("source              : " + CSV)
    print("data range          : %.2f mK .. %.3f K  (%d points)"
          % (t_min * 1e3, rows[-1][0], len(rows)))
    print("C at first point    : %.4f J/(mol_Ce K)" % c_at_tmin)
    print("inter-peak minimum  : T_dip = %.2f mK, C = %.4f"
          % (t_dip * 1e3, c_dip))
    print("points integrated   : %d" % len(pts))
    print()
    print("S_ice = (R/2)ln(3/2) = %.4f J/(mol_Ce K)  = %.1f%% of R ln2"
          % (S_ICE, 100 * S_ICE / S_FULL))
    print()
    print("measured  [%.1f mK -> dip]        = %.4f   (%.1f%% of S_ice)"
          % (t_min * 1e3, measured, 100 * measured / S_ICE))
    print("+ tail T^3   (photon-like)       = %.4f   -> total %.4f  (%.1f%%)"
          % (tail_t3, measured + tail_t3,
             100 * (measured + tail_t3) / S_ICE))
    print("+ tail linear (disorder-like)    = %.4f   -> total %.4f  (%.1f%%)"
          % (tail_lin, measured + tail_lin,
             100 * (measured + tail_lin) / S_ICE))
    print()
    print("VERDICT GUIDE")
    print("  ~S_ice under BOTH tails  -> low peak IS the ice-manifold crossover;")
    print("                              the gauge reading survives this test.")
    print("  << S_ice under both      -> only a fraction of the moments take part.")
    print("  >> S_ice                 -> the feature is not confined to the ice")
    print("                              manifold.")
    print("  tails straddle S_ice     -> INCONCLUSIVE; needs author data below")
    print("                              20 mK.  Report it that way; do not pick")
    print("                              a tail model.")


if __name__ == "__main__":
    main()
