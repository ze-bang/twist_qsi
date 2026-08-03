#!/usr/bin/env python3
"""Consolidated summary of the large-|J_pm| campaign (one row per coupling)."""

from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "campaign" / "outputs"


def load(pattern):
    table = {}
    for path in sorted(OUT.glob(pattern)):
        with open(path) as handle:
            record = json.load(handle)
        table[round(record["jpm"], 4)] = record
    return table


anatomy = load("feshbach_anatomy/anatomy_*.json")
raw = load("strong_coupling/gs_*.json")
coherence = load("corner_coherence/coherence_*.json")
decomposition = load("corner_decomposition/decomposition_*.json")
classct = load("counterterm_observables/ctobs_*.json")
residual = load("winding_residual/residual_*.json")
exact = load("exact_counterterm/exact_ct_*.json")
trunc16 = load("truncated_feshbach/trunc_cubic16_L1_*.json")
trunc32 = load("truncated_feshbach/trunc_fcc32_L1_*.json")
band16 = load("fcc32_band/band_cubic16_L1_*.json")
band32 = load("fcc32_band/band_fcc32_L1_*.json")

couplings = sorted(set(anatomy) | set(raw) | set(exact), reverse=True)

header = (f"{'jpm':>7} | {'g/g6':>6} {'V/g':>7} {'long/hex':>8} {'Z':>5} "
          f"{'pk/g6(F)':>8} | {'pk/g6(H+C)':>10} {'flux(H+C)':>9} "
          f"{'dE/width':>9} | {'flux(raw)':>9} {'wind4(raw)':>10} "
          f"{'ice(raw)':>8} | {'rho(bare)':>9} {'rho(class)':>10}")
print(header)
print("-" * len(header))
for jpm in couplings:
    if jpm > 0:
        continue
    a = anatomy.get(jpm, {})
    r = raw.get(jpm, {})
    e = exact.get(jpm, {})
    g = residual.get(jpm, {})

    def fmt(record, key, spec=".3f"):
        value = record.get(key)
        return format(value, spec) if isinstance(value, (int, float)) \
            else "-"

    print(f"{jpm:>7.3f} | "
          f"{abs(a.get('g_eff', float('nan')))/(12*abs(jpm)**3):>6.3f} "
          f"{fmt(a, 'rk_ratio'):>7} {fmt(a, 'long_over_hex'):>8} "
          f"{fmt(a, 'z_ground'):>5} {fmt(a, 'ring_peak_over_g6'):>8} | "
          f"{fmt(e, 'ring_peak_over_g6'):>10} {fmt(e, 'hexagon_flux'):>9} "
          f"{fmt(e, 'root_agreement_relative', '.1e'):>9} | "
          f"{fmt(r, 'hexagon_flux'):>9} {fmt(r, 'winding4_amplitude'):>10} "
          f"{fmt(r, 'ice_weight'):>8} | "
          f"{fmt(g, 'winding_fraction_bare', '.4f'):>9} "
          f"{fmt(g, 'winding_fraction_corrected', '.4f'):>10}")

print()
print("size comparison (one-pair truncation) and winding-free bands")
print(f"{'jpm':>7} | {'g/g6 c16':>8} {'g/g6 f32':>8} {'c4/4J2 c16':>10} "
      f"{'c4/4J2 f32':>10} | {'peak/g6 c16':>11} {'peak/g6 f32':>11} "
      f"{'flux c16':>8} {'flux f32':>8}")
for jpm in sorted(set(trunc16) | set(band32), reverse=True):
    a, b = trunc16.get(jpm, {}), trunc32.get(jpm, {})
    c, d = band16.get(jpm, {}), band32.get(jpm, {})

    def f(record, key, spec=".3f"):
        value = record.get(key)
        return format(abs(value), spec) if isinstance(value, (int, float)) \
            else "-"

    print(f"{jpm:>7.3f} | {f(a, 'g_over_g6'):>8} {f(b, 'g_over_g6'):>8} "
          f"{f(a, 'wind_over_bare'):>10} {f(b, 'wind_over_bare'):>10} | "
          f"{f(c, 'ring_peak_over_g6'):>11} {f(d, 'ring_peak_over_g6'):>11} "
          f"{f(c, 'gs_flux'):>8} {f(d, 'gs_flux'):>8}")

print()
print("corner coherence and its decomposition")
print(f"{'jpm':>7} | {'minK':>6} {'return':>7} {'ice(band)':>9} "
      f"{'sigma':>6} {'rank>0.8':>8} | {'ret raw':>8} {'ret H+C':>8} "
      f"{'vec raw':>8} {'vec H+C':>8}")
for jpm in sorted(set(coherence) | set(decomposition), reverse=True):
    c = coherence.get(jpm, {})
    d = decomposition.get(jpm, {})

    def f(record, key, spec=".4f"):
        value = record.get(key)
        return format(value, spec) if isinstance(value, (int, float)) else "-"

    print(f"{jpm:>7.3f} | {f(c, 'band_coherence_min'):>6} "
          f"{f(c, 'manifold_return_mean'):>7} {f(c, 'band_ice_weight'):>9} "
          f"{f(c, 'corner_sigma_min'):>6} "
          f"{c.get('coherence_rank_0p8', '-'):>8} | "
          f"{f(d, 'raw_manifold_return'):>8} {f(d, 'wf_manifold_return'):>8} "
          f"{f(d, 'raw_vector_return'):>8} {f(d, 'wf_vector_return'):>8}")
