"""Polarization resolution and off-pole weight at EVERY cluster momentum.

The referee objects, correctly, that calling an exact level "one photon"
because its energy falls within an arbitrary window of a fitted Gaussian
line is not a definition.  Here we replace that criterion with one that
uses no energy window at all.

At each momentum the electric field lives in a four-dimensional
sublattice space; the lattice curl has exactly two nonzero singular
values there, so the neutron probe S^z(q)|0> is confined to a
two-dimensional TRANSVERSE space (the two remaining directions are the
kernel, and the weight found there is the longitudinal residue, which
must vanish because div E = 0 -- an exact gate, not a fit).

Every bright level is therefore resolved into (pol-1, pol-2, kernel).
We then define the photon descendants operationally as the brightest
level in each of the two polarization channels, and the off-pole weight
as everything else.  No fitted scale and no tolerance enter.  For
comparison we also report the "two brightest levels" variant.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "notes"))
sys.path.insert(0, str(ROOT / "campaign"))

import recompute_finite_size_artifact as geometry               # noqa: E402
from run_momentum_resolved_spectrum import star                 # noqa: E402
from run_ring48_dssf import contractible                        # noqa: E402
from ring48_momenta_fix import fcc_momenta                      # noqa: E402
from ring48_channels import unwrap_hexagons, analytic_blocks    # noqa: E402
from ring64_pipeline import (fast_labels, fast_ring,            # noqa: E402
                             hex_masks_patterns)

OUT = ROOT / "campaign" / "outputs" / "ring_model_dssf"
EDEGEN = 1e-8
WMIN = 1e-4


def main() -> int:
    t0 = time.perf_counter()
    cl = geometry.build_cluster("fcc", (2, 2, 3))
    n = int(cl.n_sites)
    states = np.sort(np.asarray(cl.ice_states, dtype=np.uint64))
    positions = np.asarray(cl.positions, dtype=float)
    labels, spins_all = fast_labels(states, positions)
    hmp = hex_masks_patterns(contractible(cl))
    ham, _ = fast_ring(states, hmp, 0.0)
    _, v0 = eigsh(ham, k=1, which="SA", tol=1e-11, ncv=80)
    occ = {int(s): float(np.sum(v0[labels == s, 0] ** 2))
           for s in np.unique(labels)}
    sector = max(occ, key=occ.get)
    rows = np.where(labels == sector)[0]
    values, vectors = eigh(np.asarray(ham[np.ix_(rows, rows)].todense()))
    omega = values - values[0]
    gs = vectors[:, 0].astype(complex)
    spins = spins_all[rows]
    print(f"sector {sector} dim {len(rows)}, E0 = {values[0]:.6f} K "
          f"({time.perf_counter()-t0:.0f} s)", flush=True)

    block = analytic_blocks(cl, unwrap_hexagons(cl, contractible(cl)))
    momenta, _ = fcc_momenta(cl)
    sub = np.arange(n) % 4

    groups, start = [], 0
    for i in range(1, len(omega) + 1):
        if i == len(omega) or omega[i] - omega[start] > EDEGEN:
            groups.append((float(omega[start]), slice(start, i)))
            start = i
    print(f"{len(groups)} degenerate level groups")

    rec, offpole_a, offpole_b = {}, [], []
    for qi, momentum in enumerate(momenta):
        lab = star(momentum)
        phase = np.exp(-2j * np.pi * (positions @ momentum))
        A = np.zeros((len(rows), 4), dtype=complex)
        for mu in range(4):
            A[:, mu] = vectors.conj().T @ ((spins @ (phase * (sub == mu)))
                                           * gs)
        A /= np.sqrt(n)
        w_tot = np.sum(np.abs(A) ** 2, axis=1)
        S_q = float(w_tot.sum())
        if S_q < 1e-12:
            print(f"\n{lab}: dark (S = {S_q:.2e}) -- Gamma, skipped")
            continue

        u, s, vh = np.linalg.svd(block(-momentum), full_matrices=True)
        ntrans = int((s > 1e-10).sum())
        w_dir = np.abs(A @ vh.conj().T) ** 2
        resolve = float(np.max(np.abs(w_dir.sum(axis=1) - w_tot)))
        print(f"\n{lab}  (q index {qi})  S(q) = {S_q:.5f}   "
              f"{ntrans} transverse branches   sigma = "
              + ", ".join(f"{x:.4f}" for x in s)
              + f"   completeness residue {resolve:.2e}", flush=True)
        if resolve > 1e-10 or ntrans != 2:
            print("   *** gate failed -- skipped ***")
            continue

        lines = []
        for e, sl in groups:
            wt = float(w_tot[sl].sum())
            if wt / S_q < WMIN:
                continue
            d = w_dir[sl].sum(axis=0)
            lines.append({"omega": e, "w": wt,
                          "pol": [float(d[0]), float(d[1])],
                          "kernel": float(d[2] + d[3])})
        lines.sort(key=lambda r: r["omega"])
        kmax = max(abs(r["kernel"]) for r in lines) / S_q

        # (a) brightest level in each polarization channel
        i1 = max(range(len(lines)), key=lambda i: lines[i]["pol"][0])
        i2 = max(range(len(lines)), key=lambda i: lines[i]["pol"][1])
        if i1 == i2:
            one_a = lines[i1]["w"]
        else:
            one_a = lines[i1]["w"] + lines[i2]["w"]
        # (b) two brightest levels overall
        order = sorted(range(len(lines)), key=lambda i: -lines[i]["w"])
        one_b = sum(lines[i]["w"] for i in order[:2])

        oa, ob = 1 - one_a / S_q, 1 - one_b / S_q
        offpole_a.append(oa)
        offpole_b.append(ob)
        print("    omega      share    pol-1   pol-2   long/S")
        for r in lines[:8]:
            tot = r["w"]
            print(f"   {r['omega']:7.4f}   {tot/S_q:.4f}   "
                  f"{r['pol'][0]/tot:.3f}   {r['pol'][1]/tot:.3f}   "
                  f"{r['kernel']/S_q:.1e}")
        print(f"    photon descendants: levels at "
              f"{lines[i1]['omega']:.4f} (pol-1) and "
              f"{lines[i2]['omega']:.4f} (pol-2)")
        print(f"    off-pole weight: {oa:.4f} (per-channel)   "
              f"{ob:.4f} (two brightest)   max longitudinal {kmax:.1e}")
        rec[f"{lab}_{qi}"] = {"star": lab, "q": [float(x) for x in momentum],
                              "S": S_q, "sigma": [float(x) for x in s],
                              "lines": lines,
                              "photon_pol1": lines[i1]["omega"],
                              "photon_pol2": lines[i2]["omega"],
                              "offpole_per_channel": oa,
                              "offpole_two_brightest": ob,
                              "max_longitudinal": kmax}

    print(f"\n=== off-pole weight across momenta ===")
    print(f"  per-channel definition : [{min(offpole_a):.3f}, "
          f"{max(offpole_a):.3f}]  = {100*min(offpole_a):.0f}-"
          f"{100*max(offpole_a):.0f}%")
    print(f"  two-brightest definition: [{min(offpole_b):.3f}, "
          f"{max(offpole_b):.3f}]  = {100*min(offpole_b):.0f}-"
          f"{100*max(offpole_b):.0f}%")
    rec["_summary"] = {
        "offpole_per_channel": [float(min(offpole_a)), float(max(offpole_a))],
        "offpole_two_brightest": [float(min(offpole_b)),
                                  float(max(offpole_b))]}
    with open(OUT / "ring48_polarization_all.json", "w") as fh:
        json.dump(rec, fh, indent=1, default=float)
    print(f"\nwrote ring48_polarization_all.json "
          f"({time.perf_counter()-t0:.0f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
