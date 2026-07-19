#!/usr/bin/env python3
"""Campaign driver for the π-flux QSI gauge-sector ED study (Phases A/B).

Runs bare vs. δ=0-projected ice-manifold thermodynamics over a coupling sweep
and writes a JSON per phase.  Reuses the verified from-scratch module in
notes/recompute_finite_size_artifact.py so results match the notes and paper.

Phase C (dynamical S^zz) and Phase D (spinon FTLM) are separate drivers; see
SIMULATION_PLAN.md, Tasks T2/T3.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

# make the verified recompute module importable
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "notes"))
import recompute_finite_size_artifact as R  # noqa: E402


def entropy(E, T):
    E = np.asarray(E, float)
    E = E - E.min()
    w = np.exp(-(1.0 / T[:, None]) * E[None, :])
    Z = w.sum(1)
    return np.log(Z) + (w * E[None, :]).sum(1) / Z / T


PHASES = {
    # phase: list of Jpm/Jzz values
    "A": [0.03, 0.04, 0.046, 0.05, 0.06, 0.08, 0.10],          # 0-flux benchmark
    "B": [-0.03, -0.04, -0.05, -0.06, -0.08, -0.10],           # pi-flux delivery
}


def run(phase: str, clusters, outdir: Path, nt: int):
    T = np.geomspace(1e-4, 0.3, nt)
    jpms = PHASES[phase]
    results = {"phase": phase, "T_grid": [float(T[0]), float(T[-1]), nt], "cases": []}
    for basis, shape in clusters:
        t0 = time.time()
        cl = R.build_cluster(basis, shape)
        pt = R.sw_order23(cl, verbose=False)
        print(f"[{phase}] built {basis}{shape} ice={cl.n_ice} "
              f"in {time.time()-t0:.1f}s", flush=True)
        for jpm in jpms:
            g4 = 4 * jpm * jpm
            ghex = 12 * abs(jpm) ** 3
            row = {"basis": basis, "shape": list(shape), "Jpm": jpm,
                   "g4": g4, "ghex": ghex}
            for mode in ("all", "delta0"):
                E = np.linalg.eigvalsh(R.assemble(cl, pt, jpm, mode))
                C = R.specific_heat(E, T)
                S = entropy(E, T)
                Tpk = R.refined_peak(T, C)
                row[mode] = {
                    "Tpeak": Tpk,
                    "Tpeak_over_ghex": Tpk / ghex,
                    "Tpeak_over_g4": Tpk / g4,
                    "gap": float(np.sort(E)[1] - np.sort(E)[0]),
                    "S_at_ghex": float(np.interp(ghex, T, S)),
                }
            results["cases"].append(row)
            print(f"  Jpm={jpm:+.3f}: bare Tpk/ghex={row['all']['Tpeak_over_ghex']:.2f}"
                  f"  clean Tpk/ghex={row['delta0']['Tpeak_over_ghex']:.2f}", flush=True)
    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / f"campaign_phase{phase}.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"[{phase}] wrote {out}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="B", choices=sorted(PHASES))
    ap.add_argument("--outdir", type=Path, default=ROOT / "output")
    ap.add_argument("--nt", type=int, default=1200)
    ap.add_argument("--big", action="store_true",
                    help="include heavier clusters (none beyond FCC-32 here)")
    args = ap.parse_args()
    clusters = [("cubic", (1, 1, 1)), ("fcc", (2, 2, 2))]
    run(args.phase, clusters, args.outdir / f"phase{args.phase}", args.nt)


if __name__ == "__main__":
    main()
