"""
Compute the T=0 SzSz dynamical structure factor on the 16-site
(1,1,1) cubic pyrochlore cluster, both bare and at every {0,pi}^3 twist
corner, by re-using the existing twist run directories under output/demo/.

For each twist, S^{zz}(q, omega) is computed by the C++ ED `dssf
ground_state_dssf` continued-fraction kernel at the four cluster-allowed
momenta of the (1,1,1) cubic cluster:

    Q_Gamma = (0, 0, 0) * 2pi    (zero norm, trivially vanishes)
    Q_X1    = (1, 0, 0) * 2pi
    Q_X2    = (0, 1, 0) * 2pi
    Q_X3    = (0, 0, 1) * 2pi

The momentum points are passed in the binary's "multiples of pi"
convention as "0,0,0; 2,0,0; 0,2,0; 0,0,2".

Output: one HDF5 file per corner, layout
    dynamical/ground_state_dssf/SzSz_q_Qx<x>_Qy<y>_Qz<z>/
        frequencies, spectral_real, spectral_imag, error_*

Aggregation and plotting is done in analyze_dssf.py.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from itertools import product
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ED_BIN = ROOT.parent / "QED" / "build" / "ED"

OMEGA_MIN = -0.05
OMEGA_MAX = 1.5
OMEGA_POINTS = 600
BROADENING = 0.012  # ~ g_hex at Jpm=-0.1, Jzz=1
MOMENTUM_POINTS = "0,0,0;2,0,0;0,2,0;0,0,2"


def twist_label(phi):
    return "phi_" + "_".join(
        ("pi" if abs(p - np.pi) < 1e-8 else f"{p/np.pi:.3f}pi") for p in phi
    )


def run_one(phi, demo_root: Path, dssf_root: Path, force=False):
    tag = twist_label(phi)
    ham_dir = demo_root / tag / "ham"
    if not ham_dir.is_dir():
        raise FileNotFoundError(f"missing Hamiltonian directory {ham_dir}")

    out_dir = dssf_root / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    h5 = out_dir / "ed_results.h5"
    log = out_dir / "run.log"

    if h5.exists() and not force:
        return {"phi": list(map(float, phi)), "tag": tag, "h5": str(h5),
                "skipped": True}

    cmd = [
        str(ED_BIN), "dssf", "ground_state_dssf", str(ham_dir),
        "--num_sites=16",
        f"--output={out_dir}",
        '--dyn-spin-combinations=2,2',
        f'--dyn-momentum-points={MOMENTUM_POINTS}',
        f'--dyn-omega-min={OMEGA_MIN}',
        f'--dyn-omega-max={OMEGA_MAX}',
        f'--dyn-omega-points={OMEGA_POINTS}',
        f'--dyn-broadening={BROADENING}',
    ]

    t0 = time.time()
    log.write_text(f"# twist = {tuple(phi)}\n$ {' '.join(cmd)}\n")
    with open(log, "a") as f:
        rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT).returncode
    dt = time.time() - t0
    with open(log, "a") as f:
        f.write(f"# DSSF done in {dt:.1f} s (rc={rc})\n")
    if rc != 0:
        raise RuntimeError(f"DSSF failed for {tag} (rc={rc})  see {log}")

    return {"phi": list(map(float, phi)), "tag": tag, "h5": str(h5),
            "skipped": False, "elapsed_s": dt}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo-root", default=str(ROOT / "output" / "demo"),
                    help="path to existing twist runs (must contain phi_*/ham/)")
    ap.add_argument("--out", default=str(ROOT / "output" / "dssf"),
                    help="output root for DSSF runs")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    demo_root = Path(args.demo_root)
    dssf_root = Path(args.out)
    dssf_root.mkdir(parents=True, exist_ok=True)

    twists = list(product([0.0, np.pi], repeat=3))

    summary = {
        "n_sites": 16,
        "dim": [1, 1, 1],
        "omega_min": OMEGA_MIN,
        "omega_max": OMEGA_MAX,
        "omega_points": OMEGA_POINTS,
        "broadening": BROADENING,
        "momentum_points_pi": MOMENTUM_POINTS,
        "twists": [],
    }

    t0 = time.time()
    for k, phi in enumerate(twists):
        print(f"\n[{k+1}/{len(twists)}] twist = {tuple(phi)}", flush=True)
        rec = run_one(np.asarray(phi, dtype=float), demo_root, dssf_root,
                      force=args.force)
        summary["twists"].append(rec)
        with open(dssf_root / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        print(f"   -> {rec}", flush=True)

    print(f"\nTotal elapsed: {time.time() - t0:.1f} s")
    print(f"Summary: {dssf_root / 'summary.json'}")


if __name__ == "__main__":
    main()
