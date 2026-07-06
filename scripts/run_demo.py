"""
16-site pyrochlore QSI demo: bare vs twist-averaged ED at J_pm = -0.1.

Runs FTLM thermodynamics and Lanczos low-spectrum on 8 twist corners
phi in {0, pi}^3, plus the bare phi = (0,0,0) reference (already a corner),
and saves all eigenvalues + thermodynamic curves for downstream analysis
and plotting.

Defaults:
    Jxx = Jyy = 0.2,  Jzz = 1.0   ->   J_pm = -(Jxx+Jyy)/4 = -0.1
    DIM = (1,1,1)                 ->   16 sites
    twist grid:  {0, pi} per axis ->   8 corners
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
sys.path.insert(0, str(ROOT / "scripts"))
from twist_helper import write_pyrochlore_xxz_with_twist  # noqa: E402

JXX = 0.2
JYY = 0.2
JZZ = 1.0

NUM_SITES = 16
DIM = (1, 1, 1)

FTLM_SAMPLES = 20
FTLM_KRYLOV = 200
TEMP_MIN = 0.005
TEMP_MAX = 5.0
TEMP_BINS = 120

LANCZOS_NEIGS = 64


def twist_label(phi):
    return "phi_" + "_".join(
        ("pi" if abs(p - np.pi) < 1e-8 else f"{p/np.pi:.3f}pi") for p in phi
    )


def run_one(phi, run_root: Path, jxx=JXX, jyy=JYY, jzz=JZZ,
            ftlm_samples=FTLM_SAMPLES, ftlm_krylov=FTLM_KRYLOV,
            n_eigs=LANCZOS_NEIGS, force=False):
    tag = twist_label(phi)
    case_dir = run_root / tag
    ham_dir = case_dir / "ham"
    ftlm_dir = case_dir / "ftlm"
    spec_dir = case_dir / "spectrum"
    ftlm_dir.mkdir(parents=True, exist_ok=True)
    spec_dir.mkdir(parents=True, exist_ok=True)

    info = write_pyrochlore_xxz_with_twist(str(ham_dir), jxx, jyy, jzz, tuple(phi),
                                           dim1=DIM[0], dim2=DIM[1], dim3=DIM[2])

    log = case_dir / "run.log"
    log.write_text(f"# twist = {tuple(phi)}\n# {info}\n")

    ftlm_h5 = ftlm_dir / "ed_results.h5"
    if force or not ftlm_h5.exists():
        cmd = [str(ED_BIN), str(ham_dir),
               "--method=FTLM",
               f"--num_sites={NUM_SITES}",
               f"--output={ftlm_dir}",
               f"--samples={ftlm_samples}",
               f"--ftlm-krylov={ftlm_krylov}",
               f"--temp_min={TEMP_MIN}",
               f"--temp_max={TEMP_MAX}",
               f"--temp_bins={TEMP_BINS}"]
        t0 = time.time()
        with open(log, "a") as f:
            f.write(f"\n$ {' '.join(cmd)}\n")
            f.flush()
            rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT).returncode
        dt = time.time() - t0
        with open(log, "a") as f:
            f.write(f"# FTLM done in {dt:.1f} s (rc={rc})\n")
        if rc != 0:
            raise RuntimeError(f"FTLM failed for {tag} (rc={rc})  see {log}")

    spec_h5 = spec_dir / "ed_results.h5"
    if force or not spec_h5.exists():
        cmd = [str(ED_BIN), str(ham_dir),
               "--method=LANCZOS",
               f"--num_sites={NUM_SITES}",
               f"--output={spec_dir}",
               f"--eigenvalues={n_eigs}",
               "--iterations=2000",
               "--tolerance=1e-9"]
        t0 = time.time()
        with open(log, "a") as f:
            f.write(f"\n$ {' '.join(cmd)}\n")
            f.flush()
            rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT).returncode
        dt = time.time() - t0
        with open(log, "a") as f:
            f.write(f"# LANCZOS done in {dt:.1f} s (rc={rc})\n")
        if rc != 0:
            raise RuntimeError(f"LANCZOS failed for {tag} (rc={rc})  see {log}")

    return {"phi": list(map(float, phi)), "tag": tag, "ftlm_h5": str(ftlm_h5),
            "spec_h5": str(spec_h5), "info": {k: v for k, v in info.items() if k != "wrap_summary"}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "output" / "demo"))
    ap.add_argument("--Jxx", type=float, default=JXX,
                    help="default 0.2 (=> Jpm=-0.1, pi-flux side)")
    ap.add_argument("--Jyy", type=float, default=JYY)
    ap.add_argument("--Jzz", type=float, default=JZZ)
    ap.add_argument("--ftlm-samples", type=int, default=FTLM_SAMPLES)
    ap.add_argument("--ftlm-krylov", type=int, default=FTLM_KRYLOV)
    ap.add_argument("--n-eigs", type=int, default=LANCZOS_NEIGS)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    twists = list(product([0.0, np.pi], repeat=3))

    summary = {
        "Jxx": args.Jxx, "Jyy": args.Jyy, "Jzz": args.Jzz,
        "Jpm": -(args.Jxx + args.Jyy) / 4.0,
        "n_sites": NUM_SITES, "dim": list(DIM),
        "ftlm_samples": args.ftlm_samples, "ftlm_krylov": args.ftlm_krylov,
        "lanczos_neigs": args.n_eigs,
        "temp_grid": {"min": TEMP_MIN, "max": TEMP_MAX, "bins": TEMP_BINS},
        "twists": [],
    }

    t0 = time.time()
    for k, phi in enumerate(twists):
        print(f"\n[{k+1}/{len(twists)}] twist = {tuple(phi)}", flush=True)
        rec = run_one(np.asarray(phi, dtype=float), out_root,
                      jxx=args.Jxx, jyy=args.Jyy, jzz=args.Jzz,
                      ftlm_samples=args.ftlm_samples,
                      ftlm_krylov=args.ftlm_krylov,
                      n_eigs=args.n_eigs,
                      force=args.force)
        summary["twists"].append(rec)
        with open(out_root / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

    print(f"\nTotal elapsed: {time.time() - t0:.1f} s")
    print(f"Summary: {out_root / 'summary.json'}")


if __name__ == "__main__":
    main()
