"""
Small-transverse-field sweep on the 16-site pyrochlore (1x1x1) cluster.

Adds a uniform local field h_perp * S^x_i (via twist_helper's h_perp option)
to the twisted XXZ Hamiltonian, at both flux signs (Jpm=-0.1 "pi-flux" and
Jpm=+0.05 "0-flux"), across the eight-corner {0,pi}^3 twist grid, and tracks
the low-energy Lanczos spectrum as a function of h_perp.

Goal: extract the field-linear/quadratic response of the twist-averaged
ground-state energy and of the gap to the first excited manifold, and see
whether the slope differs between the two flux sectors (same cluster,
opposite sign of Jpm -- NOT a different real-space geometry).

Usage:
    python3 sweep_transverse_field.py --out ../output/hfield_sweep
"""
from __future__ import annotations

import argparse
import json
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

NUM_SITES = 16
DIM = (1, 1, 1)
N_EIGS = 24
ITERATIONS = 2000
TOLERANCE = 1e-9

# (label, Jxx=Jyy, Jzz) -- Jpm = -(Jxx+Jyy)/4
FLUX_CASES = [
    ("piflux", 0.2, 1.0),   # Jpm = -0.1
    ("zeroflux", -0.1, 1.0),  # Jpm = +0.05
]

H_VALUES = [0.0, 0.005, 0.01, 0.02, 0.03, 0.04, -0.005, -0.01, -0.02, -0.03, -0.04]


def h_label(h):
    sign = "m" if h < 0 else "p"
    return f"h{sign}{abs(h):.3f}"


def twist_label(phi):
    return "phi_" + "_".join(
        ("pi" if abs(p - np.pi) < 1e-8 else f"{p/np.pi:.3f}pi") for p in phi
    )


def run_one(flux_label, jxy, jzz, h_perp, phi, out_root: Path, force=False):
    tag = f"{flux_label}/{h_label(h_perp)}/{twist_label(phi)}"
    case_dir = out_root / flux_label / h_label(h_perp) / twist_label(phi)
    ham_dir = case_dir / "ham"
    spec_dir = case_dir / "spectrum"
    spec_dir.mkdir(parents=True, exist_ok=True)

    info = write_pyrochlore_xxz_with_twist(
        str(ham_dir), jxy, jxy, jzz, tuple(phi),
        dim1=DIM[0], dim2=DIM[1], dim3=DIM[2], h_perp=h_perp,
    )

    spec_h5 = spec_dir / "ed_results.h5"
    log = case_dir / "run.log"
    if force or not spec_h5.exists():
        cmd = [str(ED_BIN), str(ham_dir),
               "--method=LANCZOS",
               f"--num_sites={NUM_SITES}",
               f"--output={spec_dir}",
               f"--eigenvalues={N_EIGS}",
               f"--iterations={ITERATIONS}",
               f"--tolerance={TOLERANCE}"]
        t0 = time.time()
        with open(log, "w") as f:
            f.write(f"# {tag}\n# {info}\n$ {' '.join(cmd)}\n")
            f.flush()
            rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT).returncode
        dt = time.time() - t0
        with open(log, "a") as f:
            f.write(f"\n# done in {dt:.1f}s rc={rc}\n")
        if rc != 0:
            raise RuntimeError(f"LANCZOS failed for {tag} (rc={rc}), see {log}")
    return {"flux": flux_label, "jxy": jxy, "jzz": jzz, "h_perp": h_perp,
            "phi": list(map(float, phi)), "spec_h5": str(spec_h5)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "output" / "hfield_sweep"))
    ap.add_argument("--corners", choices=["bare", "eight"], default="bare",
                    help="'bare' = phi=(0,0,0) only (fast); 'eight' = full {0,pi}^3 grid")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    if args.corners == "bare":
        corners = [(0.0, 0.0, 0.0)]
    else:
        pi = float(np.pi)
        corners = list(product([0.0, pi], repeat=3))

    rows = []
    n_total = len(FLUX_CASES) * len(H_VALUES) * len(corners)
    k = 0
    for flux_label, jxy, jzz in FLUX_CASES:
        for h in H_VALUES:
            for phi in corners:
                k += 1
                print(f"[{k}/{n_total}] {flux_label} h={h:+.3f} phi={phi}", flush=True)
                rows.append(run_one(flux_label, jxy, jzz, h, phi, out_root, force=args.force))

    summary = {"n_eigs": N_EIGS, "num_sites": NUM_SITES, "dim": list(DIM),
               "flux_cases": FLUX_CASES, "h_values": H_VALUES,
               "corners": args.corners, "rows": rows}
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2))
    print("Wrote", out_root / "summary.json")


if __name__ == "__main__":
    main()
