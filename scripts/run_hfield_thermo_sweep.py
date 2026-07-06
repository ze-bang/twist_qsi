"""
FTLM thermodynamics vs transverse-field strength h_perp on the 16-site
pyrochlore (1x1x1) cluster, twist-averaged over the eight-corner {0,pi}^3
grid, at fixed flux sign (default: Jpm=-0.1, "pi-flux").

C(T, h_perp) is even in h_perp (confirmed in the Lanczos sweep), so only
h_perp >= 0 is run here.

Output goes to a dedicated folder (default ../hfield_thermo/data), kept
separate from both the main twist_qsi_demo/output/demo pipeline and the
paper/figs directory.

Usage:
    python3 run_hfield_thermo_sweep.py --out ../hfield_thermo/data
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

# NOTE: this ED binary's `--method=FTLM --thermo` builds thermodynamics
# from the *finite* computed eigenvalue list (calculate_thermodynamics_
# from_spectrum in workflows.cpp), not from genuine stochastic FTLM
# sampling -- --samples/--ftlm-krylov are unused for this code path. So
# we request enough low-lying eigenvalues to resolve the low-T peak, and
# restrict temp_max well below where the truncated spectrum runs out of
# levels (high-T saturation is not reliable at this eigenvalue count).
N_EIGS = 400
ITERATIONS = 3000
TOLERANCE = 1e-9
TEMP_MIN = 0.005
TEMP_MAX = 1.0
TEMP_BINS = 150

H_VALUES = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.12]

# (label, Jxx=Jyy, Jzz) -- Jpm = -(Jxx+Jyy)/4
FLUX_CASES = {
    "piflux": (0.2, 1.0),     # Jpm = -0.1
    "zeroflux": (-0.1, 1.0),  # Jpm = +0.05
}


def h_label(h):
    return f"h{h:.3f}"


def twist_label(phi):
    return "phi_" + "_".join(
        ("pi" if abs(p - np.pi) < 1e-8 else f"{p/np.pi:.3f}pi") for p in phi
    )


def run_one(flux_label, jxy, jzz, h_perp, phi, out_root: Path, force=False):
    tag = f"{flux_label}/{h_label(h_perp)}/{twist_label(phi)}"
    case_dir = out_root / flux_label / h_label(h_perp) / twist_label(phi)
    ham_dir = case_dir / "ham"
    ftlm_dir = case_dir / "ftlm"
    ftlm_dir.mkdir(parents=True, exist_ok=True)

    write_pyrochlore_xxz_with_twist(
        str(ham_dir), jxy, jxy, jzz, tuple(phi),
        dim1=DIM[0], dim2=DIM[1], dim3=DIM[2], h_perp=h_perp,
    )

    ftlm_h5 = ftlm_dir / "ed_results.h5"
    log = case_dir / "run.log"
    if force or not ftlm_h5.exists():
        cmd = [str(ED_BIN), str(ham_dir),
               "--method=FTLM",
               f"--num_sites={NUM_SITES}",
               f"--output={ftlm_dir}",
               f"--eigenvalues={N_EIGS}",
               f"--iterations={ITERATIONS}",
               f"--tolerance={TOLERANCE}",
               f"--temp_min={TEMP_MIN}",
               f"--temp_max={TEMP_MAX}",
               f"--temp_bins={TEMP_BINS}",
               "--thermo"]
        t0 = time.time()
        with open(log, "w") as f:
            f.write(f"# {tag}\n$ {' '.join(cmd)}\n")
            f.flush()
            rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT).returncode
        dt = time.time() - t0
        with open(log, "a") as f:
            f.write(f"\n# done in {dt:.1f}s rc={rc}\n")
        if rc != 0:
            raise RuntimeError(f"FTLM failed for {tag} (rc={rc}), see {log}")
    return {"flux": flux_label, "jxy": jxy, "jzz": jzz, "h_perp": h_perp,
            "phi": list(map(float, phi)), "ftlm_h5": str(ftlm_h5)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "hfield_thermo" / "data"))
    ap.add_argument("--flux", choices=["piflux", "zeroflux"], default="piflux")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    jxy, jzz = FLUX_CASES[args.flux]
    pi = float(np.pi)
    corners = list(product([0.0, pi], repeat=3))

    rows = []
    n_total = len(H_VALUES) * len(corners)
    k = 0
    for h in H_VALUES:
        for phi in corners:
            k += 1
            print(f"[{k}/{n_total}] {args.flux} h={h:+.3f} phi={phi}", flush=True)
            rows.append(run_one(args.flux, jxy, jzz, h, phi, out_root, force=args.force))

    summary = {"flux": args.flux, "jxy": jxy, "jzz": jzz,
               "num_sites": NUM_SITES, "dim": list(DIM),
               "n_eigs": N_EIGS, "iterations": ITERATIONS,
               "h_values": H_VALUES, "rows": rows}
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2))
    print("Wrote", out_root / "summary.json")


if __name__ == "__main__":
    main()
