#!/usr/bin/env python3
"""
Twist-averaged finite-T thermodynamics on the 16-site (1x1x1) pyrochlore super-cell
(default J_pm = -0.3 J_zz from Jxx=Jyy=0.6), with optional Kadowaki three-spin couplings
J_{3s,1}=J_{3s,2}=J_{3s,3}=J3.

By default this uses **full exact diagonalization** (``--method=FULL``) plus the
integrated spectrum thermodynamics path (``--thermo``), which is feasible at 2^16
Hilbert space with streaming ``--symm``.  Use ``--thermo-mode ftlm`` (or ``both``)
to recover the older FTLM stochastic estimator.

Use ``--jpm`` to pick another isotropic point with Jxx=Jyy=-2*Jpm (e.g. ``--jpm -0.1``).

By default each ``ED`` subprocess uses **one BLAS/OpenMP thread**, runs at
**lower CPU priority** (``nice -n 10``), and cases are processed **one at a time**
(no inner parallelism).  Override with ``--blas-threads``, ``--nice`` (0 =
off), or ``--taskset-cpus`` (e.g. ``0`` or ``0-3``) if you want different limits.

Requires a built ``QED/build/ED`` binary. Example::

    python3 run_jpm03_twist_j3_ftlm.py --out ../output/jpm03_j3_scan --j3-list 0,0.03,0.06

Optional finite-T spin moments (microcanonical TPQ with Sz/Sx/Sy fluctuation files)::

    python3 run_jpm03_twist_j3_ftlm.py --out ... --mtpq-spin \\
        --mtpq-iterations=400 --mtpq-interval=40

This is *much* slower than thermodynamics alone; use for moderate ``--mtpq-iterations`` first.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from itertools import product
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

from twist_trilinear_helper import write_pyrochlore_twisted_xxz_trilinear  # noqa: E402

# J_pm = -(Jxx+Jyy)/4 = -0.3  =>  Jxx = Jyy = 0.6
JXX = 0.6
JYY = 0.6
JZZ = 1.0

NUM_SITES = 16
DIM = (1, 1, 1)

# Light defaults (override with --ftlm-samples / --ftlm-krylov / …).
# Streaming symmetry + FTLM is still costly on 16 sites; keep these modest.
FTLM_SAMPLES = 3
FTLM_KRYLOV = 40
TEMP_MIN = 0.005
TEMP_MAX = 5.0
TEMP_BINS = 28

# 0 = skip Lanczos block (saves a lot of time; use --lanczos-neigs > 0 if needed).
LANCZOS_NEIGS = 0
LANCZOS_ITERATIONS = 600


def twist_label(phi):
    return "phi_" + "_".join(
        ("pi" if abs(p - np.pi) < 1e-8 else f"{p/np.pi:.3f}pi") for p in phi
    )


def parse_twist_list(spec: str):
    """Parse semicolon-separated triples in units of pi, e.g. ``0,0,0;1,0,1``."""
    twists = []
    for item in spec.split(";"):
        item = item.strip()
        if not item:
            continue
        vals = [float(x) for x in item.split(",") if x.strip() != ""]
        if len(vals) != 3:
            raise ValueError(f"twist entry must have three comma-separated values: {item!r}")
        twists.append(tuple(v * np.pi for v in vals))
    if not twists:
        raise ValueError("--twist-list did not contain any twist triples")
    return twists


def find_ed_binary() -> Path:
    cands = [
        ROOT.parent / "QED" / "build" / "ED",
        Path(os.environ.get("QED_ED_BIN", "")),
    ]
    for p in cands:
        if p and p.is_file() and os.access(p, os.X_OK):
            return p
    raise FileNotFoundError(
        "Could not find executable QED/build/ED. Build the QED project first "
        "(see QED/README.md) or set QED_ED_BIN to the ED binary path."
    )


def _limited_compute_env(*, blas_threads: int) -> dict:
    """Limit BLAS / OpenMP parallelism so one ``ED`` process does not occupy the host."""
    e = os.environ.copy()
    t = str(max(1, int(blas_threads)))
    for k in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
        "GOTO_NUM_THREADS",
        "ATLAS_NUM_THREADS",
        "ARMPL_NUM_THREADS",
    ):
        e[k] = t
    # Avoid nested / dynamic OpenMP teams fighting the caps above.
    e["OMP_DYNAMIC"] = "FALSE"
    e["OMP_MAX_ACTIVE_LEVELS"] = "1"
    e["MKL_DYNAMIC"] = "FALSE"
    e["KMP_AFFINITY"] = "disabled"
    e["KMP_BLOCKTIME"] = "0"
    return e


def _ed_argv_prefix(*, nice: int, taskset_cpus: str) -> list[str]:
    """Optional ``nice`` / ``taskset`` prefix placed before the ``ED`` binary."""
    pre: list[str] = []
    if nice > 0:
        if shutil.which("nice"):
            pre.extend(["nice", "-n", str(nice)])
        else:
            print("warning: nice(1) not found; ED will run at normal priority", file=sys.stderr)
    cpus = taskset_cpus.strip()
    if cpus:
        if shutil.which("taskset"):
            pre.extend(["taskset", "-c", cpus])
        else:
            print(
                "warning: taskset(1) not found; cannot apply --taskset-cpus; "
                "ED may use all CPUs allowed by BLAS threads",
                file=sys.stderr,
            )
    return pre


def run_ftlm(
    ed_bin: Path,
    ed_argv_prefix: list[str],
    ham_dir: Path,
    out_dir: Path,
    log: Path,
    force: bool,
    *,
    use_symm: bool,
    samples: int,
    krylov: int,
    temp_bins: int,
    subprocess_env: dict,
):
    h5 = out_dir / "ed_results.h5"
    if not force and h5.exists():
        return
    cmd = ed_argv_prefix + [
        str(ed_bin),
        str(ham_dir),
        "--method=FTLM",
        f"--num_sites={NUM_SITES}",
        f"--output={out_dir}",
        f"--samples={samples}",
        f"--ftlm-krylov={krylov}",
        f"--temp_min={TEMP_MIN}",
        f"--temp_max={TEMP_MAX}",
        f"--temp_bins={temp_bins}",
    ]
    if use_symm:
        cmd.append("--symm")
    t0 = time.time()
    with open(log, "a") as f:
        f.write(f"\n$ {' '.join(cmd)}\n")
        f.flush()
        rc = subprocess.run(
            cmd, stdout=f, stderr=subprocess.STDOUT, env=subprocess_env
        ).returncode
    with open(log, "a") as f:
        f.write(f"# FTLM done in {time.time()-t0:.1f}s rc={rc}\n")
    if rc != 0:
        raise RuntimeError(f"FTLM failed rc={rc} (see {log})")


def run_full_ed_thermo(
    ed_bin: Path,
    ed_argv_prefix: list[str],
    ham_dir: Path,
    out_dir: Path,
    log: Path,
    force: bool,
    *,
    use_symm: bool,
    temp_bins: int,
    subprocess_env: dict,
):
    """Full spectrum ED + partition-function thermodynamics (writes ``/thermodynamics``)."""
    h5 = out_dir / "ed_results.h5"
    if not force and h5.exists():
        return
    cmd = ed_argv_prefix + [
        str(ed_bin),
        str(ham_dir),
        "--method=FULL",
        "--eigenvalues=FULL",
        "--thermo",
        f"--num_sites={NUM_SITES}",
        f"--output={out_dir}",
        f"--temp_min={TEMP_MIN}",
        f"--temp_max={TEMP_MAX}",
        f"--temp_bins={temp_bins}",
    ]
    if use_symm:
        cmd.append("--symm")
    t0 = time.time()
    with open(log, "a") as f:
        f.write(f"\n$ {' '.join(cmd)}\n")
        f.flush()
        rc = subprocess.run(
            cmd, stdout=f, stderr=subprocess.STDOUT, env=subprocess_env
        ).returncode
    with open(log, "a") as f:
        f.write(f"# FULL+thermo done in {time.time()-t0:.1f}s rc={rc}\n")
    if rc != 0:
        raise RuntimeError(f"FULL+thermo failed rc={rc} (see {log})")


def run_lanczos(
    ed_bin: Path,
    ed_argv_prefix: list[str],
    ham_dir: Path,
    out_dir: Path,
    log: Path,
    force: bool,
    n_eigs: int,
    *,
    use_symm: bool,
    iterations: int,
    subprocess_env: dict,
):
    h5 = out_dir / "ed_results.h5"
    if n_eigs <= 0:
        return
    if not force and h5.exists():
        return
    cmd = ed_argv_prefix + [
        str(ed_bin),
        str(ham_dir),
        "--method=LANCZOS",
        f"--num_sites={NUM_SITES}",
        f"--output={out_dir}",
        f"--eigenvalues={n_eigs}",
        f"--iterations={iterations}",
        "--tolerance=1e-9",
    ]
    if use_symm:
        cmd.append("--symm")
    t0 = time.time()
    with open(log, "a") as f:
        f.write(f"\n$ {' '.join(cmd)}\n")
        f.flush()
        rc = subprocess.run(
            cmd, stdout=f, stderr=subprocess.STDOUT, env=subprocess_env
        ).returncode
    with open(log, "a") as f:
        f.write(f"# LANCZOS done in {time.time()-t0:.1f}s rc={rc}\n")
    if rc != 0:
        raise RuntimeError(f"Lanczos failed rc={rc} (see {log})")


def run_mtpq_spin(
    ed_bin: Path,
    ed_argv_prefix: list[str],
    ham_dir: Path,
    out_dir: Path,
    log: Path,
    force: bool,
    iterations: int,
    interval: int,
    *,
    use_symm: bool,
    subprocess_env: dict,
):
    h5 = out_dir / "ed_results.h5"
    if not force and h5.exists():
        return
    cmd = ed_argv_prefix + [
        str(ed_bin),
        str(ham_dir),
        "--method=mTPQ",
        f"--num_sites={NUM_SITES}",
        f"--output={out_dir}",
        "--samples=1",
        f"--iterations={iterations}",
        f"--measurement_interval={interval}",
        "--compute-spin-correlations",
    ]
    if use_symm:
        cmd.append("--symm")
    t0 = time.time()
    with open(log, "a") as f:
        f.write(f"\n$ {' '.join(cmd)}\n")
        f.flush()
        rc = subprocess.run(
            cmd, stdout=f, stderr=subprocess.STDOUT, env=subprocess_env
        ).returncode
    with open(log, "a") as f:
        f.write(f"# mTPQ done in {time.time()-t0:.1f}s rc={rc}\n")
    if rc != 0:
        raise RuntimeError(f"mTPQ failed rc={rc} (see {log})")


def run_one_j3(
    j3: float,
    phi,
    run_root: Path,
    ed_bin: Path,
    *,
    jxx: float,
    jyy: float,
    force: bool,
    mtpq: bool,
    mtpq_iter: int,
    mtpq_interval: int,
    use_symm: bool,
    thermo_mode: str,
    ftlm_samples: int,
    ftlm_krylov: int,
    temp_bins: int,
    lanczos_neigs: int,
    lanczos_iterations: int,
    subprocess_env: dict,
    ed_argv_prefix: list[str],
):
    tag_phi = twist_label(phi)
    case_dir = run_root / f"J3_{j3:.4f}" / tag_phi
    ham_dir = case_dir / "ham"
    ftlm_dir = case_dir / "ftlm"
    full_thermo_dir = case_dir / "full_thermo"
    spec_dir = case_dir / "spectrum"
    mtpq_dir = case_dir / "mtpq_spin"
    if thermo_mode in ("ftlm", "both"):
        ftlm_dir.mkdir(parents=True, exist_ok=True)
    if thermo_mode in ("full", "both"):
        full_thermo_dir.mkdir(parents=True, exist_ok=True)
    spec_dir.mkdir(parents=True, exist_ok=True)
    log = case_dir / "run.log"

    info = write_pyrochlore_twisted_xxz_trilinear(
        str(ham_dir),
        jxx,
        jyy,
        JZZ,
        tuple(float(x) for x in phi),
        three_spin_coeff=j3,
        dim1=DIM[0],
        dim2=DIM[1],
        dim3=DIM[2],
        h_field=0.0,
    )
    log.write_text(f"# J3={j3} twist={tuple(phi)}\n# {info}\n")

    if thermo_mode in ("ftlm", "both"):
        run_ftlm(
            ed_bin, ed_argv_prefix, ham_dir, ftlm_dir, log, force,
            use_symm=use_symm,
            samples=ftlm_samples,
            krylov=ftlm_krylov,
            temp_bins=temp_bins,
            subprocess_env=subprocess_env,
        )
    if thermo_mode in ("full", "both"):
        run_full_ed_thermo(
            ed_bin, ed_argv_prefix, ham_dir, full_thermo_dir, log, force,
            use_symm=use_symm,
            temp_bins=temp_bins,
            subprocess_env=subprocess_env,
        )
    run_lanczos(
        ed_bin, ed_argv_prefix, ham_dir, spec_dir, log, force, lanczos_neigs,
        use_symm=use_symm,
        iterations=lanczos_iterations,
        subprocess_env=subprocess_env,
    )
    if mtpq:
        mtpq_dir.mkdir(parents=True, exist_ok=True)
        run_mtpq_spin(
            ed_bin, ed_argv_prefix, ham_dir, mtpq_dir, log, force,
            iterations=mtpq_iter, interval=mtpq_interval,
            use_symm=use_symm,
            subprocess_env=subprocess_env,
        )

    rec = {
        "J3": j3,
        "phi": list(map(float, phi)),
        "tag": tag_phi,
        "spec_h5": str((spec_dir / "ed_results.h5").resolve()),
        "info": {k: v for k, v in info.items() if k != "wrap_summary"},
    }
    if thermo_mode in ("ftlm", "both"):
        rec["ftlm_h5"] = str((ftlm_dir / "ed_results.h5").resolve())
    if thermo_mode in ("full", "both"):
        rec["full_thermo_h5"] = str((full_thermo_dir / "ed_results.h5").resolve())
    # Primary path for downstream twist-average scripts (exact ED preferred).
    if thermo_mode == "ftlm":
        rec["thermo_h5"] = rec["ftlm_h5"]
    elif thermo_mode == "full":
        rec["thermo_h5"] = rec["full_thermo_h5"]
    elif thermo_mode == "both":
        rec["thermo_h5"] = rec["full_thermo_h5"]
    else:
        rec["ham_dir"] = str(ham_dir.resolve())
    if mtpq:
        rec["mtpq_dir"] = str(mtpq_dir)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default=str(ROOT / "output" / "jpm03_j3_scan"))
    ap.add_argument(
        "--jpm",
        type=float,
        default=None,
        help="isotropic transverse coupling in Hamiltonian units (Jzz=1): set "
        "Jxx=Jyy=-2*Jpm, overriding built-in JXX=JYY=0.6. Example: --jpm -0.1 => Jxx=Jyy=0.2.",
    )
    ap.add_argument(
        "--j3-list",
        type=str,
        default="0.0,0.06",
        help="comma-separated J3 values (broadcast to all three geometric classes)",
    )
    ap.add_argument(
        "--twist-list",
        type=str,
        default="",
        help="semicolon-separated twist triples in units of pi; default is all 8 corners, e.g. '0,0,0;1,0,1'",
    )
    ap.add_argument(
        "--no-symm",
        dest="use_symm",
        action="store_false",
        default=True,
        help="do not pass --symm to ED (heavier; default is symmetry on)",
    )
    ap.add_argument(
        "--thermo-mode",
        type=str,
        choices=("full", "ftlm", "both", "none"),
        default="full",
        help="finite-T thermodynamics driver: full ED spectrum (default), FTLM, both, or none",
    )
    ap.add_argument(
        "--blas-threads",
        type=int,
        default=1,
        metavar="N",
        help="OMP/BLAS/MKL thread cap per ED process (default: 1; keeps one case polite)",
    )
    ap.add_argument(
        "--nice",
        type=int,
        default=10,
        metavar="N",
        help="run each ED under ``nice -n N`` (0 disables; default 10 lowers CPU priority)",
    )
    ap.add_argument(
        "--taskset-cpus",
        type=str,
        default="",
        metavar="LIST",
        help="if set, prefix ED with ``taskset -c LIST`` (e.g. 0 or 2-5) to pin CPUs",
    )
    ap.add_argument("--ftlm-samples", type=int, default=FTLM_SAMPLES)
    ap.add_argument("--ftlm-krylov", type=int, default=FTLM_KRYLOV)
    ap.add_argument("--temp-bins", type=int, default=TEMP_BINS)
    ap.add_argument("--lanczos-neigs", type=int, default=LANCZOS_NEIGS)
    ap.add_argument("--lanczos-iterations", type=int, default=LANCZOS_ITERATIONS)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--mtpq-spin", action="store_true",
                    help="also run mTPQ with spin fluctuation files (slow)")
    ap.add_argument("--mtpq-iterations", type=int, default=500)
    ap.add_argument("--mtpq-interval", type=int, default=50)
    args = ap.parse_args()

    if int(args.blas_threads) < 1:
        ap.error("--blas-threads must be >= 1")
    if int(args.nice) < 0:
        ap.error("--nice must be >= 0 (0 disables the nice wrapper)")

    j3s = [float(x) for x in args.j3_list.split(",") if x.strip() != ""]
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    jxx = float(JXX)
    jyy = float(JYY)
    if args.jpm is not None:
        j_iso = -2.0 * float(args.jpm)
        jxx = jyy = j_iso

    try:
        ed_bin = find_ed_binary()
    except FileNotFoundError as e:
        print(e, file=sys.stderr)
        sys.exit(1)

    try:
        twists = (
            parse_twist_list(args.twist_list)
            if args.twist_list.strip()
            else list(product([0.0, np.pi], repeat=3))
        )
    except ValueError as e:
        ap.error(str(e))
    sub_env = _limited_compute_env(blas_threads=args.blas_threads)
    ed_argv_prefix = _ed_argv_prefix(nice=args.nice, taskset_cpus=args.taskset_cpus)
    summary = {
        "Jxx": jxx,
        "Jyy": jyy,
        "Jzz": JZZ,
        "Jpm": -(jxx + jyy) / 4.0,
        "j3_values": j3s,
        "n_sites": NUM_SITES,
        "dim": list(DIM),
        "use_symm": args.use_symm,
        "thermo_mode": args.thermo_mode,
        "resource_limits": {
            "blas_threads": int(args.blas_threads),
            "nice": int(args.nice),
            "taskset_cpus": args.taskset_cpus.strip() or None,
        },
        "thread_env": {
            k: sub_env[k]
            for k in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OMP_DYNAMIC",
                "OMP_MAX_ACTIVE_LEVELS",
                "MKL_DYNAMIC",
            )
            if k in sub_env
        },
        "ftlm_samples": args.ftlm_samples,
        "ftlm_krylov": args.ftlm_krylov,
        "temp_grid": {"min": TEMP_MIN, "max": TEMP_MAX, "bins": args.temp_bins},
        "lanczos_neigs": args.lanczos_neigs,
        "lanczos_iterations": args.lanczos_iterations,
        "runs": [],
    }

    t0 = time.time()
    total = len(j3s) * len(twists)
    k = 0
    for j3 in j3s:
        for phi in twists:
            k += 1
            print(f"\n[{k}/{total}] J3={j3:.4g} twist={tuple(phi)}", flush=True)
            rec = run_one_j3(
                j3, np.asarray(phi, dtype=float), out_root, ed_bin,
                jxx=jxx,
                jyy=jyy,
                force=args.force,
                mtpq=args.mtpq_spin,
                mtpq_iter=args.mtpq_iterations,
                mtpq_interval=args.mtpq_interval,
                use_symm=args.use_symm,
                thermo_mode=args.thermo_mode,
                ftlm_samples=args.ftlm_samples,
                ftlm_krylov=args.ftlm_krylov,
                temp_bins=args.temp_bins,
                lanczos_neigs=args.lanczos_neigs,
                lanczos_iterations=args.lanczos_iterations,
                subprocess_env=sub_env,
                ed_argv_prefix=ed_argv_prefix,
            )
            summary["runs"].append(rec)
            (out_root / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\nTotal elapsed: {time.time()-t0:.1f}s")
    print(f"Wrote {out_root / 'summary.json'}")


if __name__ == "__main__":
    main()
