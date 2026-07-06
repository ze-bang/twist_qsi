#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
ROOT=twist_qsi_demo/output/jpm03_j3_verify_pbc_qsus
HAM="$ROOT/J3_0.0800/phi_0.000pi_0.000pi_0.000pi/ham"
CKPT="$HAM/full_thermo_quad/checkpoints"
OUT="$HAM/full_thermo_quad/result.npz"
LOG="$ROOT/logs/full_ed_quad_J3_0p08_nohup.log"
mkdir -p "$CKPT" "$(dirname "$LOG")"
export OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 MKL_NUM_THREADS=6 NUMEXPR_NUM_THREADS=6
exec nice -n 19 taskset -c 8-31 python3 -u twist_qsi_demo/scripts/run_full_ed_quad_qh_J3.py \
  --ham "$HAM" \
  --quad "$ROOT/quad_operators" \
  --out "$OUT" \
  --checkpoint-dir "$CKPT" \
  --parallel-sectors 4 \
  --temp-min 0.005 --temp-max 5.0 --temp-points 80
