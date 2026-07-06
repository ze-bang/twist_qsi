#!/usr/bin/env bash
# Launch Eg_Q1 and T2g_Q_xz source-field ED for Jpm=+0.02 (zero-flux QSI), J3=0
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

SCRIPT="twist_qsi_demo/scripts/run_full_ed_quad_qh_J3.py"
SCAN_ROOT="twist_qsi_demo/output/jpm_pos002_zeroflux_j3zero"

HAM="$SCAN_ROOT/J3_0.0000/phi_0.000pi_0.000pi_0.000pi/ham"
QUAD="$SCAN_ROOT/quad_operators"
OUT_ROOT="$SCAN_ROOT/source_field_campaign"
LOG_DIR="$OUT_ROOT/logs"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-6}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-6}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-6}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-6}"

echo "=== Jpm=+0.02 zero-flux QSI source-field runs ==="
echo "ROOT=$ROOT"

for OP in "Eg_Q1" "T2g_Q_xz"; do
  TAG="${OP}_1e-3"
  CASE_DIR="$OUT_ROOT/$TAG"
  CKPT_DIR="$CASE_DIR/checkpoints"
  OUT_NPZ="$CASE_DIR/result.npz"
  LOG="$LOG_DIR/$TAG.serial.log"

  mkdir -p "$LOG_DIR" "$CASE_DIR" "$CKPT_DIR"

  if [[ -f "$OUT_NPZ" ]]; then
    echo "[skip] $TAG already complete"
    continue
  fi

  echo "[run] $TAG"
  env \
    OMP_NUM_THREADS="$OMP_NUM_THREADS" \
    OPENBLAS_NUM_THREADS="$OPENBLAS_NUM_THREADS" \
    MKL_NUM_THREADS="$MKL_NUM_THREADS" \
    NUMEXPR_NUM_THREADS="$NUMEXPR_NUM_THREADS" \
    nice -n 19 taskset -c 8-31 \
    python3 -u "$SCRIPT" \
      --ham "$HAM" \
      --quad "$QUAD" \
      --out "$OUT_NPZ" \
      --checkpoint-dir "$CKPT_DIR" \
      --parallel-sectors 1 \
      --source "${OP}=1e-3" \
      --temp-min 0.005 \
      --temp-max 5.0 \
      --temp-points 80 \
      > "$LOG" 2>&1
  echo "[done] $TAG"
done

echo "Jpm=+0.02 zero-flux campaign complete"
