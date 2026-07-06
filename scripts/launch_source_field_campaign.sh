#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

SCAN_ROOT="twist_qsi_demo/output/jpm03_j3_verify_pbc_qsus"
HAM="$SCAN_ROOT/J3_0.0800/phi_0.000pi_0.000pi_0.000pi/ham"
QUAD="$SCAN_ROOT/quad_operators"
OUT_ROOT="$SCAN_ROOT/source_field_campaign"
LOG_DIR="$OUT_ROOT/logs"
mkdir -p "$OUT_ROOT" "$LOG_DIR"

SCRIPT="twist_qsi_demo/scripts/run_full_ed_quad_qh_J3.py"

# First-pass pilot grid: enough to distinguish Eg vs T2g induced responses
# without exploding wall-clock cost.
CASES=(
  "EgQ1_1e-3 Eg_Q1 1e-3"
  "EgQ2_1e-3 Eg_Q2 1e-3"
  "T2gXY_1e-3 T2g_Q_xy 1e-3"
  "T2gXZ_1e-3 T2g_Q_xz 1e-3"
)

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-6}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-6}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-6}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-6}"

echo "=== source-field campaign ==="
echo "ROOT=$ROOT"
echo "HAM=$HAM"
echo "OUT_ROOT=$OUT_ROOT"
echo "OMP_NUM_THREADS=$OMP_NUM_THREADS"

for entry in "${CASES[@]}"; do
  read -r tag op lam <<<"$entry"
  case_dir="$OUT_ROOT/$tag"
  out_npz="$case_dir/result.npz"
  ckpt_dir="$case_dir/checkpoints"
  log="$LOG_DIR/$tag.log"
  mkdir -p "$case_dir" "$ckpt_dir"

  if [[ -f "$out_npz" ]]; then
    echo "[skip] $tag already complete"
    continue
  fi

  echo "[launch] $tag  source=${op}=${lam}"
  nohup env \
    OMP_NUM_THREADS="$OMP_NUM_THREADS" \
    OPENBLAS_NUM_THREADS="$OPENBLAS_NUM_THREADS" \
    MKL_NUM_THREADS="$MKL_NUM_THREADS" \
    NUMEXPR_NUM_THREADS="$NUMEXPR_NUM_THREADS" \
    nice -n 19 taskset -c 8-31 \
    python3 -u "$SCRIPT" \
      --ham "$HAM" \
      --quad "$QUAD" \
      --out "$out_npz" \
      --checkpoint-dir "$ckpt_dir" \
      --parallel-sectors 4 \
      --source "${op}=${lam}" \
      --temp-min 0.005 \
      --temp-max 5.0 \
      --temp-points 80 \
      > "$log" 2>&1 &
done

echo
echo "Launched source-field campaign."
echo "Logs:"
echo "  $LOG_DIR"
echo "Outputs:"
echo "  $OUT_ROOT"
