#!/usr/bin/env bash
# Launch Eg_Q1 source-strength scans for Jpm=+0.02 and +0.04 (zero-flux), J3=0
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

SCRIPT="twist_qsi_demo/scripts/run_full_ed_quad_qh_J3.py"
LAMBDAS=("1e-4" "3e-4" "1e-3" "3e-3")

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-6}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-6}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-6}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-6}"

run_case() {
  local scan_root="$1"
  local tag_prefix="$2"

  local ham="$scan_root/J3_0.0000/phi_0.000pi_0.000pi_0.000pi/ham"
  local quad="$scan_root/quad_operators"
  local out_root="$scan_root/lambda_scan_eg_q1"
  local log_dir="$out_root/logs"

  mkdir -p "$log_dir"

  for lam in "${LAMBDAS[@]}"; do
    local lam_tag="${lam/+}"
    lam_tag="${lam_tag//./p}"
    local case_tag="${tag_prefix}_EgQ1_${lam_tag}"
    local case_dir="$out_root/$case_tag"
    local ckpt_dir="$case_dir/checkpoints"
    local out_npz="$case_dir/result.npz"
    local log="$log_dir/$case_tag.serial.log"

    mkdir -p "$case_dir" "$ckpt_dir"

    if [[ -f "$out_npz" ]]; then
      echo "[skip] $case_tag already complete"
      continue
    fi

    echo "[run] $case_tag (lambda=$lam)"
    env \
      OMP_NUM_THREADS="$OMP_NUM_THREADS" \
      OPENBLAS_NUM_THREADS="$OPENBLAS_NUM_THREADS" \
      MKL_NUM_THREADS="$MKL_NUM_THREADS" \
      NUMEXPR_NUM_THREADS="$NUMEXPR_NUM_THREADS" \
      nice -n 19 taskset -c 8-31 \
      python3 -u "$SCRIPT" \
        --ham "$ham" \
        --quad "$quad" \
        --out "$out_npz" \
        --checkpoint-dir "$ckpt_dir" \
        --parallel-sectors 1 \
        --source "Eg_Q1=${lam}" \
        --temp-min 1e-3 \
        --temp-max 5.0 \
        --temp-points 100 \
        > "$log" 2>&1
    echo "[done] $case_tag"
  done
}

echo "=== Eg_Q1 lambda scan, zero-flux J3=0 ==="
echo "ROOT=$ROOT"

run_case "twist_qsi_demo/output/jpm_pos002_zeroflux_j3zero" "JpmPos002"
run_case "twist_qsi_demo/output/jpm_pos004_zeroflux_j3zero" "JpmPos004"

echo "Eg_Q1 lambda scan complete"
