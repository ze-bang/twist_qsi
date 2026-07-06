#!/usr/bin/env bash
# Launch H-decomposition Eg_Q1 runs for Jpm=+0.02,+0.04 and -0.04 (pi-flux), J3=0
# Uses --decompose-H to also store alpha_{Q,Ising} and alpha_{Q,pm} contributions
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

SCRIPT="twist_qsi_demo/scripts/run_full_ed_quad_qh_J3.py"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-6}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-6}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-6}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-6}"

run_case() {
  local scan_root="$1"
  local tag="$2"
  local lam="$3"

  local ham="$scan_root/J3_0.0000/phi_0.000pi_0.000pi_0.000pi/ham"
  local quad="$scan_root/quad_operators"
  local out_root="$scan_root/H_decomp_campaign"
  local log_dir="$out_root/logs"
  local case_dir="$out_root/$tag"
  local ckpt_dir="$case_dir/checkpoints"
  local out_npz="$case_dir/result.npz"
  local log="$log_dir/$tag.serial.log"

  mkdir -p "$log_dir" "$case_dir" "$ckpt_dir"

  if [[ -f "$out_npz" ]]; then
    echo "[skip] $tag already complete"
    return
  fi

  echo "[run] $tag (lambda=$lam)"
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
      --decompose-H \
      --temp-min 1e-3 \
      --temp-max 5.0 \
      --temp-points 100 \
      > "$log" 2>&1
  echo "[done] $tag"
}

echo "=== H-decomposition Eg_Q1 runs, J3=0 ==="
echo "ROOT=$ROOT"

run_case "twist_qsi_demo/output/jpm_pos004_zeroflux_j3zero" "EgQ1_1e-3_decomp" "1e-3"
run_case "twist_qsi_demo/output/jpm_pos002_zeroflux_j3zero" "EgQ1_1e-3_decomp" "1e-3"
run_case "twist_qsi_demo/output/jpm_neg004_piflux_j3zero"   "EgQ1_1e-3_decomp" "1e-3"

echo "H-decomp campaign complete"
