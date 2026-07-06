#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

SCRIPT="twist_qsi_demo/scripts/run_full_ed_quad_qh_J3.py"

CASES=(
  "twist_qsi_demo/output/jpm03_j3_verify_pbc_qsus Jpm03_J3zero_EgQ1 Eg_Q1 1e-3"
  "twist_qsi_demo/output/jpm03_j3_verify_pbc_qsus Jpm03_J3zero_T2gXZ T2g_Q_xz 1e-3"
  "twist_qsi_demo/output/jpm02_j3_scan_symm_light Jpm02_J3zero_EgQ1 Eg_Q1 1e-3"
  "twist_qsi_demo/output/jpm02_j3_scan_symm_light Jpm02_J3zero_T2gXZ T2g_Q_xz 1e-3"
  "twist_qsi_demo/output/jpm01_j3_scan_symm_light Jpm01_J3zero_EgQ1 Eg_Q1 1e-3"
  "twist_qsi_demo/output/jpm01_j3_scan_symm_light Jpm01_J3zero_T2gXZ T2g_Q_xz 1e-3"
)

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-6}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-6}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-6}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-6}"

echo "=== serial source-field J3=0 map ==="
echo "ROOT=$ROOT"
echo "OMP_NUM_THREADS=$OMP_NUM_THREADS"

for entry in "${CASES[@]}"; do
  read -r scan_root tag op lam <<<"$entry"
  ham="$scan_root/J3_0.0000/phi_0.000pi_0.000pi_0.000pi/ham"
  quad="$scan_root/quad_operators"
  out_root="$scan_root/source_field_campaign_j3zero"
  log_dir="$out_root/logs"
  case_dir="$out_root/$tag"
  out_npz="$case_dir/result.npz"
  ckpt_dir="$case_dir/checkpoints"
  log="$log_dir/$tag.serial.log"

  mkdir -p "$log_dir" "$case_dir" "$ckpt_dir"

  if [[ -f "$out_npz" ]]; then
    echo "[skip] $tag already complete"
    continue
  fi

  echo "[run] $tag source=${op}=${lam}"
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
      --source "${op}=${lam}" \
      --temp-min 0.005 \
      --temp-max 5.0 \
      --temp-points 80 \
      > "$log" 2>&1
  echo "[done] $tag"
done

echo "serial J3=0 source-field map complete"
