#!/bin/bash
# QED winding-free M=2 at the remaining points, GPU, sequential.
cd /home/pc_linux/exact_diagonalization_clean/twist_qsi_demo
LOG=campaign/outputs/logs
run() {  # jpm jpmpm tag
  echo "=== QED $3  $(date) ==="
  OMP_NUM_THREADS=8 python3 campaign/qed_material_band.py \
    --jpm "$1" --jpmpm "$2" --n-grid 2 --solver krylov_schur --device gpu \
    > "$LOG/qed_$3.log" 2>&1
  grep -E "winding_free_peak_over_g6|winding_free_peak_mK|periodic_peak_mK|ice_overlap_min" "$LOG/qed_$3.log"
}
run -0.15   0.0    xxz15
run -0.125  0.085  nlcA
run -0.1515 0.085  tuned
echo "=== BATCH DONE $(date) ==="
