#!/bin/bash
# Full regeneration of every Fig-1 product from the verified current code.
# Order matters: the full-ED cache gates the 0.046 nonperturbative run, which
# gates the DSSF and the loop audit (the audit also needs 0.030).
set -e
cd /home/pc_linux/exact_diagonalization_clean/twist_qsi_demo
LOG=campaign/outputs/logs
mkdir -p "$LOG"

step() {
  local name="$1"
  shift
  echo "=== $name  $(date) ==="
  "$@" > "$LOG/regen_$name.log" 2>&1
  echo "    done $(date)"
}

step full_ed        python3 campaign/regen_full_ed_cache.py
step nonpert_0046   python3 campaign/run_nonperturbative.py --jpm 0.046 --max-grid 4
step nonpert_0030   python3 campaign/run_nonperturbative.py --jpm 0.030 --max-grid 4
step dssf           python3 campaign/run_dssf.py
step validation     python3 campaign/run_validation.py
step loop_audit     python3 campaign/run_loop_amplitude_audit.py
step figures        python3 campaign/make_figures.py
echo "=== PIPELINE DONE $(date) ==="
