#!/bin/bash
# Regenerate every main-text figure from current campaign outputs.
#
# This is a PROVENANCE TEST as much as a rendering step.  fig_summary.pdf and
# fig_peak_tracking.pdf are dated 3 August and have been carried forward
# unchanged through a campaign that has since corrected several numbers; if
# either no longer regenerates from the outputs on disk, the manuscript is
# citing a figure whose inputs no longer exist in the form that produced it.
# That is exactly the provenance gap this audit is meant to find, so the job
# keeps going after a failure and reports which targets succeeded rather than
# aborting on the first one.
#SBATCH --account=def-ybkim_cpu
#SBATCH --job-name=figures
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16 --mem=120G
#SBATCH --time=03:00:00
#SBATCH --output=/lustre09/project/6003507/zhouzb79/twist_qsi/campaign/outputs/figures_%j.log
set -uo pipefail
module purge
module load StdEnv/2023 gcc/12.3 python/3.11.5 scipy-stack/2024b
export FLEXIBLAS=AOCL
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export MPLBACKEND=Agg
cd /lustre09/project/6003507/zhouzb79/twist_qsi
PY=$(command -v python3)

FAIL=0
run () {
  echo "########## $* ##########"
  "$PY" -u "$@"
  local status=$?
  if [ $status -ne 0 ]; then echo "FAILED (exit $status): $*"; FAIL=1; fi
  return 0
}

# new figure first: all of its inputs are current
run campaign/make_loop_hierarchy_figure.py

# provenance re-test of the two figures already cited by main.tex
run campaign/make_peak_tracking_figure.py
run campaign/make_figures.py

echo "########## figure inventory ##########"
ls -la paper/figs/*.pdf
echo "aggregate_fail=$FAIL  $(date)"
exit $FAIL
