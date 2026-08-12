#!/bin/bash
# WP2 first pass: the fit-constrained region, certified point by point.
#
# Each task does BOTH steps, because the pilot showed the cheap one is not
# trustworthy on its own: at QMC_ordered the multi-anchor solve reported 0/90
# while the bordered per-sector solver found all 90.  A grid built on the
# anchor solve alone would have painted the entire positive-Jpm half as
# dissolved, and that would have been the load-bearing claim of the null.
#
#   step 1  run_masked_band_complete --full-basis   (~25 min) -> bordered band
#   step 2  run_jpmpm_fold_plane --bordered         (~8 min)  -> certified peaks
#
# Certification rule for reading the output: only points reporting 90/90 carry
# peak positions.  A partial band leaves ice-descended raw levels in the
# ensemble beside their replacements and invents peaks -- NLC_B at 25/90
# produced a spurious middle peak that would have read as a plausible 0.072
# two-peak ratio.
#
# The question this grid answers: does ANY certified NN-XYZ point put a gauge
# feature at the measured 25/65 mK ratio of 0.385?  Best certified so far is
# 0.111 (QMC_ordered); NLC_A gives 0.047.  A clean null here is the publishable
# outcome the plan anticipated -- a rigorous exclusion answering the
# experiment's own open question about beyond-nearest-neighbour terms.
#SBATCH --account=def-ybkim_cpu
#SBATCH --job-name=wp2_grid
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=32 --mem=120G
#SBATCH --time=05:00:00
#SBATCH --array=1-23
#SBATCH --output=/lustre09/project/6003507/zhouzb79/twist_qsi/campaign/outputs/wp2_gate/wp2_grid_%A_%a.log
set -uo pipefail
module purge
module load StdEnv/2023 gcc/12.3 python/3.11.5 scipy-stack/2024b
export FLEXIBLAS=AOCL
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-32}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-32}
cd /lustre09/project/6003507/zhouzb79/twist_qsi

POINTS=campaign/wp2_grid_points.txt
LINE=$(grep -v '^#' "$POINTS" | sed -n "${SLURM_ARRAY_TASK_ID}p")
JPM=$(echo "$LINE" | awk '{print $1}')
PP=$(echo "$LINE" | awk '{print $2}')
if [ -z "$JPM" ]; then echo "no point for index $SLURM_ARRAY_TASK_ID"; exit 0; fi
echo "point $SLURM_ARRAY_TASK_ID: jpm=$JPM jpmpm=$PP"

python3 -u campaign/run_masked_band_complete.py "$JPM" --jpmpm "$PP" --full-basis
BAND=$?
if [ "$BAND" -ne 0 ]; then echo "bordered band FAILED (exit $BAND)"; exit $BAND; fi

python3 -u campaign/run_jpmpm_fold_plane.py --bordered "$JPM" "$PP"
STATUS=$?
echo "exit=$STATUS  $(date)"
exit $STATUS
