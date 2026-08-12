#!/bin/bash
# WP2: the bordered cross-check the pilot's "band 0/90" demands.
#
# The multi-anchor solve returned ZERO roots at NLC_B (+0.049, 0.128).  That is
# NOT yet a dissolution result.  The anchor solve brackets roots below the
# GLOBAL continuum edge, so a branch that has risen past that edge is reported
# as lost even when its own transport sector still holds it -- the plan records
# exactly this at (-0.125, 0.30), where the anchor solve found 27 of 90 and the
# bordered per-sector solver found all 90.
#
# The bordered solver uses each sector's own live-pole edge, so it is the
# instrument that can tell "the band is gone" from "the anchor solve could not
# see it".  Until this runs, 0/90 means only that the cheap solver found
# nothing, and a null published on that basis would be an artifact of the
# search, not a statement about the nearest-neighbour model.
#SBATCH --account=def-ybkim_cpu
#SBATCH --job-name=wp2_bordered
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=32 --mem=120G
#SBATCH --time=06:00:00
#SBATCH --array=0-2
#SBATCH --output=/lustre09/project/6003507/zhouzb79/twist_qsi/campaign/outputs/wp2_gate/wp2_bordered_%A_%a.log
set -uo pipefail
module purge
module load StdEnv/2023 gcc/12.3 python/3.11.5 scipy-stack/2024b
export FLEXIBLAS=AOCL
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-32}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-32}
cd /lustre09/project/6003507/zhouzb79/twist_qsi

JPM=(-0.125  0.049  0.071)
PP=(  0.085  0.128  0.038)
SEED=(NLC_A NLC_B QMC_ordered)

I=$SLURM_ARRAY_TASK_ID
echo "seed ${SEED[$I]}: jpm=${JPM[$I]} jpmpm=${PP[$I]}"
python3 -m py_compile campaign/run_masked_band_complete.py || { echo "SYNTAX FAIL"; exit 1; }
python3 -u campaign/run_masked_band_complete.py "${JPM[$I]}" \
    --jpmpm "${PP[$I]}" --full-basis
STATUS=$?
echo "exit=$STATUS  $(date)"
exit $STATUS
