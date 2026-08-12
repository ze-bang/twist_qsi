#!/bin/bash
# WP2: certified fold peaks at the seeds, using the bordered bands.
#
# The bordered cross-check overturned the pilot's reading, which is exactly why
# it was run:
#
#   NLC_A       (-0.125, 0.085)  anchor 90/90 -> bordered 90/90   certified
#   QMC_ordered (+0.071, 0.038)  anchor  0/90 -> bordered 90/90   certified
#   NLC_B       (+0.049, 0.128)  anchor  0/90 -> bordered 25/90   dissolving
#
# QMC_ordered has a COMPLETE winding-free band.  Reporting "both positive seeds
# dissolve" on the anchor solve's 0/90 would have been flatly wrong, and would
# have been the load-bearing claim of a null result.
#
# NLC_B at 25/90 is a different matter: the fold there removes 25 raw levels
# and leaves 65 ice-descended ones in the ensemble alongside their
# replacements, so its peaks are NOT winding-free statements.  That point is
# reported as uncertifiable rather than as a number.
#SBATCH --account=def-ybkim_cpu
#SBATCH --job-name=wp2_foldb
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=32 --mem=120G
#SBATCH --time=03:00:00
#SBATCH --array=0-2
#SBATCH --output=/lustre09/project/6003507/zhouzb79/twist_qsi/campaign/outputs/wp2_gate/wp2_foldb_%A_%a.log
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
python3 -u campaign/run_jpmpm_fold_plane.py --bordered "${JPM[$I]}" "${PP[$I]}"
STATUS=$?
echo "exit=$STATUS  $(date)"
exit $STATUS
