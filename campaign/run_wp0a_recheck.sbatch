#!/bin/bash
# Re-verify WP0a after retiring the duplicated tomography code.
#
# run_feshbach_anatomy.py carried its own copy of the loop tomography, and that
# copy is the one that averaged in the mask's zeros.  It is now deleted and
# both drivers import qsi_campaign.tomography.  The WP0a conclusions are
# ratios between points on one cluster, so the survival-fraction constants
# cancel and nothing should move -- but "should not move" is a prediction, and
# this run checks it rather than assuming it.
#
# Also re-verifies the single-valuedness family fix: two adjacent LADDER points
# are no longer counted as independent dials, and near-zero observables (|V/g|
# never exceeds 0.022) no longer generate huge ratios out of noise.
#SBATCH --account=def-ybkim_cpu
#SBATCH --job-name=wp0a_recheck
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/lustre09/project/6003507/zhouzb79/twist_qsi/campaign/outputs/wp0_preflight/wp0a_recheck_%j.log
set -uo pipefail
module purge
module load StdEnv/2023 gcc/12.3 python/3.11.5 scipy-stack/2024b
export FLEXIBLAS=AOCL
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
cd /lustre09/project/6003507/zhouzb79/twist_qsi
python3 -m py_compile campaign/run_wp0a_collapse.py campaign/run_feshbach_anatomy.py \
  || { echo "SYNTAX FAIL"; exit 1; }
python3 -u campaign/run_wp0a_collapse.py
STATUS=$?
echo "exit=$STATUS  $(date)"
exit $STATUS
