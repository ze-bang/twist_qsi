#!/bin/bash
# WP1 production: FCC-32 loop tomography at the seven plan couplings.
#
# Depth is levels=2, not the levels=1 the plan budgeted.  The cubic-16 gate
# measured K_8/K_6 at levels=1 to be 67-73% BELOW the exact answer -- the long
# loops are exactly what truncation starves, and levels=1 would have produced a
# confident, badly short xi_gauge.  levels=2 lands within 2.4-18.9%, levels=3
# within 1.2%.
#
# levels=3 on FCC-32 is 21.7M states against 2.5M at levels=2, so it is not
# affordable across seven couplings; it is run once, separately, as the
# self-convergence check (run_wp1_l3.sbatch).  That check is not optional:
# cubic-16 cannot calibrate K_10/K_12 at all, because every connected string
# longer than eight winds on sixteen sites and the mask deletes it.
#
# Sizing from the measured FCC-32 levels=1 multi-anchor cost (~70 min for eight
# anchors, i.e. ~9 min per anchor at 140k states): one anchor at 2.5M states
# scales to roughly 2.5-3 h.  8 h is the safety margin, and --resume means a
# timeout resumes from the last completed column chunk instead of restarting.
#SBATCH --account=def-ybkim_cpu
#SBATCH --job-name=wp1_fcc32
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=32 --mem=120G
#SBATCH --time=08:00:00
#SBATCH --array=0-6
#SBATCH --output=/lustre09/project/6003507/zhouzb79/twist_qsi/campaign/outputs/wp1_tomography/wp1_fcc32_%A_%a.log
set -uo pipefail
module purge
module load StdEnv/2023 gcc/12.3 python/3.11.5 scipy-stack/2024b
export FLEXIBLAS=AOCL
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-32}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-32}
cd /lustre09/project/6003507/zhouzb79/twist_qsi

JPM=(-0.03 -0.06 -0.10 -0.15 -0.20 -0.25 -0.30)
THIS=${JPM[$SLURM_ARRAY_TASK_ID]}

python3 -m py_compile campaign/run_wp1_tomography.py || { echo "SYNTAX FAIL"; exit 1; }
python3 -u campaign/run_wp1_tomography.py --resume fcc32 2 "$THIS"
STATUS=$?
echo "exit=$STATUS  $(date)"
exit $STATUS
