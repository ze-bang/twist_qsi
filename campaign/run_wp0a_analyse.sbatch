#!/bin/bash
# WP0a, part 3: the collapse verdict.
#
# Reads every cached anatomy npz (both signs) and the Jpmpm column, recomputes
# the loop tomography from the stored masked maps, and runs the two prediction
# legs.  Cheap -- it is 90x90 linear algebra on a few dozen files -- but it
# still goes to a compute node.
#
# Chained behind the positive ladder so the sign leg has its test points.
#SBATCH --account=def-ybkim_cpu
#SBATCH --job-name=wp0a_analyse
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/lustre09/project/6003507/zhouzb79/twist_qsi/campaign/outputs/wp0_preflight/wp0a_analyse_%j.log
set -uo pipefail
module purge
module load StdEnv/2023 gcc/12.3 python/3.11.5 scipy-stack/2024b
export FLEXIBLAS=AOCL
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
cd /lustre09/project/6003507/zhouzb79/twist_qsi
python3 -m py_compile campaign/run_wp0a_collapse.py || { echo "SYNTAX FAIL"; exit 1; }
python3 -u campaign/run_wp0a_collapse.py
STATUS=$?
echo "exit=$STATUS  $(date)"
exit $STATUS
