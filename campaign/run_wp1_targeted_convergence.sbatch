#!/bin/bash
# Matched FCC-32 process-targeted convergence ladder at the strongest Fig. 3
# coupling.  All depths use the d=3 ground-energy anchor and identical columns.
#SBATCH --account=def-ybkim_cpu
#SBATCH --job-name=wp1_target
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=32 --mem=120G
#SBATCH --time=23:59:00
#SBATCH --array=4
#SBATCH --output=/lustre09/project/6003507/zhouzb79/twist_qsi/campaign/outputs/wp1_tomography/wp1_target_%A_%a.log
set -uo pipefail
module purge
module load StdEnv/2023 gcc/12.3 python/3.11.5 scipy-stack/2024b
export FLEXIBLAS=AOCL
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-32}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-32}
cd /lustre09/project/6003507/zhouzb79/twist_qsi

DEPTH=4
python3 -m py_compile campaign/run_wp1_targeted_convergence.py || exit 1
python3 -u campaign/run_wp1_targeted_convergence.py \
  "$DEPTH" -0.30 -3.952734742711136 \
  --ncolumns=16 --selection-pool=32 --chunk=4 --resume --compact
