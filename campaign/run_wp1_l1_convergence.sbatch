#!/bin/bash
# Missing first rung of the matched FCC-32 depth-convergence ladder.
# These use exactly the same first 512 ice columns as the existing d=2 and
# d=3 checks, so differences isolate virtual depth rather than sampling.
#SBATCH --account=def-ybkim_cpu
#SBATCH --job-name=wp1_l1conv
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=32 --mem=120G
#SBATCH --time=02:00:00
#SBATCH --array=0-1
#SBATCH --output=/lustre09/project/6003507/zhouzb79/twist_qsi/campaign/outputs/wp1_tomography/wp1_l1conv_%A_%a.log
set -uo pipefail
module purge
module load StdEnv/2023 gcc/12.3 python/3.11.5 scipy-stack/2024b
export FLEXIBLAS=AOCL
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-32}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-32}
cd /lustre09/project/6003507/zhouzb79/twist_qsi

JPM=(-0.20 -0.30)
THIS=${JPM[$SLURM_ARRAY_TASK_ID]}
python3 -m py_compile campaign/run_wp1_tomography.py || exit 1
python3 -u campaign/run_wp1_tomography.py --chunk=256 --columns=512 fcc32 1 "$THIS"
