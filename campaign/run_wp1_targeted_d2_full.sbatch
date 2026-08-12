#!/bin/bash
# Complete-column d=2 reference at the common d=3 anchor.  This validates the
# 32-column targeted estimator used for the otherwise intractable d=4 point.
#SBATCH --account=def-ybkim_cpu
#SBATCH --job-name=wp1_targfull
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=32 --mem=120G
#SBATCH --time=08:00:00
#SBATCH --output=/lustre09/project/6003507/zhouzb79/twist_qsi/campaign/outputs/wp1_tomography/wp1_targfull_%j.log
set -uo pipefail
module purge
module load StdEnv/2023 gcc/12.3 python/3.11.5 scipy-stack/2024b
export FLEXIBLAS=AOCL
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-32}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-32}
cd /lustre09/project/6003507/zhouzb79/twist_qsi

python3 -m py_compile campaign/run_wp1_targeted_convergence.py || exit 1
python3 -u campaign/run_wp1_targeted_convergence.py \
  2 -0.30 -3.952734742711136 \
  --ncolumns=2970 --chunk=128 --resume
