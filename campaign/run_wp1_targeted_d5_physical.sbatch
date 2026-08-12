#!/bin/bash
# Optional fifth physical rung.  Submit only if d=4 leaves a material depth
# drift: the d=5 space contains 264,168,874 states and requires a full-memory
# node.  Eight matched columns retain 44/34/62/18 L=6/8/10/12 partners.
#SBATCH --account=def-ybkim_cpu
#SBATCH --job-name=wp1_d5_phys
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=32 --mem=249G
#SBATCH --time=23:59:00
#SBATCH --output=/lustre09/project/6003507/zhouzb79/twist_qsi/campaign/outputs/wp1_tomography/wp1_d5_phys_%j.log
set -uo pipefail
module purge
module load StdEnv/2023 gcc/12.3 python/3.11.5 scipy-stack/2024b
export FLEXIBLAS=AOCL
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-32}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-32}
cd /lustre09/project/6003507/zhouzb79/twist_qsi

python3 -m py_compile campaign/run_wp1_targeted_convergence.py || exit 1
python3 -u campaign/run_wp1_targeted_convergence.py \
  5 -0.30 auto \
  --ncolumns=8 --selection-pool=32 --chunk=2 --resume --compact
