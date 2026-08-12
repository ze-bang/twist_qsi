#!/bin/bash
# Matched-column physical depth-three member of the FCC-32 convergence ladder.
# The depth-three ground energy was already determined by the full-space run.
#SBATCH --account=def-ybkim_cpu
#SBATCH --job-name=wp1_d3_phys
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=32 --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/lustre09/project/6003507/zhouzb79/twist_qsi/campaign/outputs/wp1_tomography/wp1_d3_phys_%j.log
set -uo pipefail
module purge
module load StdEnv/2023 gcc/12.3 python/3.11.5 scipy-stack/2024b
export FLEXIBLAS=AOCL
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-32}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-32}
cd /lustre09/project/6003507/zhouzb79/twist_qsi

python3 -m py_compile campaign/run_wp1_targeted_convergence.py || exit 1
python3 -u campaign/run_wp1_targeted_convergence.py \
  3 -0.30 -3.952734742711136 \
  --ncolumns=16 --selection-pool=32 --chunk=8 --resume
