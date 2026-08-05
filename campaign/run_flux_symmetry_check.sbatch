#!/bin/bash
#SBATCH --account=def-ybkim_cpu
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=48G --time=1:00:00
#SBATCH --job-name=flux_symmetry
#SBATCH --output=/lustre09/project/6003507/zhouzb79/twist_qsi/campaign/outputs/xy_phase_plane/fluxsym_%j.log
# Solve both sides of six mirror pairs and compare: the only honest test of
# the reflection that halves the flux map.
set -uo pipefail
module purge
module load StdEnv/2023 gcc/12.3 python/3.11.5 aocl-blas/5.1 aocl-lapack/5.1 scipy-stack/2024b
export FLEXIBLAS=AOCL OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

cd /lustre09/project/6003507/zhouzb79/twist_qsi/campaign
python3 -m py_compile check_flux_symmetry.py || { echo "SYNTAX FAIL"; exit 1; }
python3 -u check_flux_symmetry.py
echo "exit=$?  $(date)"
