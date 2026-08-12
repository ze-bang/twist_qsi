#!/bin/bash
# Smoke test: GPU device lane + low-memory ChebFSI together, one outer
# iteration on the SMALLEST sector. Never been run in combination.
#SBATCH --account=def-ybkim_gpu
#SBATCH --job-name=fcc32_smoke
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16 --mem=440G
#SBATCH --gpus-per-node=1
#SBATCH --time=03:00:00
#SBATCH --output=%x-%j.out
set -uo pipefail
module purge
module load StdEnv/2023 gcc/12.3 openmpi/4.1.5 aocl-blas/5.1 aocl-lapack/5.1 \
            scalapack/2.2.0 cmake eigen arpack-ng hdf5-mpi cuda/12.6 \
            python/3.11.5 scipy-stack/2024b
export PYTHONPATH=/lustre09/project/6003507/zhouzb79/QED/python_gpu:${PYTHONPATH:-}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
cd /lustre09/project/6003507/zhouzb79/twist_qsi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -u campaign/pilot/fcc32_band_production.py --sector 3 --outer 1 --degree 4 \
    --outdir /lustre09/project/6003507/zhouzb79/twist_qsi/campaign/outputs/fcc32_smoke
echo "=== SMOKE DONE ==="
