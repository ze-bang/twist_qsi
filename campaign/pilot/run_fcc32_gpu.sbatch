#!/bin/bash
#SBATCH --account=def-ybkim_gpu
#SBATCH --job-name=fcc32_gpu
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16 --mem=120G
#SBATCH --gpus-per-node=1
#SBATCH --time=01:00:00
#SBATCH --output=%x-%j.out
set -uo pipefail
module purge
module load StdEnv/2023 gcc/12.3 openmpi/4.1.5 aocl-blas/5.1 aocl-lapack/5.1 \
            scalapack/2.2.0 cmake eigen arpack-ng hdf5-mpi cuda/12.6 \
            python/3.11.5 scipy-stack/2024b
export PYTHONPATH=/lustre09/project/6003507/zhouzb79/QED/python:${PYTHONPATH:-}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
cd /lustre09/project/6003507/zhouzb79/twist_qsi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -u campaign/pilot/pilot_fcc32_gpu.py
echo "=== GPU BENCH DONE ==="
