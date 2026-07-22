#!/bin/bash
#SBATCH --account=def-ybkim_cpu
#SBATCH --job-name=qsi_cheb
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=32G --time=01:00:00
#SBATCH --output=%x-%j.out
set -uo pipefail
module purge; module load StdEnv/2023 gcc/12.3 python/3.11.5 scipy-stack/2024b
export PYTHONPATH=/lustre09/project/6003507/zhouzb79/QED/python:${PYTHONPATH:-}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
cd /lustre09/project/6003507/zhouzb79/twist_qsi
python -u campaign/pilot/pilot_lowmem.py
echo "=== CHEBFSI PILOT DONE ==="
