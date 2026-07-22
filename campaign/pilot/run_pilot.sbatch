#!/bin/bash
# Cubic-16 validation of the FCC-32 production path (sector blocking + rep-basis
# ice gather), checked against the frozen reference operators.
#SBATCH --account=def-ybkim_cpu
#SBATCH --job-name=qsi_pilot16
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=%x-%j.out

set -euo pipefail
module purge
module load StdEnv/2023 gcc/12.3 python/3.11.5 scipy-stack/2024b
export PYTHONPATH=/lustre09/project/6003507/zhouzb79/QED/python:${PYTHONPATH:-}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
cd /lustre09/project/6003507/zhouzb79/twist_qsi
echo "host=$(hostname)"
python -u campaign/pilot/pilot_cubic16_sector.py
echo "=== PILOT DONE ==="
