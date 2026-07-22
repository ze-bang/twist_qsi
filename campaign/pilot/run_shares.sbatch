#!/bin/bash
#SBATCH --account=def-ybkim_cpu
#SBATCH --job-name=fcc32_shares
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16 --mem=120G --time=00:30:00
#SBATCH --output=%x-%j.out
set -uo pipefail
module purge; module load StdEnv/2023 gcc/12.3 python/3.11.5 scipy-stack/2024b
export PYTHONPATH=/lustre09/project/6003507/zhouzb79/QED/python:${PYTHONPATH:-}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
cd /lustre09/project/6003507/zhouzb79/twist_qsi
python -u campaign/pilot/fcc32_band_shares.py
echo "=== SHARES DONE ==="
