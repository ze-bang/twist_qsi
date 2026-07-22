#!/bin/bash
#SBATCH --account=def-ybkim_cpu
#SBATCH --job-name=qsi_merge_ver
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=32G --time=00:40:00
#SBATCH --output=%x-%j.out
set -uo pipefail
module purge; module load StdEnv/2023 gcc/12.3 python/3.11.5 scipy-stack/2024b
cd /lustre09/project/6003507/zhouzb79/twist_qsi
export PYTHONPATH=src:${PYTHONPATH:-}
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
echo "### pytest ###"
python -m pytest -q 2>&1 | tail -5
echo "### driver ###"
python -u campaign/run_nonperturbative.py --max-grid 4 2>&1 | grep -E "reduced by|orbit [0-9]+/|character_convergence|\"error\"|leakage" | head -20
