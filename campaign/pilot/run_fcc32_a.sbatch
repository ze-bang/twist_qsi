#!/bin/bash
#SBATCH --account=def-ybkim_cpu
#SBATCH --job-name=fcc32_a
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=32 --mem=249G --time=03:00:00
#SBATCH --output=%x-%j.out
set -uo pipefail
module purge
module load StdEnv/2023 gcc/12.3 python/3.11.5 scipy-stack/2024b
export PYTHONPATH=/lustre09/project/6003507/zhouzb79/QED/python:${PYTHONPATH:-}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-32}
export ED_SYM_CACHE_DIR=/lustre09/project/6003507/zhouzb79/twist_qsi/campaign/cache/fcc32_sym
mkdir -p "$ED_SYM_CACHE_DIR"
cd /lustre09/project/6003507/zhouzb79/twist_qsi
echo "host=$(hostname) cpus=${SLURM_CPUS_PER_TASK} mem=${SLURM_MEM_PER_NODE}M"
python -u campaign/pilot/pilot_fcc32_stage_a.py
echo "=== STAGE A DONE ==="
