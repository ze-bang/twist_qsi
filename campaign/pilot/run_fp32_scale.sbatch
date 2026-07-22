#!/bin/bash
#SBATCH --account=def-ybkim_cpu
#SBATCH --job-name=fp32_scale
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16 --mem=200G --time=00:40:00
#SBATCH --output=%x-%j.out
set -uo pipefail
module purge; module load StdEnv/2023 gcc/12.3 python/3.11.5 scipy-stack/2024b
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
cd /lustre09/project/6003507/zhouzb79/twist_qsi
python -u campaign/pilot/fp32_scale_test.py
echo "=== FP32 SCALE DONE ==="
