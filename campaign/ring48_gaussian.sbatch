#!/bin/bash
#SBATCH --account=def-ybkim_cpu
#SBATCH --job-name=ring48g
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=16000M
#SBATCH --time=00:20:00
#SBATCH --output=/lustre09/project/6003507/zhouzb79/twist_qsi/campaign/outputs/ring_model_dssf/ring48g_%j.log
set -uo pipefail
module load StdEnv/2023 gcc/12.3 python/3.11.5 aocl-blas/5.1 aocl-lapack/5.1 scipy-stack/2024b
cd /lustre09/project/6003507/zhouzb79/twist_qsi
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
python3 -u campaign/ring48_gaussian.py
echo "EXIT=$? done $(date)"
