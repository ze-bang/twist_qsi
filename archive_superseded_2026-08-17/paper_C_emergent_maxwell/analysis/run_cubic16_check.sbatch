#!/bin/bash
#SBATCH --account=def-ybkim_cpu
#SBATCH --job-name=cub16sec
#SBATCH --nodes=1 --exclusive --cpus-per-task=32 --mem=0
#SBATCH --time=02:00:00
#SBATCH --output=/lustre09/project/6003507/zhouzb79/twist_qsi/paper_C_emergent_maxwell/analysis/cubic16_%j.log
set -uo pipefail
module load StdEnv/2023 gcc/12.3 python/3.11.5 aocl-blas/5.1 aocl-lapack/5.1 scipy-stack/2024b
cd /lustre09/project/6003507/zhouzb79/twist_qsi/paper_C_emergent_maxwell
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
python3 -u analysis/cubic16_sector_check.py
echo "EXIT=$? done $(date)"
