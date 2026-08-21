#!/bin/bash
#SBATCH --account=def-ybkim_cpu
#SBATCH --job-name=mapnorm
#SBATCH --nodes=1 --exclusive --cpus-per-task=32 --mem=0
#SBATCH --time=04:00:00
#SBATCH --output=/lustre09/project/6003507/zhouzb79/twist_qsi/campaign/outputs/map_normalization/mapnorm_%j.log
set -uo pipefail
module load StdEnv/2023 gcc/12.3 python/3.11.5 aocl-blas/5.1 aocl-lapack/5.1 scipy-stack/2024b
cd /lustre09/project/6003507/zhouzb79/twist_qsi
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
python3 -u campaign/run_map_normalization.py -0.005 -0.02 -0.05 -0.10 -0.15 -0.20 -0.30 0.046
echo "EXIT=$? done $(date)"
