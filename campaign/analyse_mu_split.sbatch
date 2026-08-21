#!/bin/bash
#SBATCH --account=def-ybkim_cpu
#SBATCH --job-name=musplit
#SBATCH --nodes=1 --exclusive --cpus-per-task=32 --mem=0
#SBATCH --time=02:00:00
#SBATCH --output=/lustre09/project/6003507/zhouzb79/twist_qsi/campaign/outputs/map_normalization/musplit_%j.log
set -uo pipefail
module load StdEnv/2023 gcc/12.3 python/3.11.5 aocl-blas/5.1 aocl-lapack/5.1 scipy-stack/2024b
cd /lustre09/project/6003507/zhouzb79/twist_qsi
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
python3 -u campaign/analyse_mu_split.py -0.02 -0.03 -0.05 -0.06 -0.08 -0.10 -0.12
echo "EXIT=$? done $(date)"
