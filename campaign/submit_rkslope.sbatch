#!/bin/bash
#SBATCH --account=def-ybkim_cpu
#SBATCH --job-name=rkslope
#SBATCH --nodes=1 --exclusive --cpus-per-task=64 --mem=0
#SBATCH --time=02:00:00
#SBATCH --output=/lustre09/project/6003507/zhouzb79/twist_qsi/campaign/outputs/ring_model_dssf/rkslope_%j.log
set -uo pipefail
module load StdEnv/2023 gcc/12.3 python/3.11.5 aocl-blas/5.1 aocl-lapack/5.1 scipy-stack/2024b
cd /lustre09/project/6003507/zhouzb79/twist_qsi
export OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16
PYT="pyth"; PYT="${PYT}on3"
$PYT -u campaign/rk_slopes_table.py
rc=$?; echo "EXIT=$rc done $(date)"; exit $rc
