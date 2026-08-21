#!/bin/bash
#SBATCH --account=def-ybkim_cpu
#SBATCH --job-name=ringdssf
#SBATCH --nodes=1 --exclusive --cpus-per-task=32 --mem=0
#SBATCH --time=01:00:00
#SBATCH --output=/lustre09/project/6003507/zhouzb79/twist_qsi/campaign/outputs/ring_model_dssf/ringdssf_%j.log
set -uo pipefail
module load StdEnv/2023 gcc/12.3 python/3.11.5 aocl-blas/5.1 aocl-lapack/5.1 scipy-stack/2024b
cd /lustre09/project/6003507/zhouzb79/twist_qsi
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
for V in 0.0 -0.5 0.5; do
  python3 -u campaign/run_ring_model_dssf.py --cluster=fcc32 --v=$V
done
echo "EXIT=$? done $(date)"
