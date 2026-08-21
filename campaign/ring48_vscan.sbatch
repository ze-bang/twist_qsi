#!/bin/bash
#SBATCH --account=def-ybkim_cpu
#SBATCH --job-name=ring48v
#SBATCH --nodes=1 --exclusive --cpus-per-task=32 --mem=0
#SBATCH --time=02:00:00
#SBATCH --output=/lustre09/project/6003507/zhouzb79/twist_qsi/campaign/outputs/ring_model_dssf/vscan_%j.log
set -uo pipefail
module load StdEnv/2023 gcc/12.3 python/3.11.5 aocl-blas/5.1 aocl-lapack/5.1 scipy-stack/2024b
cd /lustre09/project/6003507/zhouzb79/twist_qsi
export OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16
for V in -0.50 -0.25 0.25 0.50; do
  python3 -u campaign/ring48_momenta_fix.py --shape=2,2,3 --v=$V
done
rc=$?; echo "EXIT=$rc done $(date)"; exit $rc
