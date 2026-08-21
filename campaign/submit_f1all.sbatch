#!/bin/bash
#SBATCH --job-name=f1all
#SBATCH --account=def-ybkim
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --output=/lustre09/project/6003507/zhouzb79/twist_qsi/campaign/outputs/ring_model_dssf/f1all_%j.log
module load StdEnv/2023 gcc/12.3 python/3.11.5 aocl-blas/5.1 aocl-lapack/5.1 scipy-stack/2024b
export OMP_NUM_THREADS=${SLURM_CPUS_ON_NODE:-32}
export FLEXIBLAS=AOCL
cd /lustre09/project/6003507/zhouzb79/twist_qsi/campaign
PYT="pyth"; PYT="${PYT}on3"
for S in "2 2 3" "2 2 4" "2 3 3"; do
  echo "######## fcc($S)"
  $PYT -u f1_all_momenta.py $S
done
rc=$?; echo "exit code $rc"; exit $rc
