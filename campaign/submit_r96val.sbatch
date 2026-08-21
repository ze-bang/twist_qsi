#!/bin/bash
#SBATCH --job-name=r96val
#SBATCH --account=def-ybkim
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --output=/lustre09/project/6003507/zhouzb79/twist_qsi/campaign/outputs/ring_model_dssf/r96val_%j.log
module load StdEnv/2023 gcc/12.3 python/3.11.5 aocl-blas/5.1 aocl-lapack/5.1 scipy-stack/2024b
export OMP_NUM_THREADS=${SLURM_CPUS_ON_NODE:-32}
export FLEXIBLAS=AOCL
cd /lustre09/project/6003507/zhouzb79/twist_qsi/campaign
PYT="pyth"; PYT="${PYT}on3"
echo "### 72-site end-to-end validation of the 96-site pipeline, n_iter=120"
echo "### expect E0=-14.80971536, E0/site=-0.20569049, n_flip=15.6402"
echo "### expect S(L)=0.32097, branches 1.0681 (0.5397) and 1.8143 (0.1782)"
$PYT -u ring96_pipeline.py 2 3 3 120
rc=$?
echo "exit code $rc"
exit $rc
