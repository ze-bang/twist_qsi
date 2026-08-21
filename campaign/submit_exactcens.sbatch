#!/bin/bash
#SBATCH --job-name=excens
#SBATCH --account=def-ybkim
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --output=/lustre09/project/6003507/zhouzb79/twist_qsi/campaign/outputs/ring_model_dssf/excens_%j.log
module load StdEnv/2023 gcc/12.3 python/3.11.5 aocl-blas/5.1 aocl-lapack/5.1 scipy-stack/2024b
export OMP_NUM_THREADS=${SLURM_CPUS_ON_NODE:-32}
export FLEXIBLAS=AOCL
cd /lustre09/project/6003507/zhouzb79/twist_qsi/campaign
PYT="pyth"; PYT="${PYT}on3"
echo "######## 48 sites -- GATE: roton census must be +0.339"
$PYT -u exact_census.py 2 2 3 8
echo "######## 64 sites -- GATE: roton census must be +0.338"
$PYT -u exact_census.py 2 2 4 8
echo "######## 72 sites -- the question: is +0.134 real or unconverged?"
$PYT -u exact_census.py 2 3 3 8
rc=$?; echo "exit code $rc"; exit $rc
