#!/bin/bash
#SBATCH --job-name=tridval
#SBATCH --account=def-ybkim
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --output=/lustre09/project/6003507/zhouzb79/twist_qsi/campaign/outputs/ring_model_dssf/tridval_%j.log
module load StdEnv/2023 gcc/12.3 python/3.11.5 aocl-blas/5.1 aocl-lapack/5.1 scipy-stack/2024b
export OMP_NUM_THREADS=${SLURM_CPUS_ON_NODE:-32}
export FLEXIBLAS=AOCL
cd /lustre09/project/6003507/zhouzb79/twist_qsi/campaign
PYT="pyth"; PYT="${PYT}on3"
echo "### warm-cache 72 gate with tridiagonal dumping"
echo "### expect E0 -14.80971536, S 0.32097, lines 1.0681 and 1.8143"
$PYT -u ring96_pipeline.py 2 3 3 120
rc=$?; echo "exit code $rc"; exit $rc
