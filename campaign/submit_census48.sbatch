#!/bin/bash
#SBATCH --job-name=cens48
#SBATCH --account=def-ybkim
#SBATCH --time=1:00:00
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16 --mem=64000M
#SBATCH --output=/lustre09/project/6003507/zhouzb79/twist_qsi/campaign/outputs/ring_model_dssf/cens48_%j.log
module load StdEnv/2023 gcc/12.3 python/3.11.5 aocl-blas/5.1 aocl-lapack/5.1 scipy-stack/2024b
export OMP_NUM_THREADS=16
export FLEXIBLAS=AOCL
cd /lustre09/project/6003507/zhouzb79/twist_qsi/campaign
PYT="pyth"; PYT="${PYT}on3"
echo "### 48-site census gate. Full diagonalization gives, at L:"
echo "###   omega 1.0235 share 0.508 census +0.339"
echo "###   omega 1.7303 share 0.323 census -0.064"
echo "### and <n_flip> ground = 10.9525, E0 = -10.55601176"
$PYT -u ring96_pipeline.py 2 2 3 200
rc=$?; echo "exit code $rc"; exit $rc
