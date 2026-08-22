#!/bin/bash
#SBATCH --job-name=gfmcprod
#SBATCH --account=def-ybkim
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --output=/lustre09/project/6003507/zhouzb79/twist_qsi/campaign/outputs/ring_model_dssf/gfmcprod_%j.log
module load StdEnv/2023 gcc/12.3 python/3.11.5 aocl-blas/5.1 aocl-lapack/5.1 scipy-stack/2024b
export OMP_NUM_THREADS=${SLURM_CPUS_ON_NODE:-32}
export FLEXIBLAS=AOCL
cd /lustre09/project/6003507/zhouzb79/twist_qsi/campaign
PYT="pyth"; PYT="${PYT}on3"
echo "######## GATE 1: 48 sites (E0 -10.556012, S(L)=0.370365 at both L momenta)"
$PYT -u gfmc_ice.py 2 2 3 40000 || exit 1
echo "######## GATE 2: 72 sites via the path script (E0/site -0.205690, S at (.5,.5,.5)... path L)"
$PYT -u gfmc_production.py 2 3 3 30000 || exit 1
echo "######## PRODUCTION: (4,4,4) = 256 sites"
$PYT -u gfmc_production.py 4 4 4 30000 || exit 1
echo "######## PRODUCTION: (6,6,6) = 864 sites"
$PYT -u gfmc_production.py 6 6 6 30000 || exit 1
echo "######## PRODUCTION: (8,8,8) = 2048 sites"
$PYT -u gfmc_production.py 8 8 8 30000
rc=$?; echo "exit code $rc"; exit $rc
