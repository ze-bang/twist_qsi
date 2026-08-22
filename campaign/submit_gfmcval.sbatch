#!/bin/bash
#SBATCH --job-name=gfmcval
#SBATCH --account=def-ybkim
#SBATCH --time=3:00:00
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --output=/lustre09/project/6003507/zhouzb79/twist_qsi/campaign/outputs/ring_model_dssf/gfmcval_%j.log
module load StdEnv/2023 gcc/12.3 python/3.11.5 aocl-blas/5.1 aocl-lapack/5.1 scipy-stack/2024b
export OMP_NUM_THREADS=${SLURM_CPUS_ON_NODE:-32}
export FLEXIBLAS=AOCL
cd /lustre09/project/6003507/zhouzb79/twist_qsi/campaign
PYT="pyth"; PYT="${PYT}on3"
echo "### GFMC gate, 48 sites. ED truth: E0 = -10.556012, S(L) = 0.370365"
echo "### equal-amplitude values (must NOT match if forward walking works):"
echo "###   S(L) = 0.297  and  E_var = -K<nflip>_RK"
$PYT -u gfmc_ice.py 2 2 3 8000
rc=$?; echo "exit code $rc"; exit $rc
