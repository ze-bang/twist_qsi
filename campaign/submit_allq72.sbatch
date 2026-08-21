#!/bin/bash
#SBATCH --job-name=allq72
#SBATCH --account=def-ybkim
#SBATCH --time=6:00:00
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --output=/lustre09/project/6003507/zhouzb79/twist_qsi/campaign/outputs/ring_model_dssf/allq72_%j.log
module load StdEnv/2023 gcc/12.3 python/3.11.5 aocl-blas/5.1 aocl-lapack/5.1 scipy-stack/2024b
export OMP_NUM_THREADS=${SLURM_CPUS_ON_NODE:-32}
export FLEXIBLAS=AOCL
cd /lustre09/project/6003507/zhouzb79/twist_qsi/campaign
PYT="pyth"; PYT="${PYT}on3"
echo "### full momentum sweep, fcc(2,3,3), 72 sites, n_iter=120"
echo "### L-point gate: S(L)=0.32097, branches 1.0681 (0.5397) / 1.8143 (0.1782)"
$PYT -u ring_allq.py 2 3 3 120
rc=$?; echo "exit code $rc"; exit $rc
