#!/bin/bash
#SBATCH --job-name=allq64
#SBATCH --account=def-ybkim
#SBATCH --time=6:00:00
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --output=/lustre09/project/6003507/zhouzb79/twist_qsi/campaign/outputs/ring_model_dssf/allq64_%j.log
module load StdEnv/2023 gcc/12.3 python/3.11.5 aocl-blas/5.1 aocl-lapack/5.1 scipy-stack/2024b
export OMP_NUM_THREADS=${SLURM_CPUS_ON_NODE:-32}
export FLEXIBLAS=AOCL
cd /lustre09/project/6003507/zhouzb79/twist_qsi/campaign
PYT="pyth"; PYT="${PYT}on3"
echo "### full momentum sweep + RK census, fcc(2,2,4), 64 sites, n_iter=200"
echo "### independent cross-check of the OLD ring64_pipeline, which gave:"
echo "###   ice 2,756,394; sector 249,180; E0 = -13.857289514"
echo "###   L class A: S=0.36914, omega 0.9545 (share 0.4998) / 1.6518 (0.3126)"
echo "###   L class B: S=0.22700, omega 1.9976 (share 0.7712)"
$PYT -u ring_allq.py 2 2 4 200
rc=$?; echo "exit code $rc"; exit $rc
