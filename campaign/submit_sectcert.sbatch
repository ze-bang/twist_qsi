#!/bin/bash
#SBATCH --job-name=sectcert
#SBATCH --account=def-ybkim
#SBATCH --time=20:00:00
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --output=/lustre09/project/6003507/zhouzb79/twist_qsi/campaign/outputs/ring_model_dssf/sectcert_%j.log
module load StdEnv/2023 gcc/12.3 python/3.11.5 aocl-blas/5.1 aocl-lapack/5.1 scipy-stack/2024b
export OMP_NUM_THREADS=${SLURM_CPUS_ON_NODE:-32}
export FLEXIBLAS=AOCL
cd /lustre09/project/6003507/zhouzb79/twist_qsi/campaign
PYT="pyth"; PYT="${PYT}on3"
echo "######## exhaustive: 48 sites (all sectors)"
$PYT -u sector_certification.py 2 2 3 || exit 1
echo "######## exhaustive: 72 sites (all 310 sectors)"
$PYT -u sector_certification.py 2 3 3 || exit 1
echo "######## 96 sites: twelve largest candidates"
$PYT -u sector_certification.py 2 3 4 12
rc=$?; echo "exit code $rc"; exit $rc
