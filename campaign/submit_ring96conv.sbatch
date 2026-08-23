#!/bin/bash
#SBATCH --job-name=r96conv
#SBATCH --account=def-ybkim
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --output=/lustre09/project/6003507/zhouzb79/twist_qsi/campaign/outputs/ring_model_dssf/ring96conv_%j.log
module load StdEnv/2023 gcc/12.3 python/3.11.5 aocl-blas/5.1 aocl-lapack/5.1 scipy-stack/2024b
export OMP_NUM_THREADS=${SLURM_CPUS_ON_NODE:-32}
export FLEXIBLAS=AOCL
cd /lustre09/project/6003507/zhouzb79/twist_qsi/campaign
PYT="pyth"; PYT="${PYT}on3"
echo "### fcc(2,3,4), 96 sites, n_iter=120"
echo "### node: $(hostname), memory:"; free -g | head -2
$PYT -u ring96_pipeline.py 2 3 4 200
rc=$?
echo "exit code $rc"
exit $rc
