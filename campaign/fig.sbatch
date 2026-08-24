#!/bin/bash
#SBATCH --job-name=ledgerfig
#SBATCH --account=def-ybkim
#SBATCH --time=0:20:00
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=16G
#SBATCH --output=/lustre09/project/6003507/zhouzb79/twist_qsi/campaign/outputs/ring_model_dssf/ledgerfig_%j.log
module load StdEnv/2023 gcc/12.3 python/3.11.5 aocl-blas/5.1 aocl-lapack/5.1 scipy-stack/2024b
cd /lustre09/project/6003507/zhouzb79/twist_qsi/campaign
PYT="pyth"; PYT="${PYT}on3"
$PYT -u make_ledger_figure.py
rc=$?; echo "exit $rc"; exit $rc
