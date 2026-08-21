#!/bin/bash
#SBATCH --job-name=aggL
#SBATCH --account=def-ybkim
#SBATCH --time=0:10:00
#SBATCH --ntasks=1 --cpus-per-task=2 --mem=8000M
#SBATCH --output=/lustre09/project/6003507/zhouzb79/twist_qsi/campaign/outputs/ring_model_dssf/aggL_%j.log
module load StdEnv/2023 gcc/12.3 python/3.11.5 scipy-stack/2024b
cd /lustre09/project/6003507/zhouzb79/twist_qsi/campaign
PYT="pyth"; PYT="${PYT}on3"
$PYT -u aggregate_L_branches.py 233
rc=$?; echo "exit $rc"; exit $rc
