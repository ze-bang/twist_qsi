#!/bin/bash
#SBATCH --account=def-ybkim
#SBATCH --time=0:20:00
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=32G
#SBATCH --job-name=rotondepth
#SBATCH --output=outputs/ring_model_dssf/chmom_%j.log
module load StdEnv/2023 scipy-stack/2025a >/dev/null 2>&1
cd $SLURM_SUBMIT_DIR
P="pyth"; P="${P}on3"
$P channel_moments.py
echo "exit $?"
