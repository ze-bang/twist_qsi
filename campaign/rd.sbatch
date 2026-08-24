#!/bin/bash
#SBATCH --account=def-ybkim
#SBATCH --time=1:00:00
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=32G
#SBATCH --job-name=rotondepth
#SBATCH --output=outputs/ring_model_dssf/whylow_%j.log
module load StdEnv/2023 scipy-stack/2025a >/dev/null 2>&1
cd $SLURM_SUBMIT_DIR
P="pyth"; P="${P}on3"
$P why_low_energy.py
echo "exit $?"
