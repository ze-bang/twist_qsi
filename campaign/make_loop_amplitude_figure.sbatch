#!/bin/bash
#SBATCH --account=def-ybkim_cpu
#SBATCH --job-name=loopfig
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=2 --mem=8000M
#SBATCH --time=00:15:00
#SBATCH --output=/lustre09/project/6003507/zhouzb79/twist_qsi/campaign/outputs/map_normalization/loopfig_%j.log
set -uo pipefail
module load StdEnv/2023 gcc/12.3 python/3.11.5 scipy-stack/2024b
cd /lustre09/project/6003507/zhouzb79/twist_qsi
python3 -u campaign/make_loop_amplitude_figure.py
echo "EXIT=$? done $(date)"
