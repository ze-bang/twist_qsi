#!/bin/bash
#SBATCH --account=def-ybkim_cpu
#SBATCH --job-name=collapse
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=2 --mem=8000M
#SBATCH --time=00:10:00
#SBATCH --output=/lustre09/project/6003507/zhouzb79/twist_qsi/campaign/outputs/map_normalization/collapse_%j.log
set -uo pipefail
module load StdEnv/2023 gcc/12.3 python/3.11.5 scipy-stack/2024b
cd /lustre09/project/6003507/zhouzb79/twist_qsi
python3 -u campaign/analyse_loop_collapse.py
echo "EXIT=$? done $(date)"
