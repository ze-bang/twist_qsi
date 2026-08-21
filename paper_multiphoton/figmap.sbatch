#!/bin/bash
#SBATCH --account=def-ybkim_cpu
#SBATCH --job-name=figMap
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=2 --mem=8000M
#SBATCH --time=00:10:00
#SBATCH --output=/lustre09/project/6003507/zhouzb79/twist_qsi/paper_multiphoton/fig_%j.log
set -uo pipefail
module load StdEnv/2023 gcc/12.3 python/3.11.5 scipy-stack/2024b
cd /lustre09/project/6003507/zhouzb79/twist_qsi
python3 -u campaign/make_channel_maps.py
rc=$?; echo "EXIT=$rc done $(date)"; exit $rc
