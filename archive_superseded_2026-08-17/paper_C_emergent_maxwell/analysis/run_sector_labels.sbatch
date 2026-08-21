#!/bin/bash
#SBATCH --account=def-ybkim_cpu
#SBATCH --job-name=sectorC
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=16000M
#SBATCH --time=00:30:00
#SBATCH --output=/lustre09/project/6003507/zhouzb79/twist_qsi/paper_C_emergent_maxwell/analysis/sectors_%j.log
set -uo pipefail
cd /lustre09/project/6003507/zhouzb79/twist_qsi/paper_C_emergent_maxwell
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
python3 analysis/sector_labels.py
echo "done $(date)"
