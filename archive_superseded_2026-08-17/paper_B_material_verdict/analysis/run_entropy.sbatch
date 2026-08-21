#!/bin/bash
#SBATCH --account=def-ybkim_cpu
#SBATCH --job-name=entropyB
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=2000M
#SBATCH --time=00:05:00
#SBATCH --output=/lustre09/project/6003507/zhouzb79/twist_qsi/paper_B_material_verdict/analysis/entropy_%j.log
set -uo pipefail
cd /lustre09/project/6003507/zhouzb79/twist_qsi/paper_B_material_verdict
python3 analysis/entropy_budget.py
echo "done $(date)"
