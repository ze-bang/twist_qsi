#!/bin/bash
# Which cubic-16 X level carries the longitudinal weight, at depth 1 so it can
# be compared with the depth-1 FCC-32 numbers without confounding truncation
# with system size.
#SBATCH --account=def-ybkim_cpu
#SBATCH --job-name=br16
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=16000M
#SBATCH --time=00:20:00
#SBATCH --output=/lustre09/project/6003507/zhouzb79/twist_qsi/campaign/outputs/wp3_dssf/br16_%j.log
set -uo pipefail
module purge
module load StdEnv/2023 gcc/12.3 python/3.11.5 scipy-stack/2024b
export FLEXIBLAS=AOCL
cd /lustre09/project/6003507/zhouzb79/twist_qsi
python3 -u campaign/analyse_dssf_branches.py --floor=0.02 \
  campaign/outputs/wp3_dssf/dssf_cubic16_L1_-0p200_pp+0p000.json \
  campaign/outputs/wp3_dssf/dssf_cubic16_L1_+0p046_pp+0p000.json
echo "exit=$?  $(date)"
