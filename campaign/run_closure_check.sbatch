#!/bin/bash
# Combinatorial check: does the hexagon ring exchange close on a polarization
# sector, and is cluster.ice_states sorted the way its consumers assume?
#SBATCH --account=def-ybkim_cpu
#SBATCH --job-name=closure
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=32000M
#SBATCH --time=01:00:00
#SBATCH --output=/lustre09/project/6003507/zhouzb79/twist_qsi/campaign/outputs/wp3_dssf/closure_%j.log
set -uo pipefail
module purge
module load StdEnv/2023 gcc/12.3 python/3.11.5 scipy-stack/2024b
export FLEXIBLAS=AOCL
cd /lustre09/project/6003507/zhouzb79/twist_qsi
python3 -u campaign/probe_dual_incidence.py
echo "exit=$?  $(date)"
