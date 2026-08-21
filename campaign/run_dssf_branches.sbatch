#!/bin/bash
# Count the excitations actually carrying longitudinal weight at each of the
# eight FCC-32 momenta, to decide whether the features in the path figure are
# photon branches or finite-cluster level structure.
#SBATCH --account=def-ybkim_cpu
#SBATCH --job-name=dssf_branch
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=16000M
#SBATCH --time=00:20:00
#SBATCH --output=/lustre09/project/6003507/zhouzb79/twist_qsi/campaign/outputs/wp3_dssf/branches_%j.log
set -uo pipefail
module purge
module load StdEnv/2023 gcc/12.3 python/3.11.5 scipy-stack/2024b
export FLEXIBLAS=AOCL
cd /lustre09/project/6003507/zhouzb79/twist_qsi
python3 -u campaign/analyse_dssf_branches.py --floor=0.005 campaign/outputs/wp3_dssf/window_scan/dssf_fcc32_L1_-0p200_n700.json
echo "exit=$?  $(date)"
