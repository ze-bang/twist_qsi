#!/bin/bash
#SBATCH --account=def-ybkim_cpu
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=32G --time=0:25:00
#SBATCH --job-name=crossings
#SBATCH --output=/lustre09/project/6003507/zhouzb79/twist_qsi/campaign/outputs/crossings_%j.log
set -uo pipefail
module purge
module load StdEnv/2023 gcc/12.3 python/3.11.5 scipy-stack/2024b
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
cd /lustre09/project/6003507/zhouzb79/twist_qsi/campaign
python3 -m py_compile check_plane_crossings.py || { echo "SYNTAX FAIL"; exit 1; }
python3 -u check_plane_crossings.py
echo "exit=$?  $(date)"
