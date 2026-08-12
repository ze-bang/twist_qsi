#!/bin/bash
#SBATCH --account=def-ybkim_cpu
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=16G --time=0:20:00
#SBATCH --job-name=xy_fig
#SBATCH --output=/lustre09/project/6003507/zhouzb79/twist_qsi/campaign/outputs/xy_phase_plane/xyfig_%j.log
set -uo pipefail
module purge
module load StdEnv/2023 gcc/12.3 python/3.11.5 scipy-stack/2024b
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MPLBACKEND=Agg
cd /lustre09/project/6003507/zhouzb79/twist_qsi/campaign
python3 -m py_compile make_xy_phase_figure.py || { echo "SYNTAX FAIL"; exit 1; }
python3 -u make_xy_phase_figure.py
echo "exit=$?  $(date)"
