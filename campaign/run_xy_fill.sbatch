#!/bin/bash
#SBATCH --account=def-ybkim_cpu
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=2 --mem=16G --time=0:30:00
#SBATCH --job-name=xy_fill
#SBATCH --output=/lustre09/project/6003507/zhouzb79/twist_qsi/campaign/outputs/xy_phase_plane/xyfill_%j.log
# Reflect and log-scrape the four columns array 18332139 never wrote, then
# redraw the plane.  No eigensolves here: this is bookkeeping over data that
# already exists.  Whatever it cannot recover is listed in
# remaining_points.json for the array that follows.
set -uo pipefail
module purge
module load StdEnv/2023 gcc/12.3 python/3.11.5 aocl-blas/5.1 aocl-lapack/5.1 scipy-stack/2024b
export FLEXIBLAS=AOCL OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

cd /lustre09/project/6003507/zhouzb79/twist_qsi/campaign
python3 -m py_compile fill_xy_phase_gaps.py || { echo "SYNTAX FAIL"; exit 1; }
python3 -u fill_xy_phase_gaps.py 41
echo "--- redraw ---"
python3 -u make_xy_phase_figure.py
echo "exit=$?  $(date)"
