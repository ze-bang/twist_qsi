#!/bin/bash
#SBATCH --account=def-ybkim_cpu
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=48G --time=2:00:00
#SBATCH --job-name=xy_flux_pilot
#SBATCH --output=/lustre09/project/6003507/zhouzb79/twist_qsi/campaign/outputs/xy_phase_plane/xyfluxpilot_%j.log
# Validate the hexagon-flux operator against three published counterterm
# values and time two of the points that stalled the column array, before
# committing 41 tasks to the full triangle.
set -uo pipefail
module purge
module load StdEnv/2023 gcc/12.3 python/3.11.5 aocl-blas/5.1 aocl-lapack/5.1 scipy-stack/2024b
export FLEXIBLAS=AOCL OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

cd /lustre09/project/6003507/zhouzb79/twist_qsi/campaign
python3 -m py_compile scan_xy_points.py || { echo "SYNTAX FAIL"; exit 1; }
python3 -u scan_xy_points.py outputs/xy_phase_plane/pilot_points.json 0 1 pilot_flux
echo "exit=$?  $(date)"
