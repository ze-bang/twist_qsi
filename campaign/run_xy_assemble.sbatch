#!/bin/bash
#SBATCH --account=def-ybkim_cpu
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=2 --mem=16G --time=0:30:00
#SBATCH --job-name=xy_assemble
#SBATCH --output=/lustre09/project/6003507/zhouzb79/twist_qsi/campaign/outputs/xy_phase_plane/xyassemble_%j.log
# Fold the flux solves back into the plane -- they close the last eight
# cells and retire the four-decimal log scrapes -- then redraw both figures.
set -uo pipefail
module purge
module load StdEnv/2023 gcc/12.3 python/3.11.5 aocl-blas/5.1 aocl-lapack/5.1 scipy-stack/2024b
export FLEXIBLAS=AOCL OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

cd /lustre09/project/6003507/zhouzb79/twist_qsi/campaign
for script in fill_xy_phase_gaps.py make_xy_flux_figure.py; do
  python3 -m py_compile "$script" || { echo "SYNTAX FAIL $script"; exit 1; }
done
python3 -u fill_xy_phase_gaps.py 41
echo "--- phase figure ---"
python3 -u make_xy_phase_figure.py
echo "--- flux figure ---"
python3 -u make_xy_flux_figure.py 41
echo "exit=$?  $(date)"
