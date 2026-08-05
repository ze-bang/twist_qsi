#!/bin/bash
#SBATCH --account=def-ybkim_cpu
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=48G --time=3:00:00
#SBATCH --array=0-40
#SBATCH --job-name=xy_plane
#SBATCH --output=/lustre09/project/6003507/zhouzb79/twist_qsi/campaign/outputs/xy_phase_plane/xyscan_%A_%a.log
# Ground-state map over Jx, Jy in [-1,1] at Jz = 1, exact 65,536-state
# spectrum, one array task per Jx column (41 x 41 grid).  The Hamiltonian
# is linear in the couplings at theta = 0, so each task assembles the three
# pieces once and then only recombines them per point.
set -uo pipefail
module purge
module load StdEnv/2023 gcc/12.3 python/3.11.5 aocl-blas/5.1 aocl-lapack/5.1 scipy-stack/2024b
export FLEXIBLAS=AOCL OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

cd /lustre09/project/6003507/zhouzb79/twist_qsi/campaign
mkdir -p outputs/xy_phase_plane
python3 -m py_compile scan_xy_phase_plane.py || { echo "SYNTAX FAIL"; exit 1; }
python3 -u scan_xy_phase_plane.py 41 "$SLURM_ARRAY_TASK_ID"
echo "exit=$?  $(date)"
