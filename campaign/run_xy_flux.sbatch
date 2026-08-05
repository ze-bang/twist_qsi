#!/bin/bash
#SBATCH --account=def-ybkim_cpu
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=48G --time=3:00:00
#SBATCH --array=8-40
# NOTE the stride argument below must match the array width: tasks take
# points[id::stride].  The first submission ran array 0-7 while still
# passing stride 41, so it covered 168 of the 861 points and silently left
# the rest unassigned.  Tasks 0-7 of stride 41 are already done, so this
# resubmission covers 8-40 and completes the triangle without redoing them.
#SBATCH --job-name=xy_flux
#SBATCH --output=/lustre09/project/6003507/zhouzb79/twist_qsi/campaign/outputs/xy_phase_plane/xyflux_%A_%a.log
# Hexagon flux over the (Jx, Jy) square: <O_hex + h.c.> in the ground state,
# whose sign is the 0-flux / pi-flux label.  Only the upper triangle is
# solved -- Jx <-> Jy is exact and leaves the hexagon operator invariant
# (three S+ and three S- pick up (i)^3(-i)^3 = 1 under the pi/2 rotation
# that sends Jpmpm -> -Jpmpm) -- so 861 solves cover all 1681 cells.
#
# Points are handed out round-robin, so every task gets a mix of the cheap
# outskirts and the expensive near-critical strip rather than one task
# inheriting all the hard ones.  The pilot settled the sizing: with the
# Krylov fix every point costs 6-11 s, including the two that ran the old
# array into the three-hour wall, so eight tasks of ~108 points each finish
# inside twenty minutes and the three-hour request is already generous.
#
# NOTE what this measures: the flux of the RAW cluster ground state.  On
# cubic-16 the wrap-around four-loop artefact is larger than the physical
# ring exchange, so this is not the winding-free flux -- see the pilot.
set -uo pipefail
module purge
module load StdEnv/2023 gcc/12.3 python/3.11.5 aocl-blas/5.1 aocl-lapack/5.1 scipy-stack/2024b
export FLEXIBLAS=AOCL OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

cd /lustre09/project/6003507/zhouzb79/twist_qsi/campaign
python3 -m py_compile scan_xy_points.py || { echo "SYNTAX FAIL"; exit 1; }
python3 -u scan_xy_points.py \
  outputs/xy_phase_plane/flux_points.json "$SLURM_ARRAY_TASK_ID" 41 flux_points
echo "exit=$?  $(date)"
