#!/bin/bash
#SBATCH --account=def-ybkim_cpu
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=48G --time=4:00:00
#SBATCH --array=0-19
#SBATCH --job-name=xy_raw_diag
#SBATCH --output=/lustre09/project/6003507/zhouzb79/twist_qsi/campaign/outputs/xy_phase_plane/rawdiag_%A_%a.log
# Periodic (naive) ED over the upper triangle again, now carrying the
# normalised flux Phi = <sum O_h>/<sum N_h> -- directly comparable with the
# masked scan -- and the "what is it instead" order parameters: <M_z^2>
# (zero for every ice state, saturated by all-in--all-out), the
# Penrose-Onsager condensate fraction (XY / spinon-condensed order), and
# the energy split into Ising, exchange and pair-flip terms.
#
# The order parameters cost a 16x16 one-body density matrix per point, so
# points are ~100 s rather than ~10 s; 861 over 20 tasks is ~1.2 h against
# a 4 h limit.  Checkpointed per point.
set -uo pipefail
module purge
module load StdEnv/2023 gcc/12.3 python/3.11.5 aocl-blas/5.1 aocl-lapack/5.1 scipy-stack/2024b
export FLEXIBLAS=AOCL OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

cd /lustre09/project/6003507/zhouzb79/twist_qsi/campaign
python3 -m py_compile scan_xy_points.py || { echo "SYNTAX FAIL"; exit 1; }
python3 -u scan_xy_points.py \
  outputs/xy_phase_plane/flux_points.json "$SLURM_ARRAY_TASK_ID" 20 raw_diag
STATUS=$?
echo "exit=$STATUS  $(date)"
exit $STATUS
