#!/bin/bash
#SBATCH --account=def-ybkim_gpu
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16 --mem=120G
#SBATCH --gpus-per-node=1
#SBATCH --time=11:00:00
#SBATCH --array=0-19
#SBATCH --job-name=xy_masked_flux
#SBATCH --output=/lustre09/project/6003507/zhouzb79/twist_qsi/campaign/outputs/xy_phase_plane/mflux_%A_%a.log
# Winding-free (Jx, Jy) plane: the masked Feshbach map at every point of the
# upper triangle, giving ice weight, band gap, band completeness and the
# hexagon flux -- the same phase diagram as the naive scan, done with the
# mask instead of raw ED.  No counterterm anywhere in this pipeline.
#
# Sizing, from the CPU pilot and from the existing plane_bands runs: the
# cost is the four Klein-blocked ~16k dense eigendecompositions, 2600 s on
# 16 CPU cores against ~300 s on an H100, so this goes to the GPU.  861
# points over 20 tasks is ~43 each, ~3.6 h, against an 11 h limit.
#
# Resume guard: a task that already wrote its file is skipped, so a timeout
# can be re-submitted without repeating work.
set -uo pipefail
module purge
module load StdEnv/2023 gcc/12.3 cuda/12.6 python/3.11.5 scipy-stack/2024b
export FLEXIBLAS=AOCL OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

VENV=$SLURM_TMPDIR/cupy-venv
python3 -m venv --system-site-packages "$VENV"
source "$VENV/bin/activate"
pip install --no-index cupy | tail -1
nvidia-smi --query-gpu=name --format=csv,noheader

cd /lustre09/project/6003507/zhouzb79/twist_qsi/campaign
# No file-exists guard here: the script checkpoints every point and resumes
# from whatever is on disk, so a resubmission continues a partial task
# instead of skipping it.
python3 -m py_compile scan_xy_masked_flux.py || { echo "SYNTAX FAIL"; exit 1; }
python3 -u scan_xy_masked_flux.py --gpu \
  outputs/xy_phase_plane/flux_points.json "$SLURM_ARRAY_TASK_ID" 20 masked_flux2
STATUS=$?
echo "exit=$STATUS  $(date)"
# Propagate the failure.  With echo as the last command the task exits 0 and
# SLURM reports COMPLETED however badly python died -- which is how twenty
# tasks "completed" in three minutes having written nothing.
exit $STATUS
