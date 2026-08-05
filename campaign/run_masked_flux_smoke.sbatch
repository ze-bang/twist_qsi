#!/bin/bash
#SBATCH --account=def-ybkim_gpu
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16 --mem=120G
#SBATCH --gpus-per-node=1
#SBATCH --time=0:40:00
#SBATCH --job-name=mflux_smoke
#SBATCH --output=/lustre09/project/6003507/zhouzb79/twist_qsi/campaign/outputs/xy_phase_plane/mfluxsmoke_%j.log
# One point, end to end, writing a file -- the check the twenty-task launch
# should have had.  Stride 861 means task 0 gets exactly one point.
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
python3 -m py_compile scan_xy_masked_flux.py || { echo "SYNTAX FAIL"; exit 1; }
python3 -u scan_xy_masked_flux.py --gpu \
  outputs/xy_phase_plane/flux_points.json 0 861 smoke_masked_flux
STATUS=$?
echo "exit=$STATUS  $(date)"
ls -l outputs/xy_phase_plane/smoke_masked_flux_000.json || echo "NO FILE WRITTEN"
exit $STATUS
