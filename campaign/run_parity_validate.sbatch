#!/bin/bash
#SBATCH --account=def-ybkim_gpu
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16 --mem=120G
#SBATCH --gpus-per-node=1
#SBATCH --time=1:30:00
#SBATCH --job-name=parity_val
#SBATCH --output=/lustre09/project/6003507/zhouzb79/twist_qsi/campaign/outputs/xy_phase_plane/parityval_%j.log
# Same six points twice: once on the even-parity basis (the claim) and once
# on the full basis (the previous, slower path).  Energies, gap, ice weight
# and flux must agree to machine precision -- parity is supposed to discard
# a decoupled half of the Hilbert space, not approximate anything.  Also
# times both, to confirm the predicted factor of eight.
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
echo "=== even parity ==="
python3 -u scan_xy_masked_flux.py --gpu \
  outputs/xy_phase_plane/masked_pilot_points.json 0 1 parity_even
echo "=== full basis ==="
python3 -u scan_xy_masked_flux.py --gpu --full-basis \
  outputs/xy_phase_plane/masked_pilot_points.json 0 1 parity_full
STATUS=$?
echo "exit=$STATUS  $(date)"
exit $STATUS
