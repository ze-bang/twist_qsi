#!/bin/bash
#SBATCH --account=def-ybkim_gpu
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16 --mem=120G
#SBATCH --gpus-per-node=1
#SBATCH --time=1:00:00
#SBATCH --job-name=edge_check
#SBATCH --output=/lustre09/project/6003507/zhouzb79/twist_qsi/campaign/outputs/xy_phase_plane/edgecheck_%j.log
set -uo pipefail
module purge
module load StdEnv/2023 gcc/12.3 cuda/12.6 python/3.11.5 scipy-stack/2024b
export FLEXIBLAS=AOCL OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
VENV=$SLURM_TMPDIR/cupy-venv
python3 -m venv --system-site-packages "$VENV"
source "$VENV/bin/activate"
pip install --no-index cupy | tail -1
cd /lustre09/project/6003507/zhouzb79/twist_qsi/campaign
python3 -m py_compile check_continuum_edge.py || { echo "SYNTAX FAIL"; exit 1; }
python3 -u check_continuum_edge.py --gpu
STATUS=$?
echo "exit=$STATUS  $(date)"
exit $STATUS
