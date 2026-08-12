#!/bin/bash
#SBATCH --account=def-ybkim_gpu
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16 --mem=120G
#SBATCH --gpus-per-node=1
#SBATCH --time=11:00:00
#SBATCH --array=0-7
#SBATCH --job-name=plane_bands
#SBATCH --output=/lustre09/project/6003507/zhouzb79/twist_qsi/campaign/outputs/masked_band_complete/planeband_%A_%a.log
# Exact bordered per-sector bands over the whole 8x7 (Jpm, Jpmpm) plane.
# One array task per Jpm column (7 Jpmpm values each); measured ~300 s per
# cell on an H100, so a column is ~35 min and the 11 h limit is ~18x slack.
# Cells already computed are skipped by the resume guard below.
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

JPMS=(-0.05 -0.09 -0.125 -0.16 -0.20 -0.24 -0.27 -0.30)
PPS=(0.00 0.05 0.10 0.15 0.20 0.25 0.30)
JPM=${JPMS[$SLURM_ARRAY_TASK_ID]}

for PP in "${PPS[@]}"; do
  STEM=$(printf "band_%+.3f" "$JPM" | tr . p)
  if [ "$PP" = "0.00" ]; then
    TARGET="outputs/masked_band_complete/${STEM}.json"
  else
    TARGET="outputs/masked_band_complete/${STEM}_pp$(printf "%+.3f" "$PP" | tr . p).json"
  fi
  if [ -s "$TARGET" ]; then
    echo "=== skip (Jpm,Jpmpm)=($JPM,$PP): $TARGET exists ==="
    continue
  fi
  echo "=== cell (Jpm, Jpmpm) = ($JPM, $PP)  $(date) ==="
  if [ "$PP" = "0.00" ]; then
    python3 -u run_masked_band_complete.py --gpu --window 0.60 --grid 12000 "$JPM"
  else
    python3 -u run_masked_band_complete.py --gpu --jpmpm "$PP" --window 0.60 --grid 12000 "$JPM"
  fi
done
echo "exit=$?  $(date)"
