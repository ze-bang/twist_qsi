#!/bin/bash
#SBATCH --account=def-ybkim_gpu
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16 --mem=120G
#SBATCH --gpus-per-node=1
#SBATCH --time=11:00:00
#SBATCH --array=0-7
#SBATCH --job-name=plane_fold
#SBATCH --output=/lustre09/project/6003507/zhouzb79/twist_qsi/campaign/outputs/jpmpm_fold_plane/foldb_%A_%a.log
# Certified C(T) peaks: the two-sided fold rebuilt on the COMPLETE bordered
# band, so all 90 ice-descended raw levels are removed and replaced.  The
# old scan removed only n_found of them, leaving 90-n_found contaminating
# levels wherever the anchor solve lost the band (63 of them at the
# (-0.125, 0.30) cell whose 0.332 ratio this is meant to certify).
# Depends on run_plane_bands_gpu.sbatch having written the bands.
set -uo pipefail
module purge
module load StdEnv/2023 gcc/12.3 cuda/12.6 python/3.11.5 scipy-stack/2024b
export FLEXIBLAS=AOCL OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

VENV=$SLURM_TMPDIR/cupy-venv
python3 -m venv --system-site-packages "$VENV"
source "$VENV/bin/activate"
pip install --no-index cupy | tail -1

cd /lustre09/project/6003507/zhouzb79/twist_qsi/campaign
python3 -m py_compile run_jpmpm_fold_plane.py || { echo "SYNTAX FAIL"; exit 1; }

JPMS=(-0.05 -0.09 -0.125 -0.16 -0.20 -0.24 -0.27 -0.30)
PPS=(0.00 0.05 0.10 0.15 0.20 0.25 0.30)
JPM=${JPMS[$SLURM_ARRAY_TASK_ID]}

# One cross-checked cell per column: runs the anchor solve too and reports
# its deviation from the bordered band, so every column carries its own
# independent verification rather than trusting the gates alone.
for PP in "${PPS[@]}"; do
  TAG=$(printf "point_jpm%+.4f_pp%+.4f_bordered" "$JPM" "$PP" | tr . p).json
  if [ -s "outputs/jpmpm_fold_plane/$TAG" ]; then
    echo "=== skip ($JPM,$PP): exists ==="
    continue
  fi
  EXTRA=""
  if [ "$PP" = "0.00" ]; then EXTRA="--cross-check"; fi
  echo "=== fold cell (Jpm, Jpmpm) = ($JPM, $PP)  $(date) ==="
  python3 -u run_jpmpm_fold_plane.py --bordered --gpu $EXTRA "$JPM" "$PP"
done
echo "exit=$?  $(date)"
