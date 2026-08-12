#!/bin/bash
# WP0a, part 2: the off-axis leg of the collapse test.
#
# One column of the certified plane -- Jpm = -0.20, Jpmpm = 0.05 .. 0.30, all
# six points band-complete (90/90) in outputs/jpmpm_fold_plane/.  The column's
# Jpmpm = 0 anchor is already in the anatomy ladder at exactly -0.20, so the
# seven points form one clean "vary Jpmpm at fixed Jpm" line.
#
# What it decides: |Jpm| and Jpmpm are two different dials into the same
# dressing.  If 1 - Z is the organizing coordinate, moving Jpmpm must slide the
# loop ratios along the SAME curve the |Jpm| ladder traces.  If it traces its
# own curve, the ratios are functions of the couplings and Z is downstream.
#
# Cost: pair flips break Sz, so there is no basis smaller than the full 65,536
# states; each point is one sparse ground state plus ~10-16 block-CG solves of
# 90 right-hand sides on the 65,446-dimensional complement.  The existing plane
# scanner does 8 such solves in 15-50 min, so 4 h per task is comfortable.
#SBATCH --account=def-ybkim_cpu
#SBATCH --job-name=wp0a_plane
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=32 --mem=32G
#SBATCH --time=04:00:00
#SBATCH --array=0-5
#SBATCH --output=/lustre09/project/6003507/zhouzb79/twist_qsi/campaign/outputs/wp0_preflight/wp0a_plane_%A_%a.log
set -uo pipefail
module purge
module load StdEnv/2023 gcc/12.3 python/3.11.5 scipy-stack/2024b
export FLEXIBLAS=AOCL
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-32}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-32}
cd /lustre09/project/6003507/zhouzb79/twist_qsi

PP=(0.05 0.10 0.15 0.20 0.25 0.30)
THIS=${PP[$SLURM_ARRAY_TASK_ID]}

python3 -m py_compile campaign/run_wp0a_collapse.py || { echo "SYNTAX FAIL"; exit 1; }
python3 -u campaign/run_wp0a_collapse.py --plane -0.20 "$THIS"
STATUS=$?
echo "exit=$STATUS  $(date)"
exit $STATUS
