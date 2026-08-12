#!/bin/bash
# WP1: recompute K_L from the stored masked maps after the averaging fix.
#
# The bug: K_L averaged every entry of a geometry class, including the ones the
# mask had set to exactly zero.  Those zeros are not small amplitudes, they are
# deleted processes -- and the survival fraction varies sharply with perimeter
# (FCC-32: 24.7% at L=6 against 6.3% at L=8), so every K_L ratio was scaled by
# the ratio of survival fractions and the "within-class spread" was measuring
# the zero/nonzero split rather than any spread in the amplitudes.
#
# What the bug does NOT touch: the survival fractions are pure combinatorics,
# identical at every truncation depth, so they cancel in level-to-level ratios.
# The truncation gate's percentages (L=1 at -70%, L=2 within 19%, L=3 within
# 1.2%) stand, as do the WP0a comparisons, which are ratios between points on
# one cluster with the same geometry classes.  What changes is the absolute
# K_L, the K_L ratios, and hence xi_gauge -- the WP1 headline.
#
# f_masked is stored per point, so this costs seconds instead of the ~50 min
# the block-CG fold took.  Chained after the production array.
#SBATCH --account=def-ybkim_cpu
#SBATCH --job-name=wp1_reanalyse
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/lustre09/project/6003507/zhouzb79/twist_qsi/campaign/outputs/wp1_tomography/wp1_reanalyse_%j.log
set -uo pipefail
module purge
module load StdEnv/2023 gcc/12.3 python/3.11.5 scipy-stack/2024b
export FLEXIBLAS=AOCL
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
cd /lustre09/project/6003507/zhouzb79/twist_qsi
python3 -m py_compile campaign/run_wp1_tomography.py || { echo "SYNTAX FAIL"; exit 1; }
echo "########## cubic16 (recheck the gate ladder) ##########"
python3 -u campaign/run_wp1_tomography.py --reanalyse cubic16
echo "########## fcc32 (the production points) ##########"
python3 -u campaign/run_wp1_tomography.py --reanalyse fcc32
STATUS=$?
echo "exit=$STATUS  $(date)"
exit $STATUS
