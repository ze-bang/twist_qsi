#!/bin/bash
# Diagnostic: is the broken 4-fold momentum degeneracy at Jpm=+0.046 an
# artifact of where the fixed 120-state window falls?
#
# The four momenta of the (+-1/2,+-1/2,+-1/2) family are equivalent under the
# cubic point group, so their integrated inelastic weights must be equal.  At
# every negative coupling tested they are, to 1e-15:
#     -0.200   0.29878 x4        -0.050   0.30930 x4      -0.046  0.30982 x4
# At +0.046 -- the sign-free coupling the QMC comparison needs -- they are
#     0.29053, 0.30655, 0.29645, 0.30680    (5.6% spread)
# while the 3-fold family stays exactly degenerate.
#
# Hypothesis: total_inelastic_weight sums over the lowest `nstates` states, and
# the window edge lands inside a degenerate multiplet.  The multiplet at the
# top of the 120-state window has 6 members at +0.046 but 3 at both negative
# couplings, so a cut that splits it would break the four momenta unequally --
# each momentum puts different weight on the partially-included multiplet.
#
# Test: vary nstates.  The absolute weight legitimately grows with the window,
# but the SPREAD WITHIN the 4-fold family must be zero for any window that
# respects multiplet boundaries.  If the spread collapses at some window sizes
# and not others, the cut is the cause and the fix is to snap `nstates` to the
# next multiplet boundary.  If it persists at ~5% everywhere, the problem is in
# the masked construction at positive coupling and the QMC point cannot be
# used until it is understood.
#
# -0.046 at the largest window runs as a control: it must stay exactly
# degenerate.
#SBATCH --account=def-ybkim_cpu
#SBATCH --job-name=dssf_window
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=32 --mem=120G
#SBATCH --time=01:00:00
# SERIAL (%1) is mandatory: run_wp3_dssf.py derives its output name from
# (cluster, levels, jpm, jpmpm) only -- nstates is NOT in the tag -- so every
# +0.046 task writes the same path.  Run in parallel they race and corrupt each
# other.  That path is also the production 120-state file, which is backed up
# to window_scan/ as the n120 entry before this job is submitted.
#SBATCH --array=0-4%1
#SBATCH --output=/lustre09/project/6003507/zhouzb79/twist_qsi/campaign/outputs/wp3_dssf/window_%A_%a.log
set -uo pipefail
module purge
module load StdEnv/2023 gcc/12.3 python/3.11.5 scipy-stack/2024b
export FLEXIBLAS=AOCL
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-32}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-32}
cd /lustre09/project/6003507/zhouzb79/twist_qsi

JPM=(+0.046 +0.046 +0.046 +0.046 -0.046)
TAG=(+0p046 +0p046 +0p046 +0p046 -0p046)
NST=(96     144    168    192    192)
THIS_JPM=${JPM[$SLURM_ARRAY_TASK_ID]}
THIS_TAG=${TAG[$SLURM_ARRAY_TASK_ID]}
THIS_NST=${NST[$SLURM_ARRAY_TASK_ID]}

echo "task=$SLURM_ARRAY_TASK_ID jpm=$THIS_JPM nstates=$THIS_NST  start $(date)"

# The output tag carries only (cluster, levels, jpm, jpmpm), NOT nstates, so
# these runs would overwrite each other and the production 120-state file.
# Stage each into its own directory and move it aside afterwards.
STAGE=campaign/outputs/wp3_dssf/window_scan
mkdir -p "$STAGE"

python3 -u campaign/run_wp3_dssf.py fcc32 1 "$THIS_JPM" --nstates=$THIS_NST
STATUS=$?

SRC="campaign/outputs/wp3_dssf/dssf_fcc32_L1_${THIS_TAG}_pp+0p000.json"
if [ -f "$SRC" ]; then
    # ${TAG} here would expand to element 0 of the TAG ARRAY, not this task's
    # tag -- that mislabelled task 4 (-0.046) as +0p046 and clobbered task 3.
    mv "$SRC" "$STAGE/dssf_fcc32_L1_${THIS_TAG}_n${THIS_NST}.json"
    echo "staged -> $STAGE/dssf_fcc32_L1_${TAG}_n${THIS_NST}.json"
fi

echo "exit=$STATUS  finish $(date)"
exit $STATUS
