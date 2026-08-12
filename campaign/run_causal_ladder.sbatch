#!/bin/bash
# The causal operator ladder -- the highest-priority missing physics
# calculation for the combined manuscript (fig_causal_ladder.pdf).
#
# It decides whether the paper's central claim can be upgraded from
# "the ring-only theory fails at the OPERATOR level" to "...and that failure is
# visible in observables".  If H6 alone already reproduces the full masked map
# on the band spectrum, the ring-scale heat capacity and the longitudinal
# response, then the longer loops are inert and the claim must stay where it
# is.  The manuscript is currently written to the weaker claim, so a fast
# convergence is a legitimate outcome, not a failure of the run.
#
# Cheap by construction: f_masked is already cached per coupling by
# run_wp1_tomography.py, so no refolding happens here.  Each coupling is five
# dense 2970x2970 eigendecompositions -- seconds.  Runs all seven couplings,
# not just the two the figure needs, since the marginal cost is negligible and
# the trend across coupling is itself informative.
#
# Restartable: one JSON per coupling, skipped when present.
#SBATCH --account=def-ybkim_cpu
#SBATCH --job-name=causal_ladder
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=32 --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/lustre09/project/6003507/zhouzb79/twist_qsi/campaign/outputs/causal_ladder/ladder_%j.log
set -uo pipefail
module purge
module load StdEnv/2023 gcc/12.3 python/3.11.5 scipy-stack/2024b
export FLEXIBLAS=AOCL
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-32}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-32}
mkdir -p /lustre09/project/6003507/zhouzb79/twist_qsi/campaign/outputs/causal_ladder
cd /lustre09/project/6003507/zhouzb79/twist_qsi
python3 -m py_compile campaign/run_causal_ladder.py || { echo "SYNTAX FAIL"; exit 1; }
python3 -u campaign/run_causal_ladder.py
STATUS=$?
echo "exit=$STATUS  $(date)"
exit $STATUS
