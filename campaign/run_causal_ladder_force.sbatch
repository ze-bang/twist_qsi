#!/bin/bash
# Re-run the causal ladder with the off-diagonal norm budget added.
#
# The first pass returned a NON-CONVERGENT ladder: at Jpm/Jzz = -0.20 the
# ring-scale heat-capacity peak errors run 45%, 41%, 49%, 50% as simple loops
# of perimeter 8, 10 and 12 are added to H6.  That is not slow convergence, it
# is no convergence, and the sequence is not even monotone.  Two explanations
# are possible and they have opposite consequences for the manuscript:
#
#   (a) the missing weight sits in PRODUCT processes -- vertex-joined and
#       disconnected sets, correctly excluded from the K_L range fit because
#       their amplitudes factorise, but collectively large because they vastly
#       outnumber the simple loops (46080 composites against 6912 simple loops
#       at L=12).  Then the simple-loop hierarchy remains the right way to
#       measure the operator RANGE, but is not a convergent effective theory
#       on its own, and the paper must say so.
#
#   (b) the simple loops do carry the norm and the failure is a cancellation
#       between rungs.  That would be a much stronger and more surprising
#       statement, and would need its own analysis.
#
# The norm budget distinguishes these directly, so it is measured rather than
# argued.  --force is required because the first-pass JSONs already exist.
#SBATCH --account=def-ybkim_cpu
#SBATCH --job-name=ladder_force
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=32 --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/lustre09/project/6003507/zhouzb79/twist_qsi/campaign/outputs/causal_ladder/ladder_force_%j.log
set -uo pipefail
module purge
module load StdEnv/2023 gcc/12.3 python/3.11.5 scipy-stack/2024b
export FLEXIBLAS=AOCL
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-32}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-32}
cd /lustre09/project/6003507/zhouzb79/twist_qsi
PY=$(command -v python3)
"$PY" -m py_compile campaign/run_causal_ladder.py || { echo "SYNTAX FAIL"; exit 1; }
"$PY" -u campaign/run_causal_ladder.py --force
STATUS=$?
echo "exit=$STATUS  $(date)"
exit $STATUS
