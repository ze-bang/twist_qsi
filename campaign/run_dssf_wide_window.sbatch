#!/bin/bash
# Test the multi-photon reading of the second line at L.
#
# At Jpm/Jzz = -0.200 the masked response has, at L, a dominant line at
# omega = 0.01568 (88.3%) and a second at 0.05041 (9.8%), ratio 3.21.  At X it
# has a SINGLE line at 0.03793 (99.9%) and no second feature.
#
# If the second line is the first allowed multi-photon threshold (three, not
# two: S^z = E is odd under the emergent charge conjugation), then X has no
# second feature only because 3*omega(X) = 0.114 lies ABOVE the 120-state
# window, whose top is omega = 0.0593.  Widening the window must then produce a
# feature near 0.114 at X carrying roughly the same ~10%.
#
# If no such feature appears once the window covers it, the multi-photon
# reading is wrong and the L second line is something else -- which is the more
# likely outcome given that its weight is nearly coupling-independent (9.2% at
# |Jpm|=0.046 against 9.8% at 0.200, across a 4.3x change in coupling).
#
# Staged out of the way: run_wp3_dssf.py derives its filename from
# (cluster, levels, jpm, jpmpm) with no state count, so it would otherwise
# overwrite the production 120-state file.
#SBATCH --account=def-ybkim_cpu
#SBATCH --job-name=dssf_wide
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=32 --mem=160G
#SBATCH --time=06:00:00
#SBATCH --output=/lustre09/project/6003507/zhouzb79/twist_qsi/campaign/outputs/wp3_dssf/wide_%j.log
set -uo pipefail
module purge
module load StdEnv/2023 gcc/12.3 python/3.11.5 scipy-stack/2024b
export FLEXIBLAS=AOCL
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-32}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-32}
cd /lustre09/project/6003507/zhouzb79/twist_qsi

STAGE=campaign/outputs/wp3_dssf/window_scan
PROD=campaign/outputs/wp3_dssf/dssf_fcc32_L1_-0p200_pp+0p000.json
mkdir -p "$STAGE"
[ -f "$PROD" ] && cp "$PROD" "$STAGE/dssf_fcc32_L1_-0p200_n120.json"

NST=700
echo "masked -0.200 at nstates=$NST  start $(date)"
python3 -u campaign/run_wp3_dssf.py fcc32 1 -0.200 --nstates=$NST
STATUS=$?

if [ -f "$PROD" ]; then
    mv "$PROD" "$STAGE/dssf_fcc32_L1_-0p200_n${NST}.json"
    echo "staged -> $STAGE/dssf_fcc32_L1_-0p200_n${NST}.json"
    # put the production 120-state file back where the figures expect it
    cp "$STAGE/dssf_fcc32_L1_-0p200_n120.json" "$PROD"
fi

echo "exit=$STATUS  finish $(date)"
exit $STATUS
