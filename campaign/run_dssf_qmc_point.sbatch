#!/bin/bash
# Longitudinal S^zz(q,w) at the Huang-2018 QMC coupling |Jpm|/Jzz = 0.046.
#
# Why both signs.  campaign/data/README.md pins the external benchmark to
# Jpm/Jzz = +0.046 in the repository convention
#     H = -Jpm sum(S+S- + h.c.) + Jzz sum SzSz,
# which is the sign-free branch and the only one continuous-time worm QMC can
# reach.  That same reference (PRL 120, 167202) reports QMC-SAC *dynamic*
# structure factors, so +0.046 is the coupling at which our longitudinal
# response is actually comparable to theirs.  -0.046 is the same magnitude on
# the negative branch the rest of the WP3 scan uses (-0.05, -0.20); it is a
# sign control, not a QMC comparison, because g6 is odd in Jpm.
#
# Four tasks: {periodic, masked} x {+0.046, -0.046}, matched to the existing
# depth-1, 120-state runs so the new points drop straight into fig_fcc32_dssf.
#SBATCH --account=def-ybkim_cpu
#SBATCH --job-name=dssf_qmc_point
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=32 --mem=120G
#SBATCH --time=02:00:00
#SBATCH --array=0-3
#SBATCH --output=/lustre09/project/6003507/zhouzb79/twist_qsi/campaign/outputs/wp3_dssf/qmcpoint_%A_%a.log
set -uo pipefail
module purge
module load StdEnv/2023 gcc/12.3 python/3.11.5 scipy-stack/2024b
export FLEXIBLAS=AOCL
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-32}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-32}
cd /lustre09/project/6003507/zhouzb79/twist_qsi

METHOD=(periodic periodic masked masked)
JPM=(+0.046 -0.046 +0.046 -0.046)
THIS_METHOD=${METHOD[$SLURM_ARRAY_TASK_ID]}
THIS_JPM=${JPM[$SLURM_ARRAY_TASK_ID]}

echo "task=$SLURM_ARRAY_TASK_ID method=$THIS_METHOD jpm=$THIS_JPM  $(date)"

python3 -m py_compile campaign/run_fcc32_dssf_periodic.py campaign/run_wp3_dssf.py || exit 1

if [ "$THIS_METHOD" = "periodic" ]; then
    python3 -u campaign/run_fcc32_dssf_periodic.py "$THIS_JPM" \
        --depth=1 --nstates=120
else
    # NOTE: --nstates= needs the equals sign; "--nstates 120" is parsed as a
    # positional and silently leaves the default of 24, which is how
    # STALE_24state_*.quarantine in this directory was produced.
    python3 -u campaign/run_wp3_dssf.py fcc32 1 "$THIS_JPM" --nstates=120
fi
STATUS=$?

echo "exit=$STATUS  $(date)"
exit $STATUS
