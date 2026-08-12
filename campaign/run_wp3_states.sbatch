#!/bin/bash
# WP3 follow-up: is the equal weight on the four L points exact?
#
# At Jpm = -0.20 the four allowed momenta carry weight identical to 12 digits;
# at -0.05 they differ (0.273 / 0.214 / 0.257 / 0.226).  The ground state is
# non-degenerate at both, so a point-group argument would make them equal at
# both.  The likely culprit is the state count: at -0.05 the excitations are an
# order of magnitude smaller (0.0016 against 0.0157), so the band is dense on
# the scale of the coupling and the 24 states summed are a partial sum that
# captures different fractions at different momenta.
#
# If that is right, raising the count must drive the four toward equality.  If
# they stay unequal, the inequality is real and means something -- so this run
# distinguishes an artifact of my truncation from a physical asymmetry.
#
# The FORBIDDEN set is not in question either way: exact zeros at machine
# precision do not depend on how many states are summed.
#SBATCH --account=def-ybkim_cpu
#SBATCH --job-name=wp3_states
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=32 --mem=120G
#SBATCH --time=06:00:00
#SBATCH --output=/lustre09/project/6003507/zhouzb79/twist_qsi/campaign/outputs/wp3_dssf/wp3_states_%j.log
set -uo pipefail
module purge
module load StdEnv/2023 gcc/12.3 python/3.11.5 scipy-stack/2024b
export FLEXIBLAS=AOCL
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-32}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-32}
cd /lustre09/project/6003507/zhouzb79/twist_qsi
python3 -u campaign/run_wp3_dssf.py fcc32 1 -0.05 --nstates=120
python3 -u campaign/run_wp3_dssf.py fcc32 1 -0.20 --nstates=120
STATUS=$?
echo "exit=$STATUS  $(date)"
exit $STATUS
