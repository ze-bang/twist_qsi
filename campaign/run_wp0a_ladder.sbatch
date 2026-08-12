#!/bin/bash
# WP0a, part 1: the POSITIVE-Jpm anatomy ladder, restricted to the spin liquid.
#
# Scope note (why this is five points and not the eight originally planned):
# the positive-Jpm side is expected to order beyond Jpm/Jzz ~ 0.05, so an
# anatomy point at +0.10 or +0.20 measures the loop structure of a map whose
# band is not a gauge theory at all.  The comparison is therefore confined to
# +0.02 .. +0.06, where the liquid is expected on BOTH signs and the matched
# -|Jpm| points already exist in outputs/feshbach_anatomy/.  That is a short
# lever on Z; the longer one is the Jpmpm column of run_wp0a_plane.sbatch,
# which moves Z at fixed Jpm without ever leaving the cerium side.
#
# +0.10 is kept as a single deliberate boundary probe: if the instrument is
# honest it should start losing roots (n_masked < 90) as the liquid gives way.
# It is excluded from the collapse fit and is not a physics claim.
#
# The oracle is left ON: positive Jpm is a regime this instrument has only ever
# been run at once (+0.046), and the exact-spectrum check costs little next to
# the QHQ eigendecomposition that dominates.
#
# Free consistency check to watch in the log: the winding-free flux should
# come out near +1/4 (0-flux) for Jpm > 0, against -1/4 on the cerium side.
#SBATCH --account=def-ybkim_cpu
#SBATCH --job-name=wp0a_ladder
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=32 --exclusive
#SBATCH --time=02:00:00
#SBATCH --output=/lustre09/project/6003507/zhouzb79/twist_qsi/campaign/outputs/wp0_preflight/wp0a_ladder_%j.log
set -uo pipefail
module purge
module load StdEnv/2023 gcc/12.3 python/3.11.5 scipy-stack/2024b
export FLEXIBLAS=AOCL
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-32}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-32}
cd /lustre09/project/6003507/zhouzb79/twist_qsi
python3 -m py_compile campaign/run_feshbach_anatomy.py || { echo "SYNTAX FAIL"; exit 1; }
python3 -u campaign/run_feshbach_anatomy.py \
    0.02 0.03 0.04 0.05 0.06 0.10
STATUS=$?
echo "exit=$STATUS  $(date)"
exit $STATUS
