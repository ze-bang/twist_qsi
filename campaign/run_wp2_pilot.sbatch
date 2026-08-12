#!/bin/bash
# WP2 pilot: the three published seeds, exactly, before any grid is launched.
#
# Why a pilot.  The plan quotes 15-50 min per cubic-16 plane point, a 3x spread
# that decides whether the WP2 grid is a 12 h campaign or a 40 h one -- and the
# seeds sit where nobody has measured the cost, because every one of the 114
# existing plane points is at NEGATIVE Jpm and two of the three seeds are at
# positive Jpm.
#
# Seeds, mapped through Jpm = -(Jb+Jc)/4Ja and Jpmpm = (Jb-Jc)/4Ja:
#   NLC_A       (0.050,  0.021,  0.004) -> (-0.125, 0.085)   deep in the liquid
#   NLC_B       (0.051,  0.008, -0.018) -> (+0.049, 0.128)   at the order boundary
#   QMC_ordered (0.046, -0.003, -0.010) -> (+0.071, 0.038)   past it, by its own name
#
# Only NLC_A is unambiguously in the regime where "dressed gauge dynamics" is
# even the right description.  Whether the instrument still returns a complete
# band at the other two is itself a WP2 result: n_found < 90 there is the
# dissolution diagnostic doing its job, not a failure of the run.
#
# Ja is a pure energy scale -- the reduced model depends only on (Jpm/Jzz,
# Jpmpm/Jzz) -- so the "full XYZ triplet" scan is this two-dimensional plane
# plus a scale factor applied at the end when converting to Kelvin.
#SBATCH --account=def-ybkim_cpu
#SBATCH --job-name=wp2_pilot
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=32 --mem=120G
#SBATCH --time=03:00:00
#SBATCH --array=0-2
#SBATCH --output=/lustre09/project/6003507/zhouzb79/twist_qsi/campaign/outputs/wp2_gate/wp2_pilot_%A_%a.log
set -uo pipefail
module purge
module load StdEnv/2023 gcc/12.3 python/3.11.5 scipy-stack/2024b
export FLEXIBLAS=AOCL
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-32}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-32}
cd /lustre09/project/6003507/zhouzb79/twist_qsi

JPM=(-0.125  0.049  0.071)
PP=(  0.085  0.128  0.038)
SEED=(NLC_A NLC_B QMC_ordered)

I=$SLURM_ARRAY_TASK_ID
echo "seed ${SEED[$I]}: jpm=${JPM[$I]} jpmpm=${PP[$I]}"
python3 -m py_compile campaign/run_jpmpm_fold_plane.py || { echo "SYNTAX FAIL"; exit 1; }
python3 -u campaign/run_jpmpm_fold_plane.py "${JPM[$I]}" "${PP[$I]}"
STATUS=$?
echo "exit=$STATUS  $(date)"
exit $STATUS
