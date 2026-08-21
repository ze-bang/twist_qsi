#!/bin/bash
# WP1 truncation check at the WEAK-coupling end of the xi_loop curve.
#
# Why.  fig_loop_hierarchy panel (c) claims xi_loop GROWS with |Jpm| --
# delocalization -- across |Jpm|/Jzz = 0.03 .. 0.30, but levels=3 exists only
# at the top two points (-0.20, -0.30).  On those two, the 3-point fit
# xi = -4/ln(K12/K8) gives
#
#     |Jpm|    ell=2    ell=3    geometric extrapolation
#     0.20     1.518    1.511    1.439
#     0.30     1.615    1.565    1.427
#     growth   +6.4%    +3.6%    -0.8%
#
# i.e. the growth halves from ell=2 to ell=3 and reverses sign once the
# remaining truncation error is extrapolated away (K8 still has +35% / +43%
# to run at ell=3, and the increment ratio r = D2/D1 is LARGER at -0.30
# (0.597) than at -0.20 (0.558), so the strong-coupling end loses more xi).
# If the same correction applies at weak coupling the claimed growth may be
# entirely a truncation artifact.  These three points decide it.
#
# Method is identical to run_wp1_l3_sampled.sbatch: a levels=2 baseline on the
# SAME 512-column sample runs first, so sampling bias is common to both and
# cancels in the ell=2 -> ell=3 comparison.  Comparing a sampled levels=3
# against the full levels=2 would confound truncation with sampling.
#
# Sizing: levels=3 at -0.20 took 18,997 s (5.3 h) on this configuration, plus
# 997 s for its levels=2 baseline.  Weaker coupling is more diagonally
# dominant, so the block-CG should converge no slower.  The 12 h wall is
# roughly 2x headroom; the cluster grants no extensions.
#SBATCH --account=def-ybkim_cpu
#SBATCH --job-name=wp1_l3weak
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=64 --mem=249G
#SBATCH --time=12:00:00
#SBATCH --array=0-2
#SBATCH --output=/lustre09/project/6003507/zhouzb79/twist_qsi/campaign/outputs/wp1_tomography/wp1_l3weak_%A_%a.log
set -uo pipefail
module purge
module load StdEnv/2023 gcc/12.3 python/3.11.5 scipy-stack/2024b
export FLEXIBLAS=AOCL
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-64}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-64}
cd /lustre09/project/6003507/zhouzb79/twist_qsi

JPM=(-0.03 -0.06 -0.10)
THIS=${JPM[$SLURM_ARRAY_TASK_ID]}

echo "task=$SLURM_ARRAY_TASK_ID jpm=$THIS  start $(date)"
python3 -m py_compile campaign/run_wp1_tomography.py || { echo "SYNTAX FAIL"; exit 1; }

echo "########## levels=2, 512-column sample (comparison baseline) ##########"
python3 -u campaign/run_wp1_tomography.py --chunk=128 --columns=512 fcc32 2 "$THIS"
L2=$?

echo "########## levels=3, same 512-column sample ##########"
python3 -u campaign/run_wp1_tomography.py --resume --chunk=64 --columns=512 fcc32 3 "$THIS"
L3=$?

echo "exit_l2=$L2 exit_l3=$L3  finish $(date)"
[ "$L2" -eq 0 ] && [ "$L3" -eq 0 ]
exit $?
