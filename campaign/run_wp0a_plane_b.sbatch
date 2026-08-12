#!/bin/bash
# WP0a, part 2b: a SECOND Jpmpm column, at Jpm = -0.125 (the NLC_A row).
#
# Why a second column.  The first column reaches 1-Z = 0.67, but the Jpm ladder
# it must be compared against is band-complete only to |Jpm| = 0.22, i.e.
# 1-Z = 0.544: at -0.24 and beyond the cubic-16 band loses roots (89, 89, 77,
# 77, 71) and those points are not certified winding-free statements.  So four
# of the first column's six points can only be compared by extrapolating the
# ladder into its own dissolved region, and the collapse test quietly dropped
# them -- reporting "Z collapses" on the strength of the two points nearest
# Jpmpm = 0, where agreement is nearly automatic.
#
# Two columns at different fixed Jpm fix this without leaving certified data.
# Both stay band-complete to pp = 0.30 (they are on the certified plane grid),
# and they overlap in 1-Z while differing in Jpm.  If the loop ratios collapse
# onto one curve in 1-Z, Z organizes; if each column traces its own, the
# couplings do.  No extrapolation anywhere.
#
# Note in passing: Jpmpm drives Z down to 0.33 while KEEPING all 90 roots,
# where |Jpm| loses roots by Z = 0.44.  The two dials are not equivalent, which
# is itself evidence against a single-coordinate collapse.
#SBATCH --account=def-ybkim_cpu
#SBATCH --job-name=wp0a_planeb
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=32 --mem=32G
#SBATCH --time=04:00:00
#SBATCH --array=0-5
#SBATCH --output=/lustre09/project/6003507/zhouzb79/twist_qsi/campaign/outputs/wp0_preflight/wp0a_planeb_%A_%a.log
set -uo pipefail
module purge
module load StdEnv/2023 gcc/12.3 python/3.11.5 scipy-stack/2024b
export FLEXIBLAS=AOCL
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-32}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-32}
cd /lustre09/project/6003507/zhouzb79/twist_qsi

PP=(0.05 0.10 0.15 0.20 0.25 0.30)
THIS=${PP[$SLURM_ARRAY_TASK_ID]}

python3 -m py_compile campaign/run_wp0a_collapse.py || { echo "SYNTAX FAIL"; exit 1; }
python3 -u campaign/run_wp0a_collapse.py --plane -0.125 "$THIS"
STATUS=$?
echo "exit=$STATUS  $(date)"
exit $STATUS
