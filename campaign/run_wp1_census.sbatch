#!/bin/bash
# WP1 feasibility census, run BEFORE any FCC-32 production time.
#
# The cubic-16 gate turned up something the plan did not anticipate: K_4, K_10
# and K_12 are identically zero there, because on sixteen sites every connected
# flipped string longer than eight has to wrap, and the mask deletes wrapping
# processes by construction.  Only L = 6 and L = 8 survive, so "K_L ~ A
# exp(-(L-6)/xi)" on cubic-16 is a two-point estimate through two points -- not
# a fit, and not a gauge range.
#
# This census asks the same question of FCC-32 combinatorially, in seconds,
# instead of discovering it after a production run: how many perimeters have
# connected classes that survive the mask?  Three or more and WP1 is
# measurable; fewer and the headline needs a bigger cluster or a different
# observable.
#SBATCH --account=def-ybkim_cpu
#SBATCH --job-name=wp1_census
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/lustre09/project/6003507/zhouzb79/twist_qsi/campaign/outputs/wp1_tomography/wp1_census_%j.log
set -uo pipefail
module purge
module load StdEnv/2023 gcc/12.3 python/3.11.5 scipy-stack/2024b
export FLEXIBLAS=AOCL
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
cd /lustre09/project/6003507/zhouzb79/twist_qsi
python3 -m py_compile campaign/run_wp1_tomography.py || { echo "SYNTAX FAIL"; exit 1; }
echo "########## cubic16 (the known two-perimeter case) ##########"
python3 -u campaign/run_wp1_tomography.py --classes cubic16
echo "########## fcc32 (the production cluster) ##########"
python3 -u campaign/run_wp1_tomography.py --classes fcc32
STATUS=$?
echo "exit=$STATUS  $(date)"
exit $STATUS
