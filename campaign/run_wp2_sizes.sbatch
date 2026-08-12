#!/bin/bash
# WP2 sizing: how big is the FCC-32 truncated space once pair flips are hops?
#
# The plan's own note says the pair-move space is "several-fold larger" than
# the XXZ one at the same depth, and that WP2 must be budgeted accordingly.
# The XXZ ladder is now measured (140k / 2.5M / 21.7M at levels 1/2/3); this
# measures the pair-flip ladder alongside it so the WP2 scan is designed
# against a number instead of an adjective.
#
# Cheap: it is a BFS, not a solve.  It runs before the scan is designed.
#SBATCH --account=def-ybkim_cpu
#SBATCH --job-name=wp2_sizes
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16 --mem=249G
#SBATCH --time=02:00:00
#SBATCH --output=/lustre09/project/6003507/zhouzb79/twist_qsi/campaign/outputs/wp2_gate/wp2_sizes_%j.log
set -uo pipefail
module purge
module load StdEnv/2023 gcc/12.3 python/3.11.5 scipy-stack/2024b
export FLEXIBLAS=AOCL
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
cd /lustre09/project/6003507/zhouzb79/twist_qsi
python3 -m py_compile campaign/run_wp1_tomography.py || { echo "SYNTAX FAIL"; exit 1; }
python3 -u campaign/run_wp1_tomography.py --sizes fcc32 3
STATUS=$?
echo "exit=$STATUS  $(date)"
exit $STATUS
