#!/bin/bash
# WP2: the two-peak ratio across every plane point, certified and taxonomised.
#
# Chained behind the grid so it sees the complete set.  Cheap -- it recomputes
# C(T) from stored spectra -- so it can be re-run freely as points land.
#SBATCH --account=def-ybkim_cpu
#SBATCH --job-name=wp2_analyse
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/lustre09/project/6003507/zhouzb79/twist_qsi/campaign/outputs/wp2_gate/wp2_analyse_%j.log
set -uo pipefail
module purge
module load StdEnv/2023 gcc/12.3 python/3.11.5 scipy-stack/2024b
export FLEXIBLAS=AOCL
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
cd /lustre09/project/6003507/zhouzb79/twist_qsi
python3 -m py_compile campaign/run_wp2_analyse.py || { echo "SYNTAX FAIL"; exit 1; }
python3 -u campaign/run_wp2_analyse.py
STATUS=$?
echo "exit=$STATUS  $(date)"
exit $STATUS
