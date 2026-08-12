#!/bin/bash
# How big is the FCC-32 truncated space at each level?
#
# The gate says K_L needs levels >= 3 for percent-level accuracy, but the
# block-CG fold carries a DENSE right-hand side of 256 columns over the whole
# complement, so the space size sets both the runtime and the memory floor.
# Measure it before budgeting WP1 production rather than discovering it in a
# job that dies at hour six.
#SBATCH --account=def-ybkim_cpu
#SBATCH --job-name=wp1_sizes
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16 --mem=249G
#SBATCH --time=01:00:00
#SBATCH --output=/lustre09/project/6003507/zhouzb79/twist_qsi/campaign/outputs/wp1_tomography/wp1_sizes_%j.log
set -uo pipefail
module purge
module load StdEnv/2023 gcc/12.3 python/3.11.5 scipy-stack/2024b
export FLEXIBLAS=AOCL
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
cd /lustre09/project/6003507/zhouzb79/twist_qsi
python3 -m py_compile campaign/run_wp1_tomography.py || { echo "SYNTAX FAIL"; exit 1; }
python3 -u campaign/run_wp1_tomography.py --sizes fcc32 "${DEPTH:-3}"
STATUS=$?
echo "exit=$STATUS  $(date)"
exit $STATUS
