#!/bin/bash
#SBATCH --account=def-ybkim_cpu
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=64 --mem=249G --time=4:00:00
#SBATCH --exclusive
#SBATCH --job-name=cho_scan
#SBATCH --output=/lustre09/project/6003507/zhouzb79/twist_qsi/campaign/outputs/cho_scan_%A_%a.log
set -uo pipefail
module purge
module load StdEnv/2023 gcc/12.3 openmpi/4.1.5 python/3.11.5 aocl-blas/5.1 aocl-lapack/5.1 hdf5/1.14.6
export FLEXIBLAS=AOCL OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONPATH=/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v4/Compiler/gcccore/scipy-stack/2024b/lib/python3.11/site-packages
R=/lustre09/project/6003507/zhouzb79/twist_qsi

# One (J_pm, J_pmpm) point per array task, both in units of J_zz.
# Pass 1 holds J_pmpm/J_zz at the s2 value and walks J_pm/J_zz: smaller |J_pm|
# widens the band gap and moves the calculation back inside the regime where
# the projection is controlled, at the cost of a smaller ring-exchange scale.
POINTS=(
  "-0.04000 0.11532091"
  "-0.06000 0.11532091"
  "-0.08000 0.11532091"
  "-0.10000 0.11532091"
  "-0.13000 0.11532091"
  "-0.20000 0.11532091"
  "-0.25000 0.11532091"
  "-0.10000 0.00000000"
)
read -r JPM JPMPM <<< "${POINTS[$SLURM_ARRAY_TASK_ID]}"
echo "=== scan point $SLURM_ARRAY_TASK_ID: jpm=$JPM jpmpm=$JPMPM  $(date) ==="

python3 -u "$R/campaign/run_cubic16_full_ed.py" --jpm "$JPM" --jpmpm "$JPMPM"
echo "full_ed exit=$?  $(date)"

python3 -u "$R/campaign/run_nonperturbative.py" --jpm "$JPM" --jpmpm "$JPMPM" --max-grid 2
echo "band exit=$?  $(date)"
