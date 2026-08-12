#!/bin/bash
# WP0b: Gamma-point selection rule, cubic-16 control + FCC-32 verdict.
# Pure combinatorics on 2970 ice states -- minutes, but it runs on a compute
# node like everything else.  Whole node (--exclusive): shared bycore genoa
# nodes throttle this kind of numpy work badly enough to be worth avoiding
# even for a short job.
#SBATCH --account=def-ybkim_cpu
#SBATCH --job-name=wp0b_gamma
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --exclusive
#SBATCH --time=00:30:00
#SBATCH --output=/lustre09/project/6003507/zhouzb79/twist_qsi/campaign/outputs/wp0_preflight/wp0b_%j.log
set -uo pipefail
module purge
module load StdEnv/2023 gcc/12.3 python/3.11.5 scipy-stack/2024b
export FLEXIBLAS=AOCL
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
cd /lustre09/project/6003507/zhouzb79/twist_qsi
python3 -m py_compile campaign/run_wp0b_gamma_rule.py || { echo "SYNTAX FAIL"; exit 1; }
python3 -u campaign/run_wp0b_gamma_rule.py cubic16 fcc32
STATUS=$?
echo "exit=$STATUS  $(date)"
exit $STATUS
