#!/bin/bash
# Test the claimed X/L degeneracy on the eigenvalues themselves.
#
# S^zz cannot test it: a mode can exist at a momentum and be invisible there
# because the sublattice form factor gives it no overlap.  Reading omega(X) and
# omega(L) off the structure factor gave 0.03793 and 0.01568 at Jpm=-0.200 --
# a factor 2.4 apart, against a literature expectation of degeneracy.  Either
# the expectation does not apply here, or the lines S^zz shows are not the same
# excitation at the two momenta.  Projecting the eigenvectors onto the eight
# cluster momenta settles which, independently of any probe.
#SBATCH --account=def-ybkim_cpu
#SBATCH --job-name=momspec
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=32 --mem=160G
#SBATCH --time=04:00:00
#SBATCH --array=0-2
#SBATCH --output=/lustre09/project/6003507/zhouzb79/twist_qsi/campaign/outputs/wp3_dssf/momspec_%A_%a.log
set -uo pipefail
module purge
module load StdEnv/2023 gcc/12.3 python/3.11.5 scipy-stack/2024b
export FLEXIBLAS=AOCL
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-32}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-32}
cd /lustre09/project/6003507/zhouzb79/twist_qsi

JPM=(-0.200 +0.046 -0.200)
MODE=(""     ""     --periodic)
THIS_JPM=${JPM[$SLURM_ARRAY_TASK_ID]}
THIS_MODE=${MODE[$SLURM_ARRAY_TASK_ID]}

echo "task=$SLURM_ARRAY_TASK_ID jpm=$THIS_JPM mode='${THIS_MODE:-masked}' $(date)"
python3 -m py_compile campaign/run_momentum_resolved_spectrum.py || exit 1
python3 -u campaign/run_momentum_resolved_spectrum.py "$THIS_JPM" \
    --depth=1 --nstates=200 ${THIS_MODE}
STATUS=$?
echo "exit=$STATUS  $(date)"
exit $STATUS
