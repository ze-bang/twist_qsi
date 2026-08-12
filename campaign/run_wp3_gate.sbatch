#!/bin/bash
# WP3 validation ladder: cubic-16 first, FCC-32 only if it passes.
#
# The Gamma weight is a self-test with a known exact answer -- zero -- so this
# run cannot quietly produce a plausible-looking spectrum from a broken matrix
# element.  If the computed Gamma inelastic weight is not at machine zero, the
# implementation is wrong; the physics is not in question, WP0b settled it on
# 2970 FCC-32 ice states.
#
# cubic-16 at Jpm = -0.05 and -0.20 also has frozen tier-1 numbers to compare
# against: within-band longitudinal sticks dark at <= 3e-5, q=0 masked weight
# <= 9e-5 against raw 0.26-0.60.
#
# levels=1 for the gate: this measures low-lying spectral weight, not the long
# loop amplitudes that levels=1 starves, and the point here is correctness of
# the matrix elements rather than converged energies.
#SBATCH --account=def-ybkim_cpu
#SBATCH --job-name=wp3_gate
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=32 --mem=120G
#SBATCH --time=04:00:00
#SBATCH --output=/lustre09/project/6003507/zhouzb79/twist_qsi/campaign/outputs/wp3_dssf/wp3_gate_%j.log
set -uo pipefail
module purge
module load StdEnv/2023 gcc/12.3 python/3.11.5 scipy-stack/2024b
export FLEXIBLAS=AOCL
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-32}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-32}
mkdir -p /lustre09/project/6003507/zhouzb79/twist_qsi/campaign/outputs/wp3_dssf
cd /lustre09/project/6003507/zhouzb79/twist_qsi
python3 -m py_compile campaign/run_wp3_dssf.py || { echo "SYNTAX FAIL"; exit 1; }

echo "########## GATE: cubic-16, where tier-1 has the frozen answer ##########"
python3 -u campaign/run_wp3_dssf.py cubic16 3 -0.05 || { echo "GATE FAILED"; exit 1; }
python3 -u campaign/run_wp3_dssf.py cubic16 3 -0.20 || { echo "GATE FAILED"; exit 1; }

echo "########## FCC-32: the prediction ##########"
python3 -u campaign/run_wp3_dssf.py fcc32 1 -0.05
python3 -u campaign/run_wp3_dssf.py fcc32 1 -0.20
STATUS=$?
echo "exit=$STATUS  $(date)"
exit $STATUS
