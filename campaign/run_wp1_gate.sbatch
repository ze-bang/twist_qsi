#!/bin/bash
# WP1 truncation gate on cubic-16, where the untruncated answer exists.
#
# This runs BEFORE any FCC-32 tomography, because it decides whether the
# FCC-32 numbers are allowed to carry a xi_gauge claim at all.  The worry is
# specific and one-sided: a long flipped string needs many virtual spinon hops,
# so truncating Q at `levels` hops biases the LONG loops low -- precisely the
# amplitudes whose decay defines xi_gauge.  A truncation artifact would masquerade
# as a short gauge range, and would do so consistently enough to look like a trend.
#
# Pre-registered kill criterion: if the levels=1 -> 2 drift of K_10/K_6 exceeds
# its variation across couplings, the trend is not separable from the artifact
# and levels=3 spot checks are required before any claim.
#SBATCH --account=def-ybkim_cpu
#SBATCH --job-name=wp1_gate
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=32 --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/lustre09/project/6003507/zhouzb79/twist_qsi/campaign/outputs/wp1_tomography/wp1_gate_%j.log
set -uo pipefail
module purge
module load StdEnv/2023 gcc/12.3 python/3.11.5 scipy-stack/2024b
export FLEXIBLAS=AOCL
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-32}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-32}
mkdir -p /lustre09/project/6003507/zhouzb79/twist_qsi/campaign/outputs/wp1_tomography
cd /lustre09/project/6003507/zhouzb79/twist_qsi
python3 -m py_compile campaign/run_wp1_tomography.py || { echo "SYNTAX FAIL"; exit 1; }
python3 -u campaign/run_wp1_tomography.py --gate cubic16 -0.05 -0.10 -0.20 -0.30
STATUS=$?
echo "exit=$STATUS  $(date)"
exit $STATUS
