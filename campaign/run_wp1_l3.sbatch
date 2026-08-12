#!/bin/bash
# WP1: the levels=3 self-convergence spot check, at Jpm = -0.20.
#
# Why this run is not optional.  The truncation gate can calibrate K_8/K_6 on
# cubic-16 against the exact answer, but K_10 and K_12 are identically zero
# there -- every connected string longer than eight wraps on sixteen sites --
# so the amplitudes that xi_gauge depends on MOST have no external calibration
# at all.  Their truncation error can only be established by self-convergence
# in levels on FCC-32 itself, and this is that check.
#
# Coupling choice: -0.20, because the cubic-16 gate showed the levels=2 error
# growing with |Jpm| (2.4% at -0.05 to 18.9% at -0.30), so the check belongs
# where the bias is large rather than where it is comfortable.
#
# Sizing from the measured levels=2 runs (~60-70 min at 2.5M states, 12 chunks
# of 256 columns): levels=3 is 21.7M states, so chunk drops to 64 to keep the
# dense block near 11 GiB and the CG's five working blocks near 55 GiB, giving
# 47 chunks at roughly 11 min each -- call it 9-12 h.  The cluster grants no
# wall-time extensions, so --resume checkpoints every chunk and a timeout
# resumes instead of starting over.
#SBATCH --account=def-ybkim_cpu
#SBATCH --job-name=wp1_l3
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=64 --mem=249G
#SBATCH --time=23:59:00
#SBATCH --output=/lustre09/project/6003507/zhouzb79/twist_qsi/campaign/outputs/wp1_tomography/wp1_l3_%j.log
set -uo pipefail
module purge
module load StdEnv/2023 gcc/12.3 python/3.11.5 scipy-stack/2024b
export FLEXIBLAS=AOCL
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-64}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-64}
cd /lustre09/project/6003507/zhouzb79/twist_qsi
python3 -m py_compile campaign/run_wp1_tomography.py || { echo "SYNTAX FAIL"; exit 1; }
python3 -u campaign/run_wp1_tomography.py --resume --chunk=64 fcc32 3 -0.20
STATUS=$?
echo "exit=$STATUS  $(date)"
exit $STATUS
