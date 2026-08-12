#!/bin/bash
# RESCUE: resume the levels=3 sampled point at Jpm/Jzz = -0.30.
#
# Job 18812873 reached 256 of 512 columns on an 11 h wall and was cancelled
# 12 min before its limit, deliberately, so that it could not contend with its
# own checkpoint while a replacement started.  The checkpoint
# (ckpt_fcc32_L3_-0p300_n512.npz, written 07:13) holds those 256 columns and
# this job resumes from them.
#
# The levels=2 baseline on the SAME 512-column sample is already complete
# (tomo_fcc32_L2_-0p300_n512.json, xi = 1.6145), so this run does only the
# levels=3 half.  Sampling bias is common to both and cancels in the ratio.
#
# Why the wall is now 23:59 rather than 11:00: the observed rate at this
# coupling is far slower than at -0.20, which is expected -- the conjugate
# gradient is worse conditioned at stronger coupling -- and this point is not
# optional.  It is the ONLY truncation check at strong coupling, and the
# saturation of xi_loop, which is the central claim of the physics draft,
# lives entirely there.  Over-requesting costs queue priority; timing out
# again costs a day.
#SBATCH --account=def-ybkim_cpu
#SBATCH --job-name=wp1_l3s30r
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=64 --mem=249G
#SBATCH --time=23:59:00
#SBATCH --output=/lustre09/project/6003507/zhouzb79/twist_qsi/campaign/outputs/wp1_tomography/wp1_l3s30r_%j.log
set -uo pipefail
module purge
module load StdEnv/2023 gcc/12.3 python/3.11.5 scipy-stack/2024b
export FLEXIBLAS=AOCL
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-64}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-64}
cd /lustre09/project/6003507/zhouzb79/twist_qsi
python3 -m py_compile campaign/run_wp1_tomography.py || { echo "SYNTAX FAIL"; exit 1; }
python3 -u campaign/run_wp1_tomography.py --resume --chunk=64 --columns=512 fcc32 3 -0.30
STATUS=$?
echo "exit=$STATUS  $(date)"
exit $STATUS
