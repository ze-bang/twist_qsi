#!/bin/bash
# The SECOND ell=3 convergence point, at Jpm/Jzz = -0.30.
#
# Why it is required.  The manuscript's operator-range result ends in an
# apparent SATURATION of xi_loop: its increments across the last three
# couplings are 0.091, 0.058, 0.039.  That saturation is the strong-coupling
# end of the curve, and the only truncation check that exists today is at
# Jpm/Jzz = -0.20.  Claiming saturation while the deepest check sits away from
# where it happens is exactly the kind of gap a referee closes for you.
#
# Both depths are run on the SAME 512-column sample so the sampling bias is
# common and cancels in the comparison.  Comparing a sampled ell=3 against the
# full ell=2 would confound truncation with sampling; that error was made once
# already and produced a spurious 30% discrepancy.
#
# At -0.20 the answer was: ratios referred to K6 drift by ~33%, mutual ratios
# among the generated loops drift by <1%, and xi_loop moves 1.518 -> 1.511.
# The gate for this run is whether that pattern survives at -0.30.
#
# Restartable: --resume checkpoints every column chunk, and the checkpoint tag
# carries the sample size so a sampled run can never resume a full one's file.
#SBATCH --account=def-ybkim_cpu
#SBATCH --job-name=wp1_l3s30
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=64 --mem=249G
#SBATCH --time=11:00:00
#SBATCH --output=/lustre09/project/6003507/zhouzb79/twist_qsi/campaign/outputs/wp1_tomography/wp1_l3s30_%j.log
set -uo pipefail
module purge
module load StdEnv/2023 gcc/12.3 python/3.11.5 scipy-stack/2024b
export FLEXIBLAS=AOCL
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-64}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-64}
cd /lustre09/project/6003507/zhouzb79/twist_qsi
python3 -m py_compile campaign/run_wp1_tomography.py || { echo "SYNTAX FAIL"; exit 1; }

echo "########## levels=2, 512-column sample (comparison baseline) ##########"
python3 -u campaign/run_wp1_tomography.py --chunk=128 --columns=512 fcc32 2 -0.30

echo "########## levels=3, same 512-column sample ##########"
python3 -u campaign/run_wp1_tomography.py --resume --chunk=64 --columns=512 fcc32 3 -0.30
STATUS=$?
echo "exit=$STATUS  $(date)"
exit $STATUS
