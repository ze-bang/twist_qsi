#!/bin/bash
# Does the FCC-32 flux deviation survive levels=2?
#
# The cubic-16 test settled that the deviation is neither definitional nor an
# artifact of the bordered projection: that cluster returns exactly -0.25 under
# the same definition at every depth from levels=1 to levels=4 (the whole
# fixed-Sz space).  It returns it while Z falls from 0.935 to 0.473 -- an
# observable that ignores a factor-two change in the spinon admixture is pinned
# by combinatorics, not measuring the state.  Four flippable hexagons out of
# sixteen forces +-4/16 on that host.
#
# So cubic-16 cannot test truncation sensitivity of an UNPINNED flux, and
# FCC-32's deviation is still only known at levels=1.  This is that test, on
# the XXZ line where levels=2 is 2.5M states rather than the 7.3M the pair-flip
# branch would cost.
#
#   deviation stable L1 -> L2  ->  physical: virtual matter shifts the flux,
#                                  and the shift measures the explicit breaking
#                                  of the emergent one-form symmetry by charges
#   deviation moves            ->  truncation; needs levels=3 before any claim
#SBATCH --account=def-ybkim_cpu
#SBATCH --job-name=wp4_fluxl2
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=32 --mem=249G
#SBATCH --time=08:00:00
#SBATCH --output=/lustre09/project/6003507/zhouzb79/twist_qsi/campaign/outputs/wp4_sector_ring/wp4_fluxl2_%j.log
set -uo pipefail
module purge
module load StdEnv/2023 gcc/12.3 python/3.11.5 scipy-stack/2024b
export FLEXIBLAS=AOCL
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-32}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-32}
cd /lustre09/project/6003507/zhouzb79/twist_qsi
python3 -m py_compile campaign/run_wp4_sector_ring.py || { echo "SYNTAX FAIL"; exit 1; }
# the same four couplings as the levels=1 row, so the comparison is one-to-one
python3 -u campaign/run_wp4_sector_ring.py --scan 2 \
    -0.05 0.00 -0.125 0.00 -0.20 0.00 -0.30 0.00
STATUS=$?
echo "exit=$STATUS  $(date)"
exit $STATUS
