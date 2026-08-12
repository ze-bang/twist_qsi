#!/bin/bash
# Is the flux deviation physics, truncation, or definition?
#
# FCC-32 gives |<O_hex>| = 0.2462 .. 0.2496 -- close to 1/4, but deviating
# monotonically with coupling and tracking the dressing (1-Z).  Noise does not
# order itself by coupling, so this is a systematic effect and worth resolving
# rather than discarding.
#
# The candidate reading is attractive: the electric one-form symmetry is
# EMERGENT, broken by charge-1 matter, so once virtual spinons dress the state
# the flux has no reason to be exactly quantized.  The deviation would then
# measure that breaking -- the same physics as xi_gauge through a different
# observable.  Which reframes the question as: why is cubic-16 exact?
#
# The two definitions should agree by a Feshbach identity: for an exact
# eigenvector |psi>, P|psi> normalised IS the masked-map eigenvector.  cubic-16
# measures the masked band state; this script measures the ice projection of
# the bordered state.  Same object untruncated -- so a disagreement is either
# truncation or real.
#
# This runs the bordered-projection definition on cubic-16, where levels=4 is
# the untruncated answer and the masked band is known to give exactly +-1/4:
#   deviation vanishes as levels grows  -> FCC-32's is truncation, check at L=2
#   deviation survives at levels=4      -> definitional, and the "machine
#                                          precision" came from measuring the
#                                          pure-ice band state instead
#   deviation survives AND tracks 1-Z   -> physics: emergent one-form symmetry
#                                          broken by virtual matter
#SBATCH --account=def-ybkim_cpu
#SBATCH --job-name=wp4_fluxdef
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=32 --mem=120G
#SBATCH --time=04:00:00
#SBATCH --output=/lustre09/project/6003507/zhouzb79/twist_qsi/campaign/outputs/wp4_sector_ring/wp4_fluxdef_%j.log
set -uo pipefail
module purge
module load StdEnv/2023 gcc/12.3 python/3.11.5 scipy-stack/2024b
export FLEXIBLAS=AOCL
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-32}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-32}
cd /lustre09/project/6003507/zhouzb79/twist_qsi
python3 -m py_compile campaign/run_wp4_sector_ring.py || { echo "SYNTAX FAIL"; exit 1; }

# the same couplings the anatomy reports at machine-precision +-1/4,
# swept in truncation depth to L=4 (= the whole fixed-Sz space on cubic-16)
for L in 1 2 3 4; do
  echo "########## cubic-16, levels=$L ##########"
  python3 -u campaign/run_wp4_sector_ring.py --cubic16 --scan "$L" \
      -0.05 0.00 -0.125 0.00 -0.20 0.00 -0.30 0.00
done
STATUS=$?
echo "exit=$STATUS  $(date)"
exit $STATUS
