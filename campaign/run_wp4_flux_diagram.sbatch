#!/bin/bash
# The FCC-32 gauge-sector diagram: flux label over the (Jpm, Jpmpm) plane.
#
# What this IS.  A map of the one discrete, quantized observable available on
# FCC-32: the winding-free hexagon flux, -1/4 (pi) or +1/4 (0).  Energies and
# gaps move continuously and cannot mark a boundary by themselves; a quantized
# label either holds or it does not, so this is the only thing on this cluster
# that can draw a line rather than a contour.
#
# What this IS NOT.  A phase diagram of the model.  Thermodynamic phase
# boundaries need the full spectrum, and with Jpmpm breaking Sz the FCC-32 raw
# side is the whole 2^32 space -- out of reach by orders of magnitude.  Nor is
# band dissolution a phase boundary: a finite cluster can hold all 90
# ice-descended roots and still be ordered in the thermodynamic limit.  Read
# this as "where the emergent gauge sector is intact and which flux it
# carries", not as "where the material orders".
#
# It also settles a specific referee risk.  The +-1/4 quantization is currently
# demonstrated only on cubic-16, where it is plausibly tied to that host's
# four-flippable-hexagon combinatorics.  If FCC-32 reproduces +-1/4 to machine
# precision the objection dies; if it does not, the campaign learns that before
# a referee finds it.
#
# Cost: measured at ~10 min per point at levels=1 with pair flips (235k
# states), so a 7x5 grid is well under an hour of wall time as an array.
#SBATCH --account=def-ybkim_cpu
#SBATCH --job-name=wp4_flux
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=32 --mem=120G
#SBATCH --time=06:00:00
#SBATCH --array=0-6
#SBATCH --output=/lustre09/project/6003507/zhouzb79/twist_qsi/campaign/outputs/wp4_sector_ring/wp4_flux_%A_%a.log
set -uo pipefail
module purge
module load StdEnv/2023 gcc/12.3 python/3.11.5 scipy-stack/2024b
export FLEXIBLAS=AOCL
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-32}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-32}
cd /lustre09/project/6003507/zhouzb79/twist_qsi

# both signs of Jpm: the flux is expected to change sign through Jpm = 0, and
# that sign change is the one boundary this cluster can actually resolve
JPM=(-0.30 -0.20 -0.125 -0.05 0.00 0.05 0.10)
J=${JPM[$SLURM_ARRAY_TASK_ID]}

python3 -m py_compile campaign/run_wp4_sector_ring.py || { echo "SYNTAX FAIL"; exit 1; }
python3 -u campaign/run_wp4_sector_ring.py --scan 1 \
    "$J" 0.00 "$J" 0.04 "$J" 0.085 "$J" 0.13 "$J" 0.17
STATUS=$?
echo "exit=$STATUS  $(date)"
exit $STATUS
