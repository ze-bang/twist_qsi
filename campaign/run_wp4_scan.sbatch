#!/bin/bash
# WP4 step 1: locate the sector near-degeneracy locus on FCC-32.
#
# Radial cuts need a ring to cut through, and the locus is not known a priori.
# This scans the (Jpm, Jpmpm) plane for where the lowest two transport sectors
# approach each other -- where band-bottom ownership is about to change hands.
#
# levels=1 deliberately: this measures which sector sits lowest and by how
# much, a comparison between blocks of the SAME Hamiltonian at the SAME
# truncation, so the truncation bias is common to all sectors and largely
# cancels in the gap.  The absolute energies are not the point here.  (Contrast
# WP1, where levels=1 was fatal because it starves long loops specifically.)
#
# Kill criterion is pre-registered in the script docstring, before any results:
# if only the sector gap responds through the locus while Z, K_6, K_8/K_6 and
# the continuum edge stay flat, it is a host effect and is discarded.
#SBATCH --account=def-ybkim_cpu
#SBATCH --job-name=wp4_scan
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=32 --mem=120G
#SBATCH --time=06:00:00
#SBATCH --array=0-4
#SBATCH --output=/lustre09/project/6003507/zhouzb79/twist_qsi/campaign/outputs/wp4_sector_ring/wp4_scan_%A_%a.log
set -uo pipefail
module purge
module load StdEnv/2023 gcc/12.3 python/3.11.5 scipy-stack/2024b
export FLEXIBLAS=AOCL
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-32}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-32}
mkdir -p /lustre09/project/6003507/zhouzb79/twist_qsi/campaign/outputs/wp4_sector_ring
cd /lustre09/project/6003507/zhouzb79/twist_qsi

# one Jpm row per task, sweeping Jpmpm across the region WP2 mapped as
# band-complete (dissolution sets in above pp ~ 0.085-0.130 on the positive side)
JPM=(-0.30 -0.20 -0.125 -0.05 0.05)
J=${JPM[$SLURM_ARRAY_TASK_ID]}

python3 -m py_compile campaign/run_wp4_sector_ring.py || { echo "SYNTAX FAIL"; exit 1; }
python3 -u campaign/run_wp4_sector_ring.py --scan 1 \
    "$J" 0.00 "$J" 0.04 "$J" 0.085 "$J" 0.13 "$J" 0.17
STATUS=$?
echo "exit=$STATUS  $(date)"
exit $STATUS
