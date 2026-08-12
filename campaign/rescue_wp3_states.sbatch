#!/bin/bash
# RESCUE: the 120-state DSSF point at Jpm/Jzz = -0.20.
#
# Job 18780074 asked for two couplings on a 6 h wall.  The first (-0.05) alone
# took 5 h 20 m, so the second never ran and the file on disk for -0.20 is the
# superseded 24-state version -- the one whose spurious machine-zero weight at
# the X points produced the now-retracted claim that half the Brillouin zone is
# dark.  That file has been quarantined out of the glob pattern; this job
# produces its replacement.
#
# One coupling only, and 12 h rather than 6.  The response code first scans one
# eigenvalue per transport sector, then resolves 120 states only in the ground
# sector(s); excited sectors do not enter the zero-temperature response.
#
# What the result must show, if the retraction is right: finite weight at all
# seven nonzero cluster momenta, and Gamma alone suppressed by ~1e29.  A
# surviving machine-zero at any X point would mean the -0.05 result was the
# anomaly rather than the 24-state run, and the retraction would need
# revisiting.
#SBATCH --account=def-ybkim_cpu
#SBATCH --job-name=wp3_states20_fast
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=32 --mem=120G
#SBATCH --time=12:00:00
#SBATCH --output=/lustre09/project/6003507/zhouzb79/twist_qsi/campaign/outputs/wp3_dssf/wp3_states20_%j.log
set -uo pipefail
module purge
module load StdEnv/2023 gcc/12.3 python/3.11.5 scipy-stack/2024b
export FLEXIBLAS=AOCL
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-32}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-32}
cd /lustre09/project/6003507/zhouzb79/twist_qsi
python3 -m py_compile campaign/run_wp3_dssf.py || { echo "SYNTAX FAIL"; exit 1; }
python3 -u campaign/run_wp3_dssf.py fcc32 1 -0.20 --nstates=120
STATUS=$?
echo "exit=$STATUS  $(date)"
exit $STATUS
