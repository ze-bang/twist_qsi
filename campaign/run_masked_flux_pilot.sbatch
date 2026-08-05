#!/bin/bash
#SBATCH --account=def-ybkim_cpu
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16 --mem=96G --time=2:00:00
#SBATCH --job-name=mflux_pilot
#SBATCH --output=/lustre09/project/6003507/zhouzb79/twist_qsi/campaign/outputs/xy_phase_plane/mfluxpilot_%j.log
# Pilot the masked-map flux before committing the plane.  Three of the six
# points sit on the XXZ line at couplings the counterterm campaign cached,
# so the sign and rough magnitude can be checked against known answers; the
# other three carry Jpmpm, where nothing is known yet.  Also fixes the
# per-point cost for sizing the array.
set -uo pipefail
module purge
module load StdEnv/2023 gcc/12.3 python/3.11.5 aocl-blas/5.1 aocl-lapack/5.1 scipy-stack/2024b
export FLEXIBLAS=AOCL OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

cd /lustre09/project/6003507/zhouzb79/twist_qsi/campaign
python3 -m py_compile scan_xy_masked_flux.py || { echo "SYNTAX FAIL"; exit 1; }
python3 -u scan_xy_masked_flux.py \
  outputs/xy_phase_plane/masked_pilot_points.json 0 1 masked_pilot
echo "exit=$?  $(date)"
