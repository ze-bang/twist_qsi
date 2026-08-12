#!/bin/bash
#SBATCH --account=def-ybkim_cpu
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=192 --mem=750G --time=12:00:00
#SBATCH --exclusive
#SBATCH --array=0-1
#SBATCH --job-name=merge_line
#SBATCH --output=/lustre09/project/6003507/zhouzb79/twist_qsi/campaign/outputs/masked_band_complete/merge_%A_%a.log
# Merge-line pilot: the two cells that decide "peak merging while QSI" vs
# "QSI destruction".  Both are cells where the fold scanner's bracketing
# solver lost the band, so its verdict there is an instrument limit:
#   task 0: (-0.125, 0.30) -- the ONLY cell reaching the experimental
#           T_low/T_high ~ 1/3; fold reported 27/90 (uncertified).
#   task 1: (-0.160, 0.25) -- the certified gauge-spinon merge cell
#           (2 peaks -> 1); fold reported 75/90.
# Sized as a pilot, not the full array: the above-edge scan cost scales with
# the number of escaped branches, which is exactly what these cells maximize,
# so the array must be sized from a measurement rather than from the 40 min
# Gate A'/B/C cells (12 escapees) that are not representative.
set -uo pipefail
module purge
module load StdEnv/2023 gcc/12.3 openmpi/4.1.5 python/3.11.5 aocl-blas/5.1 aocl-lapack/5.1 hdf5/1.14.6
export FLEXIBLAS=AOCL OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONPATH=/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v4/Compiler/gcccore/scipy-stack/2024b/lib/python3.11/site-packages

cd /lustre09/project/6003507/zhouzb79/twist_qsi/campaign

case $SLURM_ARRAY_TASK_ID in
  0) JPM=-0.125; PP=0.30 ;;
  1) JPM=-0.160; PP=0.25 ;;
  *) echo "no such task"; exit 1 ;;
esac

echo "=== merge-line cell (Jpm, Jpmpm) = ($JPM, $PP)  $(date) ==="
python3 -u run_masked_band_complete.py --jpmpm $PP --window 0.60 --grid 12000 $JPM
echo "exit=$?  $(date)"
