#!/bin/bash
#SBATCH --account=def-ybkim_cpu
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=192 --mem=249G --time=8:00:00
#SBATCH --exclusive
#SBATCH --job-name=masked_gates
#SBATCH --output=/lustre09/project/6003507/zhouzb79/twist_qsi/campaign/outputs/masked_band_complete/gates_%j.log
# Gates of the (Jpm, Jpmpm) bordered-band campaign, Klein-blocked path:
#   A': full basis at pp=0 (-0.05, -0.28) must reproduce the tracked
#       fixed-Sz oracle (and the direct full-basis pilot at -0.05);
#   B : certified plane cells (-0.16, 0.20) and (-0.125, 0.25) must
#       reproduce the fold scanner's roots below the edge;
#   C : pp -> -pp equivalence at (-0.10, +/-0.10).
set -uo pipefail
module purge
module load StdEnv/2023 gcc/12.3 openmpi/4.1.5 python/3.11.5 aocl-blas/5.1 aocl-lapack/5.1 hdf5/1.14.6
export FLEXIBLAS=AOCL OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONPATH=/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v4/Compiler/gcccore/scipy-stack/2024b/lib/python3.11/site-packages

cd /lustre09/project/6003507/zhouzb79/twist_qsi/campaign

echo "=== validation: full-basis Hamiltonian structure ==="
python3 -u check_full_basis.py || { echo "STRUCTURE CHECK FAILED"; exit 1; }
echo "=== validation: Klein-blocked assembly vs sparse CG + tracked oracle ==="
python3 -u check_klein_blocks.py || { echo "KLEIN CHECK FAILED"; exit 1; }

echo "=== Gate A': klein full basis, pp=0 ==="
python3 -u run_masked_band_complete.py --full-basis --window 0.40 --grid 6000 -0.05
python3 -u run_masked_band_complete.py --full-basis --window 0.40 --grid 6000 -0.28
echo "=== Gate B: certified plane cells ==="
python3 -u run_masked_band_complete.py --jpmpm 0.20 --window 0.40 --grid 20000 -0.16
python3 -u run_masked_band_complete.py --jpmpm 0.25 --window 0.40 --grid 20000 -0.125
echo "=== Gate C: pp sign symmetry ==="
python3 -u run_masked_band_complete.py --jpmpm 0.10 --window 0.40 --grid 20000 -0.10
python3 -u run_masked_band_complete.py --jpmpm -0.10 --window 0.40 --grid 20000 -0.10
echo "exit=$?  $(date)"
