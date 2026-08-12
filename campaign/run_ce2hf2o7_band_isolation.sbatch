#!/bin/bash
#SBATCH --account=def-ybkim_cpu
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=64 --mem=249G --time=6:00:00
#SBATCH --exclusive
#SBATCH --job-name=cho_s2_iso
#SBATCH --output=/lustre09/project/6003507/zhouzb79/twist_qsi/campaign/outputs/cho_s2_iso_%j.log
set -uo pipefail
module purge
module load StdEnv/2023 gcc/12.3 openmpi/4.1.5 python/3.11.5 aocl-blas/5.1 aocl-lapack/5.1 hdf5/1.14.6
export FLEXIBLAS=AOCL OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONPATH=/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v4/Compiler/gcccore/scipy-stack/2024b/lib/python3.11/site-packages
echo "=== band isolation across the M=3 source grid  $(date) ==="
python3 -u /lustre09/project/6003507/zhouzb79/twist_qsi/campaign/run_ce2hf2o7_band_isolation.py
echo "exit=$?  $(date)"
