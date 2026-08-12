#!/bin/bash
#SBATCH --account=def-ybkim_cpu
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=64 --mem=249G --time=8:00:00
#SBATCH --exclusive
#SBATCH --job-name=cho_s2_ed
#SBATCH --output=/lustre09/project/6003507/zhouzb79/twist_qsi/campaign/outputs/cho_s2_ed_%j.log
set -uo pipefail
module purge
module load StdEnv/2023 gcc/12.3 openmpi/4.1.5 python/3.11.5 aocl-blas/5.1 aocl-lapack/5.1 hdf5/1.14.6
export FLEXIBLAS=AOCL OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONPATH=/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v4/Compiler/gcccore/scipy-stack/2024b/lib/python3.11/site-packages
R=/lustre09/project/6003507/zhouzb79/twist_qsi

# Ce2Hf2O7 "s2" NLCE fit, dipole-octupole XYZ (Ja,Jb,Jc)=(0.047693,0.025838,0.003838) meV
#   Jpm/Jzz   = -(Jb+Jc)/(4 Ja) = -0.15555742   (J_pm < 0: pi-flux)
#   Jpmpm/Jzz =  (Jb-Jc)/(4 Ja) = +0.11532091
# J_pm_pm != 0 breaks S^z conservation, so this runs in the full 2^16 basis.
echo "=== winding-free cubic-16 band at s2, M=2  $(date) ==="
python3 -u "$R/campaign/run_nonperturbative.py" \
  --jpm -0.15555742 --jpmpm 0.11532091 --max-grid 2
echo "exit=$?  $(date)"
