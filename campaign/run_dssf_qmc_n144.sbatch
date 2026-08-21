#!/bin/bash
# Periodic FCC-32 Szz at the QMC coupling on the CLEAN 144-state window.
#
# The window scan (job 19031050) showed the 120-state window cuts a degenerate
# multiplet at +0.046, breaking the four L-point weights that cubic symmetry
# requires to be equal:
#     n= 96   2.9% spread     n=168   L ok but Gamma zero degrades to 1.7e-9
#     n=120   5.6% spread     n=192   (not measured; task clobbered)
#     n=144   EXACT to 1e-14, Gamma = 2.2e-33   <-- the clean window
# The masked side already has its n=144 result staged.  The paper requires
# periodic and masked to use the same state count, so this supplies the
# matching periodic run.
#
# run_fcc32_dssf_periodic.py puts no nstates in its output tag, so this
# overwrites dssf_periodic_fcc32_L1_+0p046.json; the 120-state version is
# backed up alongside first.
#SBATCH --account=def-ybkim_cpu
#SBATCH --job-name=dssf_n144
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=32 --mem=120G
#SBATCH --time=01:00:00
#SBATCH --output=/lustre09/project/6003507/zhouzb79/twist_qsi/campaign/outputs/wp3_dssf/n144_%j.log
set -uo pipefail
module purge
module load StdEnv/2023 gcc/12.3 python/3.11.5 scipy-stack/2024b
export FLEXIBLAS=AOCL
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-32}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-32}
cd /lustre09/project/6003507/zhouzb79/twist_qsi

echo "periodic +0.046 at nstates=144  start $(date)"
python3 -u campaign/run_fcc32_dssf_periodic.py +0.046 --depth=1 --nstates=144
STATUS=$?
echo "exit=$STATUS  finish $(date)"
exit $STATUS
