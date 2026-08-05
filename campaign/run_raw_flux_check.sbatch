#!/bin/bash
#SBATCH --account=def-ybkim_cpu
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=48G --time=1:00:00
#SBATCH --job-name=raw_flux_check
#SBATCH --output=/lustre09/project/6003507/zhouzb79/twist_qsi/campaign/outputs/xy_phase_plane/rawfluxcheck_%j.log
# Attribution test for the pilot's flux disagreement.
#
# The pilot measures <O_hex> on the raw ground state and gets +0.091 at
# Jpm = -0.10, where the cached counterterm run reports -0.209.  Two
# explanations: my operator/basis is wrong, or the two numbers are simply
# measured on different states (H versus H + C).  run_strong_coupling_gs.py
# settles it -- it is the reference's own code path, the same Sz = 0 basis
# and the same loop_expectation, but on the BARE Hamiltonian.  If it also
# returns +0.09 the operator is fine and the gap is H versus H + C, i.e.
# the winding artefact is flipping the sign of the measured flux.
set -uo pipefail
module purge
module load StdEnv/2023 gcc/12.3 python/3.11.5 aocl-blas/5.1 aocl-lapack/5.1 scipy-stack/2024b
export FLEXIBLAS=AOCL OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

cd /lustre09/project/6003507/zhouzb79/twist_qsi/campaign
python3 -u run_strong_coupling_gs.py -0.10 -0.25 0.046
echo "exit=$?  $(date)"
