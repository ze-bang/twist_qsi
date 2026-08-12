#!/bin/bash
# WP2 blocking gate: the pair-flip truncated builder against the reference.
#
# Nothing in WP2 may run until this passes.  The XYZ scan's entire value is
# that it is a rigorous statement about the nearest-neighbour model -- a
# builder with a sign slip or a leaking pair move would produce a confident
# null about a Hamiltonian nobody proposed.
#
# Includes the Ce2Hf2O7 fit point (-0.125, 0.085) and the certified ratio
# ceiling (-0.24, 0.15), so the gate covers the couplings WP2 actually cares
# about rather than only the easy ones.
#SBATCH --account=def-ybkim_cpu
#SBATCH --job-name=wp2_gate
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16 --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/lustre09/project/6003507/zhouzb79/twist_qsi/campaign/outputs/wp2_gate/wp2_gate_%j.log
set -uo pipefail
module purge
module load StdEnv/2023 gcc/12.3 python/3.11.5 scipy-stack/2024b
export FLEXIBLAS=AOCL
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
mkdir -p /lustre09/project/6003507/zhouzb79/twist_qsi/campaign/outputs/wp2_gate
cd /lustre09/project/6003507/zhouzb79/twist_qsi
python3 -m py_compile campaign/run_wp2_gate.py campaign/run_truncated_feshbach.py \
  || { echo "SYNTAX FAIL"; exit 1; }
python3 -u campaign/run_wp2_gate.py
STATUS=$?
echo "exit=$STATUS  $(date)"
exit $STATUS
