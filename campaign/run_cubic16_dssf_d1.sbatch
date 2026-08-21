#!/bin/bash
# Which X level does S^zz couple to on cubic-16?  Depth 1, to match the
# momentum-resolved spectra already computed at depth 1 on both clusters.
#
# The test: a two-particle state sits at ~2*Delta and is roughly size
# independent; a single dispersing mode drops as the box grows.  The photon
# candidate at X already halves, 0.054804 (cubic-16) -> 0.025274 (FCC-32).
# If the level S^zz couples to instead stays flat across the two clusters,
# it is a two-particle state and the vison-pair reading gains real support.
#
# Existing cubic-16 DSSF files are depth 3, so they cannot be compared with
# the depth-1 momentum spectra without confounding truncation with size.
#SBATCH --account=def-ybkim_cpu
#SBATCH --job-name=c16_dssf
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16 --mem=60G
#SBATCH --time=01:00:00
#SBATCH --array=0-1
#SBATCH --output=/lustre09/project/6003507/zhouzb79/twist_qsi/campaign/outputs/wp3_dssf/c16d1_%A_%a.log
set -uo pipefail
module purge
module load StdEnv/2023 gcc/12.3 python/3.11.5 scipy-stack/2024b
export FLEXIBLAS=AOCL
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
cd /lustre09/project/6003507/zhouzb79/twist_qsi

JPM=(-0.200 +0.046)
THIS=${JPM[$SLURM_ARRAY_TASK_ID]}
echo "task=$SLURM_ARRAY_TASK_ID cubic16 depth=1 jpm=$THIS  $(date)"
python3 -u campaign/run_wp3_dssf.py cubic16 1 "$THIS" --nstates=300
STATUS=$?
echo "exit=$STATUS  $(date)"
exit $STATUS
