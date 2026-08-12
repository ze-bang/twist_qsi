#!/bin/bash
# WP1 levels=3, sampled: the truncation bound that is guaranteed to land.
#
# The full levels=3 point completed its first 64-column chunk in ~46 min
# INCLUDING setup, and there are 47 chunks.  If the per-chunk cost is anywhere
# near that, the full run exceeds the 24 h wall and the cluster grants no
# extensions.  Rather than gamble the only available bound on xi_gauge's
# truncation error on that, this runs the same point over the first 512 ice
# columns and takes the 512 x 512 principal block.
#
# Why a sample is legitimate here: K_L is an average over thousands of entries
# per geometry class, and the comparison being made is levels=2 against
# levels=3 on the SAME sampled block, so the sampling bias is common to both
# and cancels in the ratio.  A matching sampled levels=2 run is done alongside
# for exactly that reason -- comparing a sampled L=3 against the full L=2 would
# confound truncation with sampling.
#
# The full run (18733001) keeps going with its checkpoints; if it finishes it
# supersedes this.  This is the floor, not the ceiling.
#SBATCH --account=def-ybkim_cpu
#SBATCH --job-name=wp1_l3s
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=64 --mem=249G
#SBATCH --time=11:00:00
#SBATCH --output=/lustre09/project/6003507/zhouzb79/twist_qsi/campaign/outputs/wp1_tomography/wp1_l3s_%j.log
set -uo pipefail
module purge
module load StdEnv/2023 gcc/12.3 python/3.11.5 scipy-stack/2024b
export FLEXIBLAS=AOCL
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-64}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-64}
cd /lustre09/project/6003507/zhouzb79/twist_qsi
python3 -m py_compile campaign/run_wp1_tomography.py || { echo "SYNTAX FAIL"; exit 1; }

echo "########## levels=2, same 512-column sample (the comparison baseline) ##########"
python3 -u campaign/run_wp1_tomography.py --chunk=128 --columns=512 fcc32 2 -0.20

echo "########## levels=3, same 512-column sample ##########"
python3 -u campaign/run_wp1_tomography.py --resume --chunk=64 --columns=512 fcc32 3 -0.20
STATUS=$?
echo "exit=$STATUS  $(date)"
exit $STATUS
