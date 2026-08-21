#!/bin/bash
# The 16-site contrast for the momentum-resolved spectrum.
#
# Important asymmetry: cubic-16 holds four fcc primitive cells, so it resolves
# FOUR momenta -- Gamma and the three X points -- and has NO L point at all.
# It therefore cannot test the X/L ratio; what it tests is whether omega(X)
# itself is stable between 16 and 32 sites, which is the finite-size question
# underneath the degeneracy claim.  If omega(X) moves a lot between the two
# clusters, no X/L comparison on FCC-32 means much either.
#SBATCH --account=def-ybkim_cpu
#SBATCH --job-name=momspec16
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16 --mem=60G
#SBATCH --time=02:00:00
#SBATCH --array=0-3
#SBATCH --output=/lustre09/project/6003507/zhouzb79/twist_qsi/campaign/outputs/wp3_dssf/momspec16_%A_%a.log
set -uo pipefail
module purge
module load StdEnv/2023 gcc/12.3 python/3.11.5 scipy-stack/2024b
export FLEXIBLAS=AOCL
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
cd /lustre09/project/6003507/zhouzb79/twist_qsi

JPM=(-0.200 +0.046 -0.200 +0.046)
MODE=(""     ""     --periodic --periodic)
THIS_JPM=${JPM[$SLURM_ARRAY_TASK_ID]}
THIS_MODE=${MODE[$SLURM_ARRAY_TASK_ID]}

echo "task=$SLURM_ARRAY_TASK_ID cubic16 jpm=$THIS_JPM mode='${THIS_MODE:-masked}' $(date)"
python3 -m py_compile campaign/run_momentum_resolved_spectrum.py || exit 1
python3 -u campaign/run_momentum_resolved_spectrum.py "$THIS_JPM" \
    --cluster=cubic16 --depth=1 --nstates=200 ${THIS_MODE}
STATUS=$?
echo "exit=$STATUS  $(date)"
exit $STATUS
