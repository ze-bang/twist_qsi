#!/bin/bash
# Connected flux-flux correlator per level: photon-like (extended) against
# monopole-like (localized).  One-point flux is uniform on every momentum
# eigenstate, so only the connected two-point function can tell them apart.
#SBATCH --account=def-ybkim_cpu
#SBATCH --job-name=fluxdiag
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=32 --mem=160G
#SBATCH --time=03:00:00
#SBATCH --array=0-1
#SBATCH --output=/lustre09/project/6003507/zhouzb79/twist_qsi/campaign/outputs/wp3_dssf/flux_%A_%a.log
set -uo pipefail
module purge
module load StdEnv/2023 gcc/12.3 python/3.11.5 scipy-stack/2024b
export FLEXIBLAS=AOCL
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-32}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-32}
cd /lustre09/project/6003507/zhouzb79/twist_qsi

JPM=(+0.046 -0.200)
THIS=${JPM[$SLURM_ARRAY_TASK_ID]}
echo "task=$SLURM_ARRAY_TASK_ID jpm=$THIS  $(date)"
python3 -m py_compile campaign/run_flux_diagnostic.py || exit 1
python3 -u campaign/run_flux_diagnostic.py "$THIS" --cluster=fcc32 \
    --depth=1 --nstates=60 --levels=8
STATUS=$?
echo "exit=$STATUS  $(date)"
exit $STATUS
