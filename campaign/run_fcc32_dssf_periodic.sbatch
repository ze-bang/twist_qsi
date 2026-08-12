#!/bin/bash
# Naive periodic response at every FCC-32 momentum, matched to the existing
# depth-one, 120-state masked response.
#SBATCH --account=def-ybkim_cpu
#SBATCH --job-name=fcc32_dssf_raw
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=32 --mem=120G
#SBATCH --time=04:00:00
#SBATCH --array=0-1
#SBATCH --output=/lustre09/project/6003507/zhouzb79/twist_qsi/campaign/outputs/wp3_dssf/periodic_%A_%a.log
set -uo pipefail
module purge
module load StdEnv/2023 gcc/12.3 python/3.11.5 scipy-stack/2024b
export FLEXIBLAS=AOCL
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-32}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-32}
cd /lustre09/project/6003507/zhouzb79/twist_qsi

JPM=(-0.050 -0.200)
THIS=${JPM[$SLURM_ARRAY_TASK_ID]}

python3 -m py_compile campaign/run_fcc32_dssf_periodic.py || exit 1
python3 -u campaign/run_fcc32_dssf_periodic.py "$THIS" --depth=1 --nstates=120
STATUS=$?
echo "exit=$STATUS  $(date)"
exit $STATUS
