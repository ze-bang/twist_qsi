#!/bin/bash
# Paired periodic/masked FCC-32 thermodynamics.  Each array element is one
# shared-anchor Feshbach solve, so the two methods are exactly matched.
#SBATCH --account=def-ybkim_cpu
#SBATCH --job-name=fcc32_thermo
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=32 --mem=120G
#SBATCH --time=03:00:00
#SBATCH --array=0-19%20
#SBATCH --output=/lustre09/project/6003507/zhouzb79/twist_qsi/campaign/outputs/fcc32_thermodynamics/thermo_%A_%a.log
set -uo pipefail
module purge
module load StdEnv/2023 gcc/12.3 python/3.11.5 scipy-stack/2024b
export FLEXIBLAS=AOCL
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-32}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-32}
cd /lustre09/project/6003507/zhouzb79/twist_qsi

JPM=(+0.046 -0.010 -0.020 -0.030 -0.040 -0.050 -0.080 -0.110 -0.140 -0.170 -0.200 -0.230 -0.260 -0.290 -0.320 -0.350 -0.380 -0.410 -0.450 -0.500)
THIS=${JPM[$SLURM_ARRAY_TASK_ID]}

python3 -m py_compile campaign/run_fcc32_thermodynamics.py || exit 1
python3 -u campaign/run_fcc32_thermodynamics.py "$THIS" --depth=1
STATUS=$?
echo "exit=$STATUS  $(date)"
exit $STATUS
