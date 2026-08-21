#!/bin/bash
# S^zz(q,w) along the connected Gamma-X-W-L-Gamma circuit, periodic and masked,
# at the QMC coupling and at the -0.20 reference.
#
# Also settles the Gamma question: the run prints the U(1)-protected uniform
# channel |sum_mu A^mu|^2 alongside the incoherent trace sum_mu |A^mu|^2 that
# the eight-column panel plots.  The uniform channel must be an exact zero at
# Gamma for BOTH methods, because sum_i S^z_i commutes with H; any finite
# periodic Gamma weight therefore lives entirely in the three sublattice-
# staggered channels, which nothing protects.
#SBATCH --account=def-ybkim_cpu
#SBATCH --job-name=dssf_path
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=32 --mem=120G
#SBATCH --time=03:00:00
#SBATCH --array=0-3
#SBATCH --output=/lustre09/project/6003507/zhouzb79/twist_qsi/campaign/outputs/wp3_dssf/path_%A_%a.log
set -uo pipefail
module purge
module load StdEnv/2023 gcc/12.3 python/3.11.5 scipy-stack/2024b
export FLEXIBLAS=AOCL
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-32}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-32}
cd /lustre09/project/6003507/zhouzb79/twist_qsi

JPM=(+0.046 +0.046 -0.200 -0.200)
MODE=(""     --masked ""    --masked)
THIS_JPM=${JPM[$SLURM_ARRAY_TASK_ID]}
THIS_MODE=${MODE[$SLURM_ARRAY_TASK_ID]}

echo "task=$SLURM_ARRAY_TASK_ID jpm=$THIS_JPM mode='${THIS_MODE:-periodic}' $(date)"
python3 -m py_compile campaign/run_dssf_path.py || exit 1
python3 -u campaign/run_dssf_path.py "$THIS_JPM" --depth=1 --nstates=144 \
    --per-segment=24 ${THIS_MODE}
STATUS=$?
echo "exit=$STATUS  $(date)"
exit $STATUS
