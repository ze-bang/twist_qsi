#!/bin/bash
# Render the connected-path S^zz figures (trace and U(1)-uniform channels).
#SBATCH --account=def-ybkim_cpu
#SBATCH --job-name=pathfig
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=16000M
#SBATCH --time=00:20:00
#SBATCH --output=/lustre09/project/6003507/zhouzb79/twist_qsi/campaign/outputs/wp3_dssf/pathfig2_%j.log
set -uo pipefail
module purge
module load StdEnv/2023 gcc/12.3 python/3.11.5 scipy-stack/2024b
export FLEXIBLAS=AOCL
cd /lustre09/project/6003507/zhouzb79/twist_qsi
python3 -m py_compile campaign/make_dssf_path_figure.py || exit 1
python3 -u campaign/make_dssf_path_figure.py
STATUS=$?
echo "exit=$STATUS  $(date)"
exit $STATUS
