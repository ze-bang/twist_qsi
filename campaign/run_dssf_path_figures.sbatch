#!/bin/bash
# Regenerate the FCC-32 figures with the DSSF panels ordered along a coherent
# L-Gamma-X path through the fcc Brillouin zone.
#
# The eight cluster momenta are three high-symmetry stars -- one Gamma, three
# X, four L -- but allowed_momenta returns them interleaved (Gamma, L, L, X, L,
# X, X, L), so the old panel read as eight arbitrary columns.  Grouping them by
# star along L-Gamma-X makes it a path, puts Gamma in the middle where the mask
# sends the weight to zero, and lets a reader check by eye that members of the
# same star agree -- which is how the +0.046 window defect was caught.
#SBATCH --account=def-ybkim_cpu
#SBATCH --job-name=dssf_figs
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=/lustre09/project/6003507/zhouzb79/twist_qsi/campaign/outputs/wp3_dssf/pathfigs_%j.log
set -uo pipefail
module purge
module load StdEnv/2023 gcc/12.3 python/3.11.5 scipy-stack/2024b
export FLEXIBLAS=AOCL
cd /lustre09/project/6003507/zhouzb79/twist_qsi

python3 -m py_compile campaign/make_fcc32_paper_figures.py || exit 1
python3 -u campaign/make_fcc32_paper_figures.py
STATUS=$?
echo "exit=$STATUS  $(date)"
exit $STATUS
