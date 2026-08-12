#!/bin/bash
# Render Figs. 3 and 4 for main.tex.
#
# Fig. 3 (loop hierarchy) is a RE-render: the committed PDF was made
# 2026-08-10 23:37, before the ell=3 point at Jpm/Jzz=-0.30 landed on
# 2026-08-11 13:55.  Panel (c) currently carries only the -0.20 check.  The
# script already looks for the -0.30 file "if present" and degrades
# gracefully, so this run simply picks it up.
#
# Fig. 4 (causal ladder) is a FIRST render.  Its plotting script was
# deliberately not written until the norm budget landed, because the ladder
# result is negative and the figure had to be designed around what the data
# actually shows rather than what it was commissioned to show.
#SBATCH --account=def-ybkim_cpu
#SBATCH --job-name=figs34
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/lustre09/project/6003507/zhouzb79/twist_qsi/campaign/outputs/figs34_%j.log
set -uo pipefail
module purge
module load StdEnv/2023 gcc/12.3 python/3.11.5 scipy-stack/2024b
export FLEXIBLAS=AOCL
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
cd /lustre09/project/6003507/zhouzb79/twist_qsi
python3 -m py_compile campaign/make_loop_hierarchy_figure.py \
                      campaign/make_causal_ladder_figure.py \
  || { echo "SYNTAX FAIL"; exit 1; }

echo "########## Fig. 3 re-render (now with the -0.30 ell=3 point) ##########"
python3 -u campaign/make_loop_hierarchy_figure.py
A=$?
echo "########## Fig. 4 first render ##########"
python3 -u campaign/make_causal_ladder_figure.py
B=$?
echo "exit fig3=$A fig4=$B  $(date)"
[ "$A" -eq 0 ] && [ "$B" -eq 0 ]
