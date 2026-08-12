#!/bin/bash
#SBATCH --account=def-ybkim_cpu
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=2 --mem=8G --time=0:25:00
#SBATCH --job-name=ped_tex
#SBATCH --output=/lustre09/project/6003507/zhouzb79/twist_qsi/campaign/outputs/pedtex_%j.log
set -uo pipefail
module purge
module load StdEnv/2023 texlive/2023 2>/dev/null || module load texlive 2>/dev/null
cd /lustre09/project/6003507/zhouzb79/twist_qsi/paper
for pass in 1 2 3; do
  pdflatex -interaction=nonstopmode -halt-on-error pedagogical.tex > /dev/null 2>&1 \
    || { echo "PASS $pass FAILED"; grep -E "^!|^l\.[0-9]" pedagogical.log | head -25; exit 1; }
done
echo "=== build OK ==="
grep -c "" pedagogical.tex | xargs echo "source lines:"
pdfinfo pedagogical.pdf 2>/dev/null | grep -E "^Pages" || true
grep -iE "undefined (control|reference)|multiply defined" pedagogical.log | head -10 || true
echo "exit=$?  $(date)"
