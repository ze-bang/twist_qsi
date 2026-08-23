#!/bin/bash
#SBATCH --job-name=buildSM
#SBATCH --account=def-ybkim
#SBATCH --time=0:15:00
#SBATCH --ntasks=1 --cpus-per-task=2 --mem=8000M
#SBATCH --output=/lustre09/project/6003507/zhouzb79/twist_qsi/paper_multiphoton/buildsm_%j.log
cd /lustre09/project/6003507/zhouzb79/twist_qsi/paper_multiphoton
pdflatex -interaction=nonstopmode supplement.tex > smp1.log 2>&1
bibtex supplement > smbib.log 2>&1
pdflatex -interaction=nonstopmode supplement.tex > smp2.log 2>&1
pdflatex -interaction=nonstopmode supplement.tex > smp3.log 2>&1
echo "### undefined:"; grep -E "Reference .* undefined|Citation .* undefined" smp3.log | sort -u | head -30
echo "### errors:"; grep -E "^!" smp3.log | head -10
ls -la supplement.pdf && pdfinfo supplement.pdf 2>/dev/null | grep Pages
echo done
