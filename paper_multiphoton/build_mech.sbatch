#!/bin/bash
#SBATCH --job-name=buildmech
#SBATCH --account=def-ybkim
#SBATCH --time=0:30:00
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=2 --mem=8G
#SBATCH --output=/lustre09/project/6003507/zhouzb79/twist_qsi/paper_multiphoton/buildmech_%j.log
module load StdEnv/2023 texlive/2023 >/dev/null 2>&1
cd /lustre09/project/6003507/zhouzb79/twist_qsi/paper_multiphoton
for i in 1 2 3; do pdflatex -interaction=nonstopmode -halt-on-error mechanism_note.tex > mech_pass$i.log 2>&1; done
echo "########## errors ##########"
grep -E "^!|Undefined control|Package .* Error" mech_pass3.log | head -20
echo "########## result ##########"
ls -l mechanism_note.pdf 2>/dev/null && pdfinfo mechanism_note.pdf 2>/dev/null | grep Pages
echo done
