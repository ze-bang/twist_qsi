#!/bin/bash
# Build main.pdf: pdflatex / bibtex / pdflatex x2 for references and crossrefs.
#SBATCH --account=def-ybkim_cpu
#SBATCH --job-name=buildA
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=2 --mem=8000M
#SBATCH --time=00:20:00
#SBATCH --output=/lustre09/project/6003507/zhouzb79/twist_qsi/paper_A_selection_rules/build_%j.log
set -uo pipefail
cd /lustre09/project/6003507/zhouzb79/twist_qsi/paper_A_selection_rules

pdflatex -interaction=nonstopmode -halt-on-error main.tex > pass1.log 2>&1
STATUS=$?
if [ $STATUS -ne 0 ]; then
    echo "########## PASS 1 FAILED ##########"
    grep -E "^!|^l\.[0-9]+|Undefined control|LaTeX Error" pass1.log | head -40
    exit $STATUS
fi
bibtex main > bib.log 2>&1 || echo "(bibtex returned $?)"
pdflatex -interaction=nonstopmode main.tex > pass2.log 2>&1
pdflatex -interaction=nonstopmode main.tex > pass3.log 2>&1

echo "########## undefined references / citations ##########"
grep -E "Reference .* undefined|Citation .* undefined|Label .* multiply" \
    pass3.log | sort -u | head -20 || true
echo "########## overfull boxes over 20pt ##########"
grep -E "Overfull \\\\hbox \([0-9]{2,}" pass3.log | head -10 || true
echo "########## result ##########"
ls -la main.pdf 2>/dev/null && pdfinfo main.pdf 2>/dev/null | grep -E "^Pages"
echo "done $(date)"
