#!/bin/bash
#SBATCH --job-name=qsi_gauge
#SBATCH --output=logs/qsi_%x_%j.out
#SBATCH --error=logs/qsi_%x_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
# --- adjust for your site ---
# #SBATCH --partition=standard
# #SBATCH --account=your_account

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs output

# Phase is passed with:  sbatch --export=ALL,PHASE=A submit_qsi_campaign.sbatch
PHASE="${PHASE:-B}"

# --- environment (edit to your cluster) ---
# module load python/3.12 openblas
# source /path/to/venv/bin/activate
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

echo "=== QSI gauge-sector campaign, Phase ${PHASE} on $(hostname) ==="
python3 scripts/run_campaign.py --phase "${PHASE}" --outdir output --nt 1400

echo "=== done; outputs in output/phase${PHASE}/ ==="
# Plot with:  python3 notes/make_paper_figures.py
