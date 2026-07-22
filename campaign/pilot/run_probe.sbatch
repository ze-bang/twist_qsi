#!/bin/bash
#SBATCH --account=def-ybkim_cpu
#SBATCH --job-name=qsi_probe
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=16G --time=00:15:00
#SBATCH --output=%x-%j.out
set -euo pipefail
module purge
module load StdEnv/2023 gcc/12.3 python/3.11.5 scipy-stack/2024b
export PYTHONPATH=/lustre09/project/6003507/zhouzb79/QED/python:${PYTHONPATH:-}
cd /lustre09/project/6003507/zhouzb79/twist_qsi
python -u campaign/pilot/probe.py
