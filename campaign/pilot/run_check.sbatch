#!/bin/bash
#SBATCH --account=def-ybkim_cpu
#SBATCH --job-name=qsi_chk
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=2 --mem=8G --time=00:10:00
#SBATCH --output=%x-%j.out
module purge; module load StdEnv/2023 gcc/12.3 python/3.11.5 scipy-stack/2024b
cd /lustre09/project/6003507/zhouzb79/twist_qsi && python -u campaign/pilot/check_index.py
