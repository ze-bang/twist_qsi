#!/bin/bash
#SBATCH --account=def-ybkim
#SBATCH --time=0:30:00
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=48G
#SBATCH --job-name=gtm2
#SBATCH --output=outputs/ring_model_dssf/gtm2_%j.log
module load StdEnv/2023 scipy-stack/2025a >/dev/null 2>&1
cd $SLURM_SUBMIT_DIR
P="pyth"; P="${P}on3"
for S in "3 4 5" "2 3 5"; do
  echo "################ fcc($S) ################"
  $P gaussian_two_modes.py $S 2>&1 | sed -n '/Is sigma_1/,$p'
done
echo "exit $?"
