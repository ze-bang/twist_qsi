#!/usr/bin/env bash
#SBATCH --job-name=qsi_gauge
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:30:00
# 48-site (E1) override at submit time:
#   sbatch --cpus-per-task=16 --mem=64G --time=12:00:00 --export=PHASE=E1 submit_gauge_campaign.sbatch
#
# π-flux QSI gauge-sector thermodynamics campaign (see SIMULATION_PLAN.md).
# One shared row-table build (Phase A) feeds every downstream sweep.
# Select the phase via  --export=PHASE=<A1|B1|C1|E1|E2>  (default A1).

set -euo pipefail
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS
export MKL_NUM_THREADS=$OMP_NUM_THREADS

cd "$(dirname "$0")"
mkdir -p logs rows data
PHASE=${PHASE:-A1}
PY="python3 gauge_thermo.py"
ROWS32=rows/rows_fcc222_o4.npz
ROWS48=rows/rows_fcc322_o3.npz

echo "[$(date)] phase=$PHASE  threads=$OMP_NUM_THREADS  host=$(hostname)"

case "$PHASE" in
  A1)  # one-time production build (32-site, order 4) + order-3 cross-check
    $PY build --basis fcc --shape 2 2 2 --order 4 --out $ROWS32
    $PY build --basis fcc --shape 2 2 2 --order 3 --out rows/rows_fcc222_o3.npz
    ;;
  B1)  # 0-flux benchmark sweep (credibility firewall; compare to QMC offline)
    $PY sweep --rows $ROWS32 --order 4 --modes all delta0 \
        --jpm 0.03 0.046 0.05 0.07 0.10 --out data/thermo_0flux.json
    ;;
  C1)  # π-flux payoff sweep + scaling
    $PY sweep --rows $ROWS32 --order 4 --modes all delta0 \
        --jpm -0.03 -0.046 -0.05 -0.07 -0.10 --out data/thermo_piflux.json
    ;;
  E1)  # STRETCH: 48-site build (needs sparse assemble, Task T2). ~6-12 h.
    $PY build --basis fcc --shape 3 2 2 --order 3 --out $ROWS48
    ;;
  E2)  # STRETCH: 48-site Lanczos sweep
    $PY sweep --rows $ROWS48 --order 3 --modes all delta0 --lanczos \
        --lanczos-k 1500 --jpm -0.05 0.046 --out data/thermo_fcc322.json
    ;;
  *) echo "unknown PHASE=$PHASE"; exit 1 ;;
esac
echo "[$(date)] phase=$PHASE done"
