#!/usr/bin/env bash
# Sequential full-spectrum ED on the J3 grid for the 16-site verify deck.
#
# Each case projects H into the two bit-flip Z2 blocks and runs
# scipy.linalg.eigh(eigvals_only=True, driver='evd') per block.
#
# Per-case cost (16 cores, OpenBLAS): ~1h40 wall, ~17 GB peak memory.
set -euo pipefail
cd "$(dirname "$0")/../.."   # repo root: exact_diagonalization_clean

ROOT="twist_qsi_demo/output/jpm03_j3_verify_pbc_qsus"
PHI="phi_0.000pi_0.000pi_0.000pi"
LOG_DIR="$ROOT/logs"
mkdir -p "$LOG_DIR"

run_one () {
    local j3="$1"
    local tag="$2"
    local ham="$ROOT/J3_${j3}/${PHI}/ham"
    local out_dir="$ham/full_thermo_pyed"
    local out="$out_dir/result.npz"
    local log="$LOG_DIR/full_ed_thermo_J3_${tag}.log"

    if [[ -f "$out" ]]; then
        echo "[$(date '+%H:%M:%S')] J3=$j3 already done at $out"
        return 0
    fi
    mkdir -p "$out_dir"
    echo "[$(date '+%H:%M:%S')] >>> J3=$j3  ham=$ham"
    OMP_NUM_THREADS=24 OPENBLAS_NUM_THREADS=24 MKL_NUM_THREADS=24 \
        NUMEXPR_NUM_THREADS=24 nice -n 19 taskset -c 8-31 \
        python3 -u twist_qsi_demo/scripts/run_full_ed_thermo_J3.py \
            --ham "$ham" --out "$out" \
            --temp-min 0.005 --temp-max 5.0 --temp-points 80 \
            --driver evd \
        > "$log" 2>&1
    echo "[$(date '+%H:%M:%S')] <<< J3=$j3 done"
}

for pair in "0.0200:0p02" "0.0400:0p04" "0.0600:0p06"; do
    j3="${pair%%:*}"
    tag="${pair##*:}"
    run_one "$j3" "$tag"
done

echo "[$(date '+%H:%M:%S')] full ED J3 batch complete"
