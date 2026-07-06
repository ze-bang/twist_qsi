#!/usr/bin/env bash
# launch_all_H_decomp.sh
#
# H-decomposition (--decompose-H) rerun for ALL source-field campaigns.
# Decomposes alpha_Q into H_Ising + H_pm (+ H_J3 where applicable) contributions.
#
# Strategy: run 2 cases in parallel, each pinned to 16 cores with 16 OMP threads
#   - core set A: 0-15
#   - core set B: 16-31
# This gives ~2-3x LAPACK speedup vs the previous OMP_NUM_THREADS=6 single-case runs.
# Each case still takes ~30-40 min (4 sectors × ~500s with 16 threads).
# Total: ~7 rounds × 35 min ≈ 4 hours for all 13 remaining cases.
#
# Cases already complete (skip automatically):
#   jpm_pos004 EgQ1, jpm_pos002 EgQ1, jpm_neg004 EgQ1 (last sector running)
# Cases queued here (13 total):
#   jpm03 J3=0.08 Eg, T2g
#   jpm03 J3=0    Eg, T2g
#   jpm02 J3=0    Eg, T2g
#   jpm01 J3=0    Eg, T2g
#   jpm_neg004 T2g
#   jpm_neg002 Eg, T2g
#   jpm_pos002 T2g
#   jpm_pos004 T2g

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

SCRIPT="twist_qsi_demo/scripts/run_full_ed_quad_qh_J3.py"

# OMP thread counts — two parallel slots, each gets half the cores
THREADS_A=16   # slot A
THREADS_B=16   # slot B
CORES_A="0-15"
CORES_B="16-31"

LAMBDA="1e-3"

# ---------------------------------------------------------------------------
# run_case SCAN_ROOT TAG SOURCE CORES THREADS [J3_DIR]
#   SCAN_ROOT : e.g. twist_qsi_demo/output/jpm03_j3_verify_pbc_qsus
#   TAG       : output subdirectory name under H_decomp_campaign/
#   SOURCE    : --source argument, e.g. "Eg_Q1=1e-3"
#   CORES     : taskset core spec, e.g. "0-15"
#   THREADS   : OMP_NUM_THREADS
#   J3_DIR    : optional J3 subdirectory, default "J3_0.0000"
# ---------------------------------------------------------------------------
run_case() {
    local scan_root="$1"
    local tag="$2"
    local source="$3"
    local cores="$4"
    local threads="$5"
    local j3_dir="${6:-J3_0.0000}"

    local ham="$scan_root/$j3_dir/phi_0.000pi_0.000pi_0.000pi/ham"
    local quad="$scan_root/quad_operators"
    local out_root="$scan_root/H_decomp_campaign"
    local log_dir="$out_root/logs"
    local case_dir="$out_root/$tag"
    local ckpt_dir="$case_dir/checkpoints"
    local out_npz="$case_dir/result.npz"
    local log="$log_dir/${tag}.log"

    mkdir -p "$log_dir" "$case_dir" "$ckpt_dir"

    if [[ -f "$out_npz" ]]; then
        echo "[skip] $tag already complete ($(basename $scan_root))"
        return 0
    fi

    echo "[run ] $tag | cores=$cores threads=$threads | source=$source"
    OMP_NUM_THREADS=$threads \
    OPENBLAS_NUM_THREADS=$threads \
    MKL_NUM_THREADS=$threads \
    NUMEXPR_NUM_THREADS=$threads \
    nice -n 19 taskset -c "$cores" \
    python3 -u "$SCRIPT" \
        --ham   "$ham" \
        --quad  "$quad" \
        --out   "$out_npz" \
        --checkpoint-dir "$ckpt_dir" \
        --parallel-sectors 1 \
        --source "$source" \
        --decompose-H \
        --temp-min 1e-3 \
        --temp-max 5.0 \
        --temp-points 100 \
        > "$log" 2>&1
    echo "[done] $tag"
}

# ---------------------------------------------------------------------------
# Parallel execution helpers: run two jobs, slot A and slot B
# ---------------------------------------------------------------------------
PIDS=()
SLOTS=("$CORES_A" "$CORES_B")
THREADS=("$THREADS_A" "$THREADS_B")

wait_slot() {
    local slot_idx="$1"
    if [[ -n "${PIDS[$slot_idx]:-}" ]]; then
        wait "${PIDS[$slot_idx]}" || true
        PIDS[$slot_idx]=""
    fi
}

wait_all() {
    wait_slot 0
    wait_slot 1
}

# ---------------------------------------------------------------------------
# Queue: (scan_root, tag, source, [j3_dir])
# ---------------------------------------------------------------------------

# Helper to dispatch to the next free slot (simple round-robin queue)
QUEUE_IDX=0
dispatch() {
    local scan_root="$1"
    local tag="$2"
    local source="$3"
    local j3_dir="${4:-J3_0.0000}"

    local slot=$(( QUEUE_IDX % 2 ))
    wait_slot "$slot"
    QUEUE_IDX=$(( QUEUE_IDX + 1 ))

    # Explicitly pass cores and threads as positional args after j3_dir
    local cores="${SLOTS[$slot]}"
    local threads="${THREADS[$slot]}"
    run_case "$scan_root" "$tag" "$source" "$cores" "$threads" "$j3_dir" &
    PIDS[$slot]=$!
}

echo "================================================================"
echo "H-decomp campaign: $(date)"
echo "ROOT=$ROOT"
echo "================================================================"

# jpm03 (Jpm=-0.3), J3=0.08
dispatch \
    "twist_qsi_demo/output/jpm03_j3_verify_pbc_qsus" \
    "EgQ1_1e-3_j3_0.08_decomp" \
    "Eg_Q1=${LAMBDA}" \
    "J3_0.0800"

dispatch \
    "twist_qsi_demo/output/jpm03_j3_verify_pbc_qsus" \
    "T2gXZ_1e-3_j3_0.08_decomp" \
    "T2g_Q_xz=${LAMBDA}" \
    "J3_0.0800"

# jpm03 (Jpm=-0.3), J3=0
dispatch \
    "twist_qsi_demo/output/jpm03_j3_verify_pbc_qsus" \
    "EgQ1_1e-3_j3zero_decomp" \
    "Eg_Q1=${LAMBDA}" \
    "J3_0.0000"

dispatch \
    "twist_qsi_demo/output/jpm03_j3_verify_pbc_qsus" \
    "T2gXZ_1e-3_j3zero_decomp" \
    "T2g_Q_xz=${LAMBDA}" \
    "J3_0.0000"

# jpm02 (Jpm=-0.2), J3=0
dispatch \
    "twist_qsi_demo/output/jpm02_j3_scan_symm_light" \
    "EgQ1_1e-3_j3zero_decomp" \
    "Eg_Q1=${LAMBDA}" \
    "J3_0.0000"

dispatch \
    "twist_qsi_demo/output/jpm02_j3_scan_symm_light" \
    "T2gXZ_1e-3_j3zero_decomp" \
    "T2g_Q_xz=${LAMBDA}" \
    "J3_0.0000"

# jpm01 (Jpm=-0.1), J3=0
dispatch \
    "twist_qsi_demo/output/jpm01_j3_scan_symm_light" \
    "EgQ1_1e-3_j3zero_decomp" \
    "Eg_Q1=${LAMBDA}" \
    "J3_0.0000"

dispatch \
    "twist_qsi_demo/output/jpm01_j3_scan_symm_light" \
    "T2gXZ_1e-3_j3zero_decomp" \
    "T2g_Q_xz=${LAMBDA}" \
    "J3_0.0000"

# jpm_neg004 (Jpm=-0.04, pi-flux), J3=0 — T2g only (Eg already done/running)
dispatch \
    "twist_qsi_demo/output/jpm_neg004_piflux_j3zero" \
    "T2gXZ_1e-3_decomp" \
    "T2g_Q_xz=${LAMBDA}" \
    "J3_0.0000"

# jpm_neg002 (Jpm=-0.02, pi-flux), J3=0
dispatch \
    "twist_qsi_demo/output/jpm_neg002_piflux_j3zero" \
    "EgQ1_1e-3_decomp" \
    "Eg_Q1=${LAMBDA}" \
    "J3_0.0000"

dispatch \
    "twist_qsi_demo/output/jpm_neg002_piflux_j3zero" \
    "T2gXZ_1e-3_decomp" \
    "T2g_Q_xz=${LAMBDA}" \
    "J3_0.0000"

# jpm_pos002 (Jpm=+0.02, 0-flux), J3=0 — T2g only (Eg already done)
dispatch \
    "twist_qsi_demo/output/jpm_pos002_zeroflux_j3zero" \
    "T2gXZ_1e-3_decomp" \
    "T2g_Q_xz=${LAMBDA}" \
    "J3_0.0000"

# jpm_pos004 (Jpm=+0.04, 0-flux), J3=0 — T2g only (Eg already done)
dispatch \
    "twist_qsi_demo/output/jpm_pos004_zeroflux_j3zero" \
    "T2gXZ_1e-3_decomp" \
    "T2g_Q_xz=${LAMBDA}" \
    "J3_0.0000"

wait_all

echo "================================================================"
echo "All H-decomp cases submitted. Done at: $(date)"
echo "================================================================"
