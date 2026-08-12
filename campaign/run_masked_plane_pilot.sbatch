#!/bin/bash
#SBATCH --account=def-ybkim_cpu
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=192 --mem=750G --time=12:00:00
#SBATCH --exclusive
#SBATCH --job-name=masked_pilot
#SBATCH --output=/lustre09/project/6003507/zhouzb79/twist_qsi/campaign/outputs/masked_band_complete/pilot_%j.log
# Gate A of the (Jpm, Jpmpm) bordered-band campaign: the full 65,536-state
# basis at Jpmpm = 0 must reproduce the fixed-Sz oracle exactly (couplings
# to Sz != 0 poles vanish identically).  Also the timing pilot for the
# 65,446-dim QHQ eigendecomposition that fixes the per-cell budget.
set -uo pipefail
module purge
module load StdEnv/2023 gcc/12.3 openmpi/4.1.5 python/3.11.5 aocl-blas/5.1 aocl-lapack/5.1 hdf5/1.14.6
export FLEXIBLAS=AOCL OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONPATH=/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v4/Compiler/gcccore/scipy-stack/2024b/lib/python3.11/site-packages

cd /lustre09/project/6003507/zhouzb79/twist_qsi/campaign

echo "=== BLAS threading benchmark (dsyevd 8192, vectors) ==="
python3 -u - <<'EOF'
import time
import numpy as np
from scipy.linalg import eigh
rng = np.random.default_rng(0)
a = rng.standard_normal((8192, 8192)); a = a + a.T
t0 = time.perf_counter()
eigh(a, overwrite_a=True, check_finite=False, driver="evd")
dt = time.perf_counter() - t0
flops = 10.0 * 8192**3
print(f"8192 eigh: {dt:.1f}s  ~{flops/dt/1e9:.0f} GFLOPS effective")
print(f"projected 65446 eigh: {dt * (65446/8192)**3 / 60:.0f} min")
EOF

echo "=== Gate A: full basis at pp=0, isolated regime ==="
python3 -u run_masked_band_complete.py --full-basis --window 0.40 --grid 6000 -0.05
echo "=== Gate A: full basis at pp=0, BIC-recovery regime ==="
python3 -u run_masked_band_complete.py --full-basis --window 0.40 --grid 6000 -0.28
echo "=== fixed-Sz references at the same couplings (refactored build_blocks) ==="
python3 -u run_masked_band_complete.py --window 0.40 --grid 6000 -0.05 -0.28
echo "exit=$?  $(date)"
