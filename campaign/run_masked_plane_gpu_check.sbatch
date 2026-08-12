#!/bin/bash
#SBATCH --account=def-ybkim_gpu
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16 --mem=120G
#SBATCH --gpus-per-node=1
#SBATCH --time=2:00:00
#SBATCH --job-name=masked_gpu
#SBATCH --output=/lustre09/project/6003507/zhouzb79/twist_qsi/campaign/outputs/masked_band_complete/gpu_check_%j.log
# GPU parity + timing check for the Klein-blocked bordered solver:
# rerun the Gate A' cells with cuSOLVER eigh (--gpu) and diff the levels and
# ice weights against the CPU gate outputs.  Submit only AFTER the CPU gates
# job has written band_-0p050_fullbasis_klein.json / band_-0p280_...
set -uo pipefail
module purge
module load StdEnv/2023 gcc/12.3 cuda/12.6 python/3.11.5 scipy-stack/2024b
export FLEXIBLAS=AOCL OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

VENV=$SLURM_TMPDIR/cupy-venv
python3 -m venv --system-site-packages "$VENV"
source "$VENV/bin/activate"
pip install --no-index cupy | tail -1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

cd /lustre09/project/6003507/zhouzb79/twist_qsi/campaign

echo "=== GPU eigh benchmark (16384, vectors) ==="
python3 -u - <<'EOF'
import time
import numpy as np
import cupy
rng = np.random.default_rng(0)
a = rng.standard_normal((16384, 16384)); a = a + a.T
device = cupy.asarray(a)
cupy.linalg.eigh(cupy.asarray(rng.standard_normal((512, 512))))  # warm-up
cupy.cuda.Stream.null.synchronize()
t0 = time.perf_counter()
cupy.linalg.eigh(device)
cupy.cuda.Stream.null.synchronize()
dt = time.perf_counter() - t0
print(f"16384 eigh on GPU: {dt:.1f}s  (CPU at 91 GFLOPS: ~480s)")
EOF

echo "=== GPU rerun of the Gate A' cells ==="
python3 -u run_masked_band_complete.py --full-basis --gpu --tag-suffix _gpu \
    --window 0.40 --grid 6000 -0.05 -0.28

echo "=== parity: GPU vs CPU gate outputs ==="
python3 -u - <<'EOF'
import json
import numpy as np
worst = 0.0
for tag in ("-0p050", "-0p280"):
    cpu = json.load(open(f"outputs/masked_band_complete/band_{tag}_fullbasis_klein.json"))
    gpu = json.load(open(f"outputs/masked_band_complete/band_{tag}_fullbasis_klein_gpu.json"))
    for key in ("levels", "ice_weight"):
        a = np.asarray(cpu[key], float)
        b = np.asarray(gpu[key], float)
        mask = np.isfinite(a) & np.isfinite(b)
        assert mask.sum() == np.isfinite(a).sum() == np.isfinite(b).sum(), \
            f"{tag}/{key}: differing lost-level pattern"
        deviation = float(np.abs(a[mask] - b[mask]).max())
        worst = max(worst, deviation)
        print(f"{tag} {key}: max |cpu - gpu| = {deviation:.3e}")
        assert deviation < 1e-8, f"{tag}/{key} parity failed"
print(f"GPU PARITY OK (worst {worst:.3e})")
EOF
echo "exit=$?  $(date)"
