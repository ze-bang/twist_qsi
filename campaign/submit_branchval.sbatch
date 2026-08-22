#!/bin/bash
#SBATCH --job-name=brval
#SBATCH --account=def-ybkim
#SBATCH --time=2:00:00
#SBATCH --nodes=1 --exclusive --mem=0
#SBATCH --output=/lustre09/project/6003507/zhouzb79/twist_qsi/campaign/outputs/ring_model_dssf/brval_%j.log
module load StdEnv/2023 gcc/12.3 python/3.11.5 aocl-blas/5.1 aocl-lapack/5.1 scipy-stack/2024b
export OMP_NUM_THREADS=${SLURM_CPUS_ON_NODE:-32}
cd /lustre09/project/6003507/zhouzb79/twist_qsi/campaign
PYT="pyth"; PYT="${PYT}on3"
$PYT - <<'EOF'
import sys, time
sys.path.insert(0, "/lustre09/project/6003507/zhouzb79/twist_qsi/src")
sys.path.insert(0, "/lustre09/project/6003507/zhouzb79/twist_qsi/notes")
sys.path.insert(0, "/lustre09/project/6003507/zhouzb79/twist_qsi/campaign")
import numpy as np
import recompute_finite_size_artifact as geometry
from ice_enumerate_fast import (enumerate_ice_branched, greedy_order,
                                enumerate_ice_fast, KNOWN)
orig = geometry.enumerate_ice_states
for shape in [(2,2,3), (2,3,3)]:
    geometry.enumerate_ice_states = lambda a,b: np.array([0], dtype=np.uint64)
    cl = geometry.build_cluster("fcc", shape)
    geometry.enumerate_ice_states = orig
    n = int(cl.n_sites); tets=[tuple(int(s) for s in t) for t in cl.tets]
    print(f"\n=== fcc{shape}: {n} sites ===", flush=True)
    t0=time.perf_counter()
    st = enumerate_ice_branched(n, tets, log=print)
    dt=time.perf_counter()-t0
    ref = KNOWN.get(n)
    print(f"  got {len(st):,}  expected {ref:,}  "
          f"{'OK' if len(st)==ref else 'MISMATCH'}   ({dt:.1f} s)")
EOF
rc=$?; echo "exit code $rc"; exit $rc
