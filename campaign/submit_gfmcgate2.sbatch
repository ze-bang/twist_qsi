#!/bin/bash
#SBATCH --job-name=ggate2
#SBATCH --account=def-ybkim
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --output=/lustre09/project/6003507/zhouzb79/twist_qsi/campaign/outputs/ring_model_dssf/ggate2_%j.log
module load StdEnv/2023 gcc/12.3 python/3.11.5 aocl-blas/5.1 aocl-lapack/5.1 scipy-stack/2024b
export OMP_NUM_THREADS=${SLURM_CPUS_ON_NODE:-32}
export FLEXIBLAS=AOCL
cd /lustre09/project/6003507/zhouzb79/twist_qsi/campaign
PYT="pyth"; PYT="${PYT}on3"
echo "### GFMC 48-gate with the incremental kernel (exact: -10.556012 / 0.370365)"
$PYT -u gfmc_ice.py 2 2 3 40000 || exit 1
echo "### timing benchmark at 864 sites, 3000 steps (was ~0.20 s/step)"
$PYT - <<'EOF'
import sys, time
sys.path.insert(0, "/lustre09/project/6003507/zhouzb79/twist_qsi/src")
sys.path.insert(0, "/lustre09/project/6003507/zhouzb79/twist_qsi/notes")
sys.path.insert(0, "/lustre09/project/6003507/zhouzb79/twist_qsi/campaign")
import numpy as np
from pyro_lattice import build_lattice, extract_templates
import gfmc_ice as G
lat = build_lattice((6, 6, 6), extract_templates())
t0 = time.perf_counter()
res, lam = G.gfmc(lat, n_walkers=1024, n_steps=3000, n_equil=500,
                  qs=[np.array([0.5, 0.5, 0.5])], t0=t0)
E, Ee = G.binned(res["E_mix"])
dt = time.perf_counter() - t0
print(f"864 sites: {dt:.0f} s for 3000 steps = {dt/3000*1000:.1f} ms/step")
print(f"E/site = {E/864:.6f} +/- {Ee/864:.6f}  (fullgrid run: -0.189122(42))")
EOF
rc=$?; echo "exit code $rc"; exit $rc
