"""Does reduced-CSR beat the CSR-free rep walk at FCC-32 scale?

The rep walk canonicalises every nonzero through |G|=8 permutations; reduced-CSR
stores the matrix and the matvec becomes an ordinary SpMV. The trade is memory:
~45-60 GB for one sector, against a default aggregate budget of 8 GiB.
"""
import os, resource, sys, time
import numpy as np
TW = "/lustre09/project/6003507/zhouzb79/twist_qsi"
sys.path[:0] = [TW + "/notes", TW + "/src", TW + "/campaign/pilot"]
import qed._core as core
from recompute_finite_size_artifact import build_cluster
from pilot_fcc32_stage_a import fcc32_translations, write_directory

def rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 ** 2)

cl = build_cluster("fcc", (2, 2, 2))
n_up = cl.n_sites // 2
theta = 2.0 * np.pi * np.array([0.0, 1.0, 1.0]) / 3.0
root = TW + "/campaign/pilot/fcc32_dir"
os.makedirs(root, exist_ok=True)
write_directory(root, cl, theta, fcc32_translations(cl))

mode = os.environ.get("ED_SYM_REDUCED_CSR", "unset")
budget = os.environ.get("ED_SYM_SECTOR_CSR_BUDGET_GIB", "8")
print(f"ED_SYM_REDUCED_CSR={mode}  ED_SYM_SECTOR_CSR_BUDGET_GIB={budget}", flush=True)

t0 = time.perf_counter()
sectors = core.sector_operators(root, cl.n_sites, 0.5, n_up)
print(f"sector build {time.perf_counter()-t0:.1f}s  rss={rss_gb():.1f} GB", flush=True)

s0 = sectors[0]
dim = int(s0.dimension)
nnz_per_row = 96 * 2.0 * n_up * (cl.n_sites - n_up) / (cl.n_sites * (cl.n_sites - 1))
nnz = dim * nnz_per_row
v = np.zeros(dim, dtype=np.complex128); v[::1000] = 1.0

t0 = time.perf_counter(); s0.apply(v); first = time.perf_counter() - t0
print(f"first apply (may build CSR): {first:.2f}s  rss={rss_gb():.1f} GB", flush=True)
times = []
for _ in range(3):
    t0 = time.perf_counter(); s0.apply(v); times.append(time.perf_counter() - t0)
best = min(times)
print(f"steady apply: {best:.2f}s  ({best/nnz*1e9:.2f} ns/nnz)  rss={rss_gb():.1f} GB", flush=True)
print(f"  -> per sector-solve (~7.4e4 matvecs): {7.4e4*best/3600:.0f} h", flush=True)
print(f"  -> 72 sector-solves: {72*7.4e4*best/3600:.0f} node-hours", flush=True)
