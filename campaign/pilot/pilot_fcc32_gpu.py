"""FCC-32 GPU matvec benchmark: is the device path fast enough to make the
campaign feasible?  CPU measured 23.0 s/matvec -> ~35,000 node-hours, which is dead.
"""
import os, resource, sys, time
import numpy as np
TW = "/lustre09/project/6003507/zhouzb79/twist_qsi"
sys.path[:0] = [TW + "/notes", TW + "/src", TW + "/campaign/pilot"]
import qed
import qed._core as core
from recompute_finite_size_artifact import build_cluster
from pilot_fcc32_stage_a import fcc32_translations, write_directory

def rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 ** 2)

print(f"has_cuda_build = {qed.has_cuda_build()}", flush=True)
cl = build_cluster("fcc", (2, 2, 2))
n_up = cl.n_sites // 2
theta = 2.0 * np.pi * np.array([0.0, 1.0, 1.0]) / 3.0
root = TW + "/campaign/pilot/fcc32_dir"
os.makedirs(root, exist_ok=True)
write_directory(root, cl, theta, fcc32_translations(cl))

t0 = time.perf_counter()
sectors = core.sector_operators(root, cl.n_sites, 0.5, n_up)
print(f"sector build {time.perf_counter()-t0:.1f}s  rss={rss_gb():.1f} GB", flush=True)

s0 = sectors[0]
dim = int(s0.dimension)
nnz_per_row = 96 * 2.0 * n_up * (cl.n_sites - n_up) / (cl.n_sites * (cl.n_sites - 1))
nnz = dim * nnz_per_row
print(f"sector 0: dim={dim:,}  nnz={nnz:.3e}  vector={dim*16/1e9:.2f} GB", flush=True)

v = np.zeros(dim, dtype=np.complex128); v[::1000] = 1.0

# --- CPU reference on this same node ---
t0 = time.perf_counter(); ref = s0.apply(v); cpu = time.perf_counter() - t0
print(f"\nCPU apply: {cpu:.2f}s  ({cpu/nnz*1e9:.2f} ns/nnz)", flush=True)

# --- GPU ---
t0 = time.perf_counter()
s0.use_device()
bind = time.perf_counter() - t0
print(f"bind_cuda: {bind:.2f}s   on_device={s0.on_device}  rss={rss_gb():.1f} GB", flush=True)

t0 = time.perf_counter(); got = s0.apply(v); first = time.perf_counter() - t0
print(f"GPU apply (first): {first:.3f}s", flush=True)
times = []
for _ in range(5):
    t0 = time.perf_counter(); s0.apply(v); times.append(time.perf_counter() - t0)
best = min(times)
print(f"GPU apply, incl H2D+D2H (best of 5): {best:.3f}s", flush=True)

# CORRECTNESS FIRST -- a device kernel handed host pointers returns instantly
# and writes nothing, which reads as a spectacular speedup producing zeros.
err = np.abs(got - ref).max()
print(f"  max|GPU - CPU| = {err:.3e}   (|ref|max = {np.abs(ref).max():.3e})", flush=True)
if err > 1e-8 * max(np.abs(ref).max(), 1.0):
    print("  *** DEVICE RESULT IS WRONG -- timings below are meaningless ***", flush=True)
    sys.exit(1)
print("  device result matches CPU", flush=True)

kernel = s0.device_matvec_seconds(10)
print(f"GPU kernel only (resident, mean of 10): {kernel:.4f}s "
      f"({kernel/nnz*1e9:.3f} ns/nnz)", flush=True)
print(f"  transfer overhead per call: {best-kernel:.3f}s "
      f"({2*dim*16/1e9:.2f} GB round trip)", flush=True)
print(f"  speedup vs CPU on this node: kernel {cpu/kernel:.1f}x, "
      f"end-to-end {cpu/best:.1f}x", flush=True)
best = kernel   # resident vectors are what ChebFSI would use
print(f"\nprojected campaign cost:", flush=True)
print(f"  per sector-solve (~7.4e4 matvecs): {7.4e4*best/3600:.1f} h", flush=True)
print(f"  72 sector-solves:                  {72*7.4e4*best/3600:.0f} GPU-hours", flush=True)
for nvec in (2, 4, 8):
    block = np.zeros((nvec, dim), dtype=np.complex128); block[:, ::1000] = 1.0
    t0 = time.perf_counter(); s0.apply_block(block); dt = time.perf_counter() - t0
    print(f"  apply_block({nvec}): {dt:.3f}s -> {dt/nvec:.3f}s per vector", flush=True)
