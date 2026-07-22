"""Per-sector band dimension at FCC-32 theta=0, which sets the block widths.

The band is the one continuously connected to Ran(P_ice), so its dimension in
each momentum sector equals the ice space's dimension there. Cubic-16 showed
the split is NOT uniform -- {36,18,18,18} for 90 over four sectors -- and
sizing by the average starves the largest sector until the ice projection goes
rank deficient.
"""
import os, sys, time
import numpy as np
TW = "/lustre09/project/6003507/zhouzb79/twist_qsi"
sys.path[:0] = [TW + "/notes", TW + "/src", TW + "/campaign/pilot"]
import qed._core as core
from recompute_finite_size_artifact import build_cluster
from pilot_fcc32_stage_a import fcc32_translations, write_directory

cl = build_cluster("fcc", (2, 2, 2))
n_up, n_ice = cl.n_sites // 2, len(cl.ice_states)
root = TW + "/campaign/pilot/fcc32_dir"
os.makedirs(root, exist_ok=True)
write_directory(root, cl, np.zeros(3), fcc32_translations(cl))
t0 = time.perf_counter()
sectors = core.sector_operators(root, cl.n_sites, 0.5, n_up)
print(f"sector build {time.perf_counter()-t0:.1f}s", flush=True)

internal = np.array([(1 << cl.n_sites) - 1 - int(s) for s in cl.ice_states], dtype=np.uint64)
shares, weight = [], np.zeros(n_ice)
for k, s in enumerate(sectors):
    idx, amp = s.project_states(internal)
    weight += np.abs(amp) ** 2
    hit = idx[idx >= 0]
    shares.append(len(np.unique(hit)))
    print(f"  sector {k} qn={s.quantum_numbers}: dim={int(s.dimension):,}  "
          f"ice-space dim={shares[-1]}", flush=True)

shares = np.array(shares)
print(f"\ntotal ice-space dim across sectors = {shares.sum()}  (n_ice = {n_ice})")
print(f"completeness = {np.abs(weight-1).max():.3e}")
print(f"largest share {shares.max()}, mean {shares.mean():.1f}, "
      f"ratio {shares.max()/shares.mean():.2f}x")

width = int(np.ceil(shares.max() * 1.11))     # cubic-16 optimum was 40/36
per_vec_gb = int(sectors[0].dimension) * 8 / 1e9
print(f"\nblock width (largest share x1.11) = {width}")
print(f"  fp32 vector {per_vec_gb:.2f} GB -> widest block {width*per_vec_gb:.0f} GB")
print(f"  all sectors resident: {sum(np.ceil(shares*1.11))*per_vec_gb:.0f} GB")
cols = float(np.ceil(shares * 1.11).sum())
print(f"\nmatvecs = 3 outer x 26 x {cols:.0f} block columns = {3*26*cols:.3e}")
print(f"  at 1.03 s/matvec -> {3*26*cols*1.03/3600:.0f} GPU-hours")
