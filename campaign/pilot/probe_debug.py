"""Diagnostic: pinpoint the frame where project_states crashes. No fixes."""
import faulthandler, sys, tempfile
faulthandler.enable()
sys.path[:0] = ["/lustre09/project/6003507/zhouzb79/twist_qsi/notes",
                "/lustre09/project/6003507/zhouzb79/twist_qsi/src",
                "/lustre09/project/6003507/zhouzb79/twist_qsi/campaign/pilot"]
import numpy as np
import qed._core as core
from recompute_finite_size_artifact import build_cluster
from pilot_cubic16_sector import translation_permutations, write_directory

cl = build_cluster("cubic", (1, 1, 1))
perms = translation_permutations(cl)
with tempfile.TemporaryDirectory() as root:
    write_directory(root, cl, np.zeros(3), perms)
    sectors = core.sector_operators(root, cl.n_sites, 0.5, 8)
print(f"built {len(sectors)} sectors", flush=True)

n = cl.n_sites
comp = np.array([(1 << n) - 1 - int(s) for s in cl.ice_states], dtype=np.uint64)

# 1. does a ONE-state call survive?  isolates the loop from the setup
print("--- single state, sector 0 ---", flush=True)
one = comp[:1].copy()
idx, amp = sectors[0].project_states(one)
print(f"  ok: idx={idx[0]} amp={amp[0]}", flush=True)

# 2. grow the batch to see whether it is size dependent
for k in (2, 10, 45, 90):
    print(f"--- {k} states, sector 0 ---", flush=True)
    idx, amp = sectors[0].project_states(comp[:k].copy())
    print(f"  ok: found={(idx>=0).sum()}  sum|amp|^2={np.abs(amp)**2 @ np.ones(k):.6f}", flush=True)

# 3. all sectors
tot = np.zeros(len(comp))
for j, s in enumerate(sectors):
    print(f"--- sector {j} full ---", flush=True)
    idx, amp = s.project_states(comp)
    tot += np.abs(amp) ** 2
    print(f"  ok: found={(idx>=0).sum()}", flush=True)
print(f"completeness: max|sum_k |<k|ice>|^2 - 1| = {np.abs(tot-1).max():.3e}", flush=True)
