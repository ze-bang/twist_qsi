"""Minimal probe: is the orbit data materialised, and does project_states resolve?"""
import sys, numpy as np
sys.path[:0] = ["/lustre09/project/6003507/zhouzb79/twist_qsi/notes",
                "/lustre09/project/6003507/zhouzb79/twist_qsi/src"]
sys.path.insert(0, "/lustre09/project/6003507/zhouzb79/twist_qsi/campaign/pilot")
import qed._core as core
from recompute_finite_size_artifact import build_cluster
from pilot_cubic16_sector import translation_permutations, write_directory
import tempfile, numpy as np

cl = build_cluster("cubic", (1, 1, 1))
perms = translation_permutations(cl)
with tempfile.TemporaryDirectory() as root:
    write_directory(root, cl, np.zeros(3), perms)
    sectors = core.sector_operators(root, cl.n_sites, 0.5, 8)

n = cl.n_sites
ice = np.array([int(s) for s in cl.ice_states], dtype=np.uint64)
comp = np.array([(1 << n) - 1 - int(s) for s in cl.ice_states], dtype=np.uint64)

for k, s in enumerate(sectors):
    print(f"sector {k}: dim={s.dimension}  materialized_orbits={s.materialized_orbits}")

print("\n-- project_states on the two label conventions --")
for name, arr in (("complement", comp), ("as-is", ice)):
    tot = np.zeros(len(arr))
    found = 0
    for s in sectors:
        try:
            idx, amp = s.project_states(arr)
        except RuntimeError as exc:
            print(f"  {name}: RuntimeError -> {str(exc)[:120]}")
            tot = None
            break
        tot += np.abs(amp) ** 2
        found += int((idx >= 0).sum())
    if tot is not None:
        print(f"  {name}: indices found={found}, max|sum|amp|^2 - 1| = {np.abs(tot-1).max():.3e}")
