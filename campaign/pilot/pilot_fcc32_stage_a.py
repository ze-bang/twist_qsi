"""FCC-32 stage A: cost and correctness of everything BEFORE band extraction.

Measures the symmetry-adapted basis build, validates the ice gather at scale via
the convention-independent completeness check, and times apply/apply_block to
replace the extrapolated matvec cost with a measured one.

Deliberately does NOT extract the band: eigsh with k~371 at dim ~7.5e7 would
need ~900 GB of Krylov vectors. That is the Chebyshev-filtered subspace
iteration, and it needs this measurement first.
"""

from __future__ import annotations

import itertools, json, os, resource, sys, time
import numpy as np

TW = "/lustre09/project/6003507/zhouzb79/twist_qsi"
sys.path[:0] = [TW + "/notes", TW + "/src", TW + "/campaign/pilot"]

import qed
import qed._core as core
from qed.workflow import _write_operator_directory
from recompute_finite_size_artifact import build_cluster

JPM = 0.046


def rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 ** 2)


def fcc32_translations(cluster):
    """The eight (Z_2)^3 translations of the 2x2x2 primitive FCC cluster."""
    from recompute_finite_size_artifact import FCC_A
    basis = np.asarray(FCC_A, dtype=float)
    inv = np.linalg.inv(cluster.Lvecs)
    key = lambda p: tuple(np.rint(np.mod(p @ inv, 1.0) * 8).astype(int) % 8)
    site_of = {key(p): i for i, p in enumerate(cluster.positions)}
    perms = []
    for shift in itertools.product((0, 1), repeat=3):
        offset = np.asarray(shift, float) @ basis
        perm = [site_of[key(p + offset)] for p in cluster.positions]
        assert sorted(perm) == list(range(cluster.n_sites))
        perms.append(perm)
    return perms


def write_directory(root, cluster, theta, perms):
    auto = os.path.join(root, "automorphism_results")
    os.makedirs(auto, exist_ok=True)
    json.dump(perms, open(os.path.join(auto, "max_clique.json"), "w"))
    gens = [{"permutation": perms[i], "order": 2} for i in (1, 2, 4)]  # (0,0,1),(0,1,0),(1,0,0)
    json.dump({"generators": gens}, open(os.path.join(auto, "minimal_generators.json"), "w"))
    sectors = []
    for sid, chi in enumerate(itertools.product((0, 1), repeat=3)):
        sectors.append({"sector_id": sid, "quantum_numbers": list(chi),
                        "phase_factors": [{"real": float((-1) ** c), "imag": 0.0} for c in chi]})
    json.dump({"sectors": sectors}, open(os.path.join(auto, "sector_metadata.json"), "w"))

    op = qed.Operator(cluster.n_sites, 0.5)
    for (i, j) in cluster.bonds:
        op.add_two_body(core.OP_SZ, int(i), core.OP_SZ, int(j), 1.0)
    for (i, j), wrap in zip(cluster.bonds, cluster.bond_wrap):
        d = (cluster.positions[j] - np.asarray(wrap, float) @ cluster.Lvecs
             - cluster.positions[i])
        ph = np.exp(2j * float(theta @ d))     # sign convention: cubic-16 validated
        op.add_two_body(core.OP_SPLUS, int(i), core.OP_SMINUS, int(j), -JPM * ph)
        op.add_two_body(core.OP_SPLUS, int(j), core.OP_SMINUS, int(i), -JPM * ph.conjugate())
    _write_operator_directory(op, root)


def main():
    cl = build_cluster("fcc", (2, 2, 2))
    n_up = cl.n_sites // 2
    print(f"FCC-32: {cl.n_sites} sites, {len(cl.bonds)} bonds, {len(cl.tets)} tets, "
          f"ice rank {len(cl.ice_states)}", flush=True)
    perms = fcc32_translations(cl)
    print(f"translations: {len(perms)} (expect 8)", flush=True)

    theta = 2.0 * np.pi * np.array([0.0, 1.0, 1.0]) / 3.0   # an M=3 orbit representative
    root = os.environ.get("PILOT_DIR", "/lustre09/project/6003507/zhouzb79/twist_qsi/campaign/pilot/fcc32_dir")
    os.makedirs(root, exist_ok=True)
    t0 = time.perf_counter()
    write_directory(root, cl, theta, perms)
    print(f"directory written in {time.perf_counter()-t0:.1f}s  rss={rss_gb():.1f} GB", flush=True)

    t0 = time.perf_counter()
    sectors = core.sector_operators(root, cl.n_sites, 0.5, n_up,
                                    os.environ.get("ED_SYM_CACHE_DIR"))
    build = time.perf_counter() - t0
    dims = [int(s.dimension) for s in sectors]
    print(f"\nsector build: {build:.1f}s   rss={rss_gb():.1f} GB", flush=True)
    print(f"  dims  = {dims}", flush=True)
    print(f"  sum   = {sum(dims):,}   (C(32,16) = 601,080,390)", flush=True)

    # ---- the ice gather at scale -------------------------------------------
    ice = np.array([int(s) for s in cl.ice_states], dtype=np.uint64)
    t0 = time.perf_counter()
    weight = np.zeros(len(ice))
    found = 0
    for s in sectors:
        idx, amp = s.project_states(ice)
        weight += np.abs(amp) ** 2
        found += int((idx >= 0).sum())
    print(f"\nice gather: {time.perf_counter()-t0:.2f}s for {len(ice)} states, "
          f"{found} (sector,state) hits", flush=True)
    print(f"  completeness max|sum_k |<k|ice>|^2 - 1| = {np.abs(weight-1).max():.3e}", flush=True)

    # ---- matvec cost --------------------------------------------------------
    s0 = sectors[0]
    dim = int(s0.dimension)
    print(f"\nmatvec on sector 0 (dim {dim:,}, {dim*16/1e9:.2f} GB per complex128 vector)",
          flush=True)
    v = np.zeros(dim, dtype=np.complex128); v[::1000] = 1.0
    s0.apply(v)                                            # warm
    t0 = time.perf_counter(); s0.apply(v); single = time.perf_counter() - t0
    nnz_per_row = 96 * 2.0 * n_up * (cl.n_sites - n_up) / (cl.n_sites * (cl.n_sites - 1))
    nnz = dim * nnz_per_row
    print(f"  apply      : {single:.2f}s   ({single/nnz*1e9:.2f} ns/nnz, "
          f"nnz/row={nnz_per_row:.1f})", flush=True)
    for nvec in (2, 4):
        block = np.zeros((nvec, dim), dtype=np.complex128); block[:, ::1000] = 1.0
        t0 = time.perf_counter(); s0.apply_block(block); dt = time.perf_counter() - t0
        print(f"  apply_block({nvec}): {dt:.2f}s  ({dt/nvec:.2f}s per vector)", flush=True)
    print(f"\npeak rss = {rss_gb():.1f} GB", flush=True)


if __name__ == "__main__":
    main()
