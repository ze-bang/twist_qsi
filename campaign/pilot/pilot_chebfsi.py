"""Measure the ChebFSI matvec count on cubic-16, and check it finds the RIGHT band.

The campaign cost is linear in matvecs per sector-solve; 7.4e4 has been an
estimate throughout. Also sweeps fp32 storage of the iterates, which decides
whether an FCC-32 block is 223 GB or 446 GB.
"""
from __future__ import annotations
import itertools, os, sys, tempfile, time
import numpy as np
TW = "/lustre09/project/6003507/zhouzb79/twist_qsi"
sys.path[:0] = [TW + "/notes", TW + "/src", TW + "/campaign/pilot"]
import qed._core as core
from recompute_finite_size_artifact import build_cluster
from qsi_campaign.protocol import (band_operator_from_projected_vectors,
                                   centered_relative_error, inverse_sqrt_hermitian)
from pilot_cubic16_sector import translation_permutations, write_directory
from chebfsi import chebfsi, MatvecCounter

cl = build_cluster("cubic", (1, 1, 1))
n_up, n_band = cl.n_sites // 2, len(cl.ice_states)
shift = 0.5 * len(cl.tets)
ref = np.load(TW + "/campaign/outputs/nonperturbative_cubic16_p0p046000.npz")

M, index = 4, (1, 1, 2)
theta = 2.0 * np.pi * np.asarray(index, float) / M
lex = list(itertools.product(range(M), repeat=3)).index(index)
frozen = ref[f"M{M}_operators_by_character"][lex]
assert tuple(ref[f"M{M}_character_indices"][lex]) == index

with tempfile.TemporaryDirectory() as root:
    write_directory(root, cl, theta, translation_permutations(cl))
    sectors = core.sector_operators(root, cl.n_sites, 0.5, n_up)

internal = np.array([(1 << cl.n_sites) - 1 - int(s) for s in cl.ice_states], dtype=np.uint64)

for dtype, label in ((np.complex128, "fp64"), (np.complex64, "fp32")):
    total_matvecs = 0
    energies, projected = [], []
    t0 = time.perf_counter()
    for k, sector in enumerate(sectors):
        dim = int(sector.dimension)
        idx, amp = sector.project_states(internal)
        gather = np.zeros((dim, n_band), dtype=np.complex128)
        hit = idx >= 0
        gather[idx[hit], np.nonzero(hit)[0]] = amp[hit]
        seed = gather[:, np.abs(gather).sum(axis=0) > 0]
        if seed.shape[1] == 0:
            continue
        # storage precision of the iterates; the matvec itself stays fp64
        def apply(v, s=sector):
            out = s.apply(np.ascontiguousarray(v.astype(np.complex128)))
            return (out + shift * v).astype(dtype)
        mv = MatvecCounter(apply)
        want = min(seed.shape[1], dim - 2)
        vals, vecs, diag = chebfsi(mv, seed.astype(dtype), want, degree=25, tol=1e-11)
        total_matvecs += diag["matvecs"]
        print(f"  [{label}] sector {k}: dim={dim} want={want} "
              f"outer={diag['outer_iterations']} matvecs={diag['matvecs']} "
              f"resid={diag['max_residual']:.2e}", flush=True)
        vecs = vecs.astype(np.complex128)
        vecs = vecs @ inverse_sqrt_hermitian(vecs.conj().T @ vecs)
        energies.append(np.asarray(vals, dtype=float))
        projected.append(gather.conj().T @ vecs)

    all_e = np.concatenate(energies); all_p = np.hstack(projected)
    keep = np.argsort(all_e)[:n_band]
    h, _ = band_operator_from_projected_vectors(all_e[keep], all_p[:, keep])
    print(f"[{label}] TOTAL matvecs={total_matvecs}  ({time.perf_counter()-t0:.1f}s)")
    print(f"[{label}] max|h - h_frozen| = {np.abs(h-frozen).max():.3e}   "
          f"centered = {centered_relative_error(h, frozen):.3e}")
    per_col = total_matvecs / n_band
    print(f"[{label}] matvecs per band column = {per_col:.1f}  "
          f"-> FCC-32 estimate {per_col*2970:.3e} vs my 7.4e4 guess", flush=True)
