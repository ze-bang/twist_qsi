"""Block-width sweep for the memory-bounded ChebFSI, against the frozen h(theta).

The x1.11 proportional margin starves small sectors -- an 18-state share gets 2
spare columns, where the converged run had 22 -- so convergence needs an
ABSOLUTE margin. Find the threshold here, because it sets both the FCC-32 block
widths and the peak memory.
"""
from __future__ import annotations
import itertools, sys, tempfile, time
import numpy as np
TW = "/lustre09/project/6003507/zhouzb79/twist_qsi"
sys.path[:0] = [TW + "/notes", TW + "/src", TW + "/campaign/pilot"]
import qed._core as core
from recompute_finite_size_artifact import build_cluster
from qsi_campaign.protocol import (band_operator_from_projected_vectors,
                                   centered_relative_error, inverse_sqrt_hermitian)
from pilot_cubic16_sector import translation_permutations, write_directory
from chebfsi import spectral_bounds
from chebfsi_lowmem import MatvecCounter, chebfsi_lowmem

cl = build_cluster("cubic", (1, 1, 1))
n_up, n_band = cl.n_sites // 2, len(cl.ice_states)
shift = 0.5 * len(cl.tets)
ref = np.load(TW + "/campaign/outputs/nonperturbative_cubic16_p0p046000.npz")
M, index = 4, (1, 1, 2)
theta = 2.0 * np.pi * np.asarray(index, float) / M
lex = list(itertools.product(range(M), repeat=3)).index(index)
assert tuple(ref[f"M{M}_character_indices"][lex]) == index
frozen = ref[f"M{M}_operators_by_character"][lex]

with tempfile.TemporaryDirectory() as root:
    write_directory(root, cl, theta, translation_permutations(cl))
    sectors = core.sector_operators(root, cl.n_sites, 0.5, n_up)
internal = np.array([(1 << cl.n_sites) - 1 - int(s) for s in cl.ice_states], dtype=np.uint64)

gathers, shares = [], []
for sector in sectors:
    idx, amp = sector.project_states(internal)
    g = np.zeros((int(sector.dimension), n_band), dtype=np.complex128)
    hit = idx >= 0
    g[idx[hit], np.nonzero(hit)[0]] = amp[hit]
    gathers.append(g)
    shares.append(len(np.unique(idx[idx >= 0])))
print(f"shares = {shares}", flush=True)

RULES = [("x1.11", lambda s: int(np.ceil(s * 1.11))),
         ("+12",   lambda s: s + 12),
         ("+24",   lambda s: s + 24),
         ("x2",    lambda s: 2 * s)]

for rule_name, rule in RULES:
    for dtype, label in ((np.complex128, "fp64"), (np.complex64, "fp32")):
        energies, projected, total = [], [], 0
        t0 = time.perf_counter()
        widths = []
        for sector, g, share in zip(sectors, gathers, shares):
            dim = int(sector.dimension)
            width = min(rule(share), dim - 2)
            widths.append(width)
            seed = g[:, np.abs(g).sum(axis=0) > 0][:, :width]
            block = np.ascontiguousarray(seed.astype(dtype))
            mv = MatvecCounter(lambda v, s=sector: s.apply(
                np.ascontiguousarray(v.astype(np.complex128))) + shift * v)
            _, hi = spectral_bounds(lambda v: mv.one(v), dim)
            small = np.column_stack([mv.one(block[:, j].astype(np.complex128))
                                     for j in range(block.shape[1])])
            small = block.conj().T.astype(np.complex128) @ small
            cut = float(np.linalg.eigvalsh(0.5 * (small + small.conj().T)).max())
            vals, block = chebfsi_lowmem(mv, block, share, degree=25, outer=3, cut=cut,
                                         upper=hi, row_chunk=1 << 18, col_chunk=32)
            total += mv.count
            v = block.astype(np.complex128)
            v = v @ inverse_sqrt_hermitian(v.conj().T @ v)
            energies.append(np.asarray(vals, float))
            projected.append(g.conj().T @ v)
        all_e = np.concatenate(energies); all_p = np.hstack(projected)
        keep = np.argsort(all_e)[:n_band]
        try:
            h, _ = band_operator_from_projected_vectors(all_e[keep], all_p[:, keep])
        except ValueError as exc:
            print(f"[{rule_name}/{label}] widths={widths} RANK DEFICIENT ({exc})", flush=True)
            continue
        print(f"[{rule_name}/{label}] widths={widths} matvecs={total} "
              f"({total/n_band:.1f}/col) {time.perf_counter()-t0:.1f}s  "
              f"max|h-h_frozen|={np.abs(h-frozen).max():.3e}", flush=True)
