"""How many matvecs does h(theta) actually need?

Two corrections to the first measurement:
  * request only the band's share per sector, not one column per ice state --
    the surplus states live above the gap in the dense continuum and the filter
    grinds on them forever;
  * stop on h, not on eigenvector residuals. The polar pullback depends on the
    SUBSPACE, so h converged to 1.5e-10 while residuals were still 1e-4.
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
from chebfsi import MatvecCounter, spectral_bounds, chebyshev_filter

cl = build_cluster("cubic", (1, 1, 1))
n_up, n_band = cl.n_sites // 2, len(cl.ice_states)
shift = 0.5 * len(cl.tets)
ref = np.load(TW + "/campaign/outputs/nonperturbative_cubic16_p0p046000.npz")
M, index = 4, (1, 1, 2)
theta = 2.0 * np.pi * np.asarray(index, float) / M
lex = list(itertools.product(range(M), repeat=3)).index(index)
frozen = ref[f"M{M}_operators_by_character"][lex]

with tempfile.TemporaryDirectory() as root:
    write_directory(root, cl, theta, translation_permutations(cl))
    sectors = core.sector_operators(root, cl.n_sites, 0.5, n_up)
internal = np.array([(1 << cl.n_sites) - 1 - int(s) for s in cl.ice_states], dtype=np.uint64)

# gathers and per-sector band share (from the reference band's sector counts)
gathers, applies = [], []
for sector in sectors:
    dim = int(sector.dimension)
    idx, amp = sector.project_states(internal)
    g = np.zeros((dim, n_band), dtype=np.complex128)
    hit = idx >= 0
    g[idx[hit], np.nonzero(hit)[0]] = amp[hit]
    gathers.append(g)
    applies.append(lambda v, s=sector: s.apply(np.ascontiguousarray(v)) + shift * v)

# The band is NOT evenly split across sectors: the reference records
# {++:36, +-:18, -+:18, --:18} for 90 states over 4 sectors. Sizing by the
# average starves the largest sector and the ice projection goes rank
# deficient, so size by the largest share plus a margin.
for per_sector in (40, 56):
    counters = [MatvecCounter(a) for a in applies]
    blocks, cuts, his = [], [], []
    for g, mv in zip(gathers, counters):
        seed = g[:, np.abs(g).sum(axis=0) > 0][:, :per_sector]
        b, _ = np.linalg.qr(seed)
        blocks.append(b)
        lo, hi = spectral_bounds(mv, b.shape[0])
        his.append(hi)
        s = b.conj().T @ mv(b)
        cuts.append(float(np.linalg.eigvalsh(0.5 * (s + s.conj().T)).max()))
    print(f"\n=== per_sector={per_sector} (largest sector share is 36) ===", flush=True)
    t0 = time.perf_counter()
    for outer in range(1, 13):
        energies, projected = [], []
        for k, (mv, g) in enumerate(zip(counters, gathers)):
            blocks[k] = chebyshev_filter(mv, blocks[k], 25, cuts[k], his[k])
            blocks[k], _ = np.linalg.qr(blocks[k])
            p = blocks[k].conj().T @ mv(blocks[k])
            p = 0.5 * (p + p.conj().T)
            vals, sv = np.linalg.eigh(p)
            blocks[k] = blocks[k] @ sv
            cuts[k] = float(vals[-1])
            v = blocks[k] @ inverse_sqrt_hermitian(blocks[k].conj().T @ blocks[k])
            energies.append(vals); projected.append(g.conj().T @ v)
        all_e = np.concatenate(energies); all_p = np.hstack(projected)
        keep = np.argsort(all_e)[:n_band]
        total = sum(c.count for c in counters)
        # Until the blocks settle onto the band, the globally lowest n_band
        # Ritz vectors need not span the ice space, and the Gram is singular.
        # That is a legitimate "not converged yet" signal, not a failure.
        try:
            h, _ = band_operator_from_projected_vectors(all_e[keep], all_p[:, keep])
            err = np.abs(h - frozen).max()
            shown = f"max|h-h_frozen|={err:.3e}"
        except ValueError as exc:
            err = np.inf
            shown = f"ice projection still rank deficient ({str(exc).split('=')[-1]})"
        print(f"  outer {outer:2d}: matvecs={total:6d} ({total/n_band:6.1f}/col)  {shown}",
              flush=True)
        if err < 1e-10:
            print(f"  CONVERGED at {total} matvecs = {total/n_band:.1f} per band column "
                  f"({time.perf_counter()-t0:.1f}s)", flush=True)
            print(f"  -> FCC-32: {total/n_band*2970:.3e} matvecs/sector-solve set", flush=True)
            break
