"""Local analysis of the saved 90x90 operators. No solving, no queue."""
import sys
import numpy as np
sys.path[:0] = ["notes", "src"]
from recompute_finite_size_artifact import build_cluster
from qsi_campaign.exact_band import basis_character_phases

cl = build_cluster("cubic", (1, 1, 1))
ice = cl.ice_states

for M, idx in ((3, (0, 1, 1)), (4, (1, 1, 2))):
    tag = f"M{M}_{''.join(map(str, idx))}"
    mine = np.load(f"campaign/pilot/h_mine_{tag}.npy")
    froz = np.load(f"campaign/pilot/h_frozen_{tag}.npy")
    theta = 2.0 * np.pi * np.asarray(idx, float) / M
    print(f"\n================ {tag}  theta={np.round(theta,4)} ================")

    sm = np.linalg.eigvalsh(mine)
    sf = np.linalg.eigvalsh(froz)
    print(f"spectra: max|diff| = {np.abs(sm - sf).max():.3e}")
    bad = np.abs(sm - sf) > 1e-9
    print(f"  levels disagreeing above 1e-9: {bad.sum()} / {len(sm)}")
    if bad.any():
        w = np.nonzero(bad)[0]
        print(f"  first disagreeing indices {w[:6]}")
        for i in w[:5]:
            print(f"    level {i:3d}: mine={sm[i]:+.9f}  frozen={sf[i]:+.9f}  d={sm[i]-sf[i]:+.3e}")

    # --- HYPOTHESIS: the reference carries the diagonal gauge phase that
    # basis_character_phases() applies on the ice basis.
    ph = basis_character_phases(cl, ice, theta)
    for name, cand in (
        ("diag(p) h_frozen diag(p)^*", ph[:, None] * froz * ph.conj()[None, :]),
        ("diag(p)^* h_frozen diag(p)", ph.conj()[:, None] * froz * ph[None, :]),
    ):
        print(f"  max|h_mine - {name}| = {np.abs(mine - cand).max():.3e}")

    # --- otherwise: extract the unitary, if the spectra allow it
    if np.abs(sm - sf).max() < 1e-8:
        gaps = np.diff(sf)
        print(f"  min spectral gap = {gaps.min():.3e} (degeneracies make U non-unique)")
        _, vm = np.linalg.eigh(mine)
        _, vf = np.linalg.eigh(froz)
        U = vm @ vf.conj().T
        offdiag = U - np.diag(np.diag(U))
        print(f"  U=Vm Vf^H : |diag| in [{np.abs(np.diag(U)).min():.3f}, "
              f"{np.abs(np.diag(U)).max():.3f}]  max|offdiag|={np.abs(offdiag).max():.3e}")
        # is it a permutation?
        col = np.abs(U).argmax(axis=0)
        print(f"  argmax per column is a permutation: {len(set(col.tolist())) == len(col)}")
        print(f"  max|U| per column min = {np.abs(U).max(axis=0).min():.3f}")
