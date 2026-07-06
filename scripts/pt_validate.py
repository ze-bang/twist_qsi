"""Validate the path-resolved SW effective Hamiltonian (ice_pt_lib) against
exact sparse diagonalization of the full XXZ Hamiltonian on the 16-site
cluster, at phi = 0, at twist corners, and at a generic twist.

Also prints the exact low spectrum so the quoted manuscript numbers
(gap 0.10603, "7-fold ground manifold", per-corner degeneracies) can be
audited independently of the C++ Lanczos pipeline.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import eigsh

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ice_pt_lib as ipl  # noqa: E402


def build_H_sparse(cl, Jpm, phi=(0.0, 0.0, 0.0)):
    """Full XXZ Hamiltonian in the Sz=0 sector, minimum-image twist gauge."""
    N = cl.n_sites
    states = np.array(
        [s for s in range(1 << N) if bin(s).count("1") == N // 2],
        dtype=np.uint64,
    )
    idx = {int(s): i for i, s in enumerate(states)}
    dim = len(states)
    phi = np.asarray(phi, dtype=float)

    Ez = np.zeros(dim)
    for (i, j), _n in zip(cl.bonds, cl.bond_wrap):
        szi = (((states >> np.uint64(i)) & np.uint64(1)).astype(float) - 0.5)
        szj = (((states >> np.uint64(j)) & np.uint64(1)).astype(float) - 0.5)
        Ez += szi * szj

    rows, cols, vals = [], [], []
    one = np.uint64(1)
    for (i, j), nij in zip(cl.bonds, cl.bond_wrap):
        bi, bj = one << np.uint64(i), one << np.uint64(j)
        flip = np.uint64(bi | bj)
        up_i = (states & bi) != 0
        up_j = (states & bj) != 0
        th = float(np.asarray(nij, float) @ phi)
        m = (~up_i) & up_j          # S+_i S-_j, phase e^{-i th}
        new = states[m] ^ flip
        rows.append(np.array([idx[int(x)] for x in new], dtype=np.int64))
        cols.append(np.nonzero(m)[0])
        vals.append(np.full(int(m.sum()), -Jpm, dtype=complex) * np.exp(-1j * th))
        m = up_i & (~up_j)          # S-_i S+_j, phase e^{+i th}
        new = states[m] ^ flip
        rows.append(np.array([idx[int(x)] for x in new], dtype=np.int64))
        cols.append(np.nonzero(m)[0])
        vals.append(np.full(int(m.sum()), -Jpm, dtype=complex) * np.exp(+1j * th))
    H = coo_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(dim, dim),
    ).tocsr()
    return H + diags(Ez)


def low_spectrum_exact(cl, Jpm, phi, k=110):
    H = build_H_sparse(cl, Jpm, phi)
    E = eigsh(H, k=k, which="SA", return_eigenvectors=False)
    E = np.sort(E)
    return E - E[0]


def low_spectrum_pt(cl, pt, Jpm, phi, order=4):
    """pt holds reference row tables computed at Jpm=1 (H_k ~ Jpm^k exactly,
    since the resolvents are pure Ising)."""
    H = (Jpm ** 2) * ipl.rows_to_matrix(cl, pt["H2"], phi=phi) \
        + (Jpm ** 3) * ipl.rows_to_matrix(cl, pt["H3"], phi=phi)
    if order >= 4 and "H4" in pt:
        H = H + (Jpm ** 4) * ipl.rows_to_matrix(cl, pt["H4"], phi=phi)
    E = np.linalg.eigvalsh(H)
    return E - E[0]


def main():
    scratch = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    cl = ipl.build_cluster("cubic", (1, 1, 1))
    print(f"16-site cluster: ice={cl.n_ice}, 4-loops={len(cl.loops4)}, "
          f"hexagons={len(cl.hexes)}")

    t0 = time.time()
    pt = ipl.sw_effective(cl, 1.0, order=4)
    print(f"reference PT run (Jpm=1): {time.time()-t0:.1f}s, "
          f"H2 rows={len(pt['H2']['c'])}, H3 rows={len(pt['H3']['c'])}, "
          f"H4 rows={len(pt['H4']['c'])}")

    pi = np.pi
    tests = [
        (-0.10, (0, 0, 0)),
        (-0.10, (pi, 0, 0)),
        (-0.10, (pi, pi, 0)),
        (-0.10, (pi, pi, pi)),
        (-0.10, (pi / 2, 0, 0)),      # generic twist: phase-convention check
        (+0.05, (0, 0, 0)),
        (+0.05, (pi, pi, pi)),
    ]
    results = {}
    for Jpm, phi in tests:
        t0 = time.time()
        Ee = low_spectrum_exact(cl, Jpm, phi, k=110)
        Ep3 = low_spectrum_pt(cl, pt, Jpm, phi, order=3)
        Ep4 = low_spectrum_pt(cl, pt, Jpm, phi, order=4)
        nc = min(len(Ee), 30)
        err3 = np.max(np.abs(Ee[:nc] - Ep3[:nc]))
        err4 = np.max(np.abs(Ee[:nc] - Ep4[:nc]))
        print(f"\nJpm={Jpm:+.2f} phi=({phi[0]/pi:.2f},{phi[1]/pi:.2f},"
              f"{phi[2]/pi:.2f})pi   [{time.time()-t0:.0f}s]")
        print("  exact:", np.array2string(np.round(Ee[:14], 5), max_line_width=200))
        print("  PT..3:", np.array2string(np.round(Ep3[:14], 5), max_line_width=200))
        print("  PT..4:", np.array2string(np.round(Ep4[:14], 5), max_line_width=200))
        print(f"  max|exact-PT| lowest {nc}:  order<=3: {err3:.2e}   order<=4: {err4:.2e}")
        results[f"{Jpm}_{phi}"] = dict(exact=Ee, pt=Ep4)

    np.savez(scratch / "pt_validation.npz",
             **{f"exact_{k}": v["exact"] for k, v in results.items()},
             **{f"pt_{k}": v["pt"] for k, v in results.items()})
    print(f"\nsaved {scratch/'pt_validation.npz'}")


if __name__ == "__main__":
    main()
