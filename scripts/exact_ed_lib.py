"""Exact sparse ED utilities for the small pyrochlore clusters (Sz=0 sector):

  * twisted XXZ Hamiltonian in the minimum-image (boundary-bond) gauge or
    the smooth (uniform vector potential) gauge,
  * explicit ring-exchange operator sums (winding 4-loops, hexagons) as bare
    sparse operators -- used to differentiate T_peak with respect to each
    emergent ring channel exactly, without perturbation theory,
  * exact Loewdin/des Cloizeaux downfolding of the lowest ice-manifold band
    onto the ice basis -> the exact effective Hamiltonian at a given twist,
    whose corner average is the operator-level twist average.

Complements ice_pt_lib (path-resolved SW PT), which supplies the winding
selection rules and the 32-site cluster where exact ED is out of reach.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import eigsh

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ice_pt_lib as ipl  # noqa: E402


class SzBasis:
    def __init__(self, cl, nup=None):
        N = cl.n_sites
        self.cl = cl
        self.nup = N // 2 if nup is None else nup
        self.states = np.array(
            [s for s in range(1 << N) if bin(s).count("1") == self.nup],
            dtype=np.uint64)
        self.index = {int(s): i for i, s in enumerate(self.states)}
        self.dim = len(self.states)
        self.ice_rows = (np.array([self.index[int(s)] for s in cl.ice_states])
                         if self.nup == N // 2 else None)
        # diagonal Ising energy
        Ez = np.zeros(self.dim)
        for (i, j) in cl.bonds:
            szi = (((self.states >> np.uint64(i)) & np.uint64(1)).astype(float) - 0.5)
            szj = (((self.states >> np.uint64(j)) & np.uint64(1)).astype(float) - 0.5)
            Ez += szi * szj
        self.Ez = Ez

    # ------------------------------------------------------------------
    def hop_matrix(self, phases=None):
        """sum_<ij> (S+_i S-_j e^{-i th_ij} + h.c.); phases per bond."""
        cl = self.cl
        one = np.uint64(1)
        rows, cols, vals = [], [], []
        if phases is None:
            phases = np.zeros(len(cl.bonds))
        for (i, j), th in zip(cl.bonds, phases):
            bi, bj = one << np.uint64(i), one << np.uint64(j)
            flip = np.uint64(bi | bj)
            up_i = (self.states & bi) != 0
            up_j = (self.states & bj) != 0
            for m, ph in (((~up_i) & up_j, np.exp(-1j * th)),
                          (up_i & (~up_j), np.exp(+1j * th))):
                new = self.states[m] ^ flip
                rows.append(np.array([self.index[int(x)] for x in new],
                                     dtype=np.int64))
                cols.append(np.nonzero(m)[0])
                vals.append(np.full(int(m.sum()), ph, dtype=complex))
        M = coo_matrix(
            (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
            shape=(self.dim, self.dim)).tocsr()
        return M

    def H_xxz(self, Jpm, phi=(0., 0., 0.), gauge="minimg"):
        """Twisted XXZ Hamiltonian. gauge='minimg': theta = n_ij . phi;
        gauge='smooth': theta = (phi/Lfrac) . d_ij with d_ij the min-image
        real bond displacement (uniform vector potential)."""
        cl = self.cl
        phi = np.asarray(phi, float)
        if gauge == "minimg":
            th = cl.bond_wrap @ phi
        elif gauge == "smooth":
            # uniform vector potential A with L_k . A = phi_k, so a closed
            # path of total wrap n picks up the same holonomy e^{-i n.phi}
            # as the minimum-image gauge (rows of Lvecs are the L_k)
            A = np.linalg.solve(cl.Lvecs, phi)
            disp = (cl.positions[cl.bonds[:, 1]] - cl.positions[cl.bonds[:, 0]]
                    - cl.bond_wrap @ cl.Lvecs)
            th = -(disp @ A)
        else:
            raise ValueError(gauge)
        return (-Jpm) * self.hop_matrix(th) + diags(self.Ez)

    # ------------------------------------------------------------------
    def ring_sum(self, loops):
        """sum_C (S+ S- S+ S- ... + h.c.) over the given loop paths."""
        rows, cols, vals = [], [], []
        one = np.uint64(1)
        for path, _w in loops:
            L = len(path)
            raise_bits = np.uint64(0)
            lower_bits = np.uint64(0)
            flip = np.uint64(0)
            for k, site in enumerate(path):
                b = one << np.uint64(site)
                flip |= b
                if k % 2 == 0:
                    raise_bits |= b
                else:
                    lower_bits |= b
            for rb, lb in ((raise_bits, lower_bits), (lower_bits, raise_bits)):
                m = ((self.states & rb) == 0) & ((self.states & lb) == lb)
                new = self.states[m] ^ flip
                rows.append(np.array([self.index[int(x)] for x in new],
                                     dtype=np.int64))
                cols.append(np.nonzero(m)[0])
                vals.append(np.ones(int(m.sum())))
        M = coo_matrix(
            (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
            shape=(self.dim, self.dim)).tocsr()
        return M

    # ------------------------------------------------------------------
    def low_spectrum(self, H, k=110):
        E = eigsh(H, k=k, which="SA", return_eigenvectors=False)
        return np.sort(E)

    def transport_mask(self):
        """Boolean (n_ice, n_ice) mask implementing the zero-net-transport
        projector at the operator level: element (a, b) between ice
        configurations is kept iff the dipole of the flipped-site set,
        rho = sum_raised r - sum_lowered r, is an integer combination of the
        cluster translations (rho = 0 mod L). Winding four-loop and wrapping
        hexagon flips carry fractional rho and are killed; contractible
        hexagons and diagonal elements carry rho = 0 and are kept. Apply as
        H_proj = mask * H_ice with H_ice a downfolded operator written in
        the ice-configuration basis (see downfold)."""
        cl = self.cl
        ice = cl.ice_states
        n = len(ice)
        pos4 = np.round(cl.positions * 4).astype(np.int64)
        L4 = np.round(cl.Lvecs * 4).astype(np.int64)
        mask = np.zeros((n, n), dtype=bool)
        for a in range(n):
            for b in range(a, n):
                raised = int(ice[b]) & ~int(ice[a])
                lowered = int(ice[a]) & ~int(ice[b])
                r4 = np.zeros(3, np.int64)
                i, rr, ll = 0, raised, lowered
                while rr or ll:
                    if rr & 1:
                        r4 += pos4[i]
                    if ll & 1:
                        r4 -= pos4[i]
                    rr >>= 1
                    ll >>= 1
                    i += 1
                # rho = 0 mod L  <=>  L4^{-1} r4 integer
                sol = np.linalg.solve(L4.T.astype(float), r4.astype(float))
                ok = np.all(np.abs(sol - np.round(sol)) < 1e-9)
                mask[a, b] = mask[b, a] = ok
        return mask

    def downfold(self, H, n_band=None):
        """Exact des Cloizeaux effective Hamiltonian of the lowest ice band.

        Takes the lowest n_band (= n_ice) eigenpairs, projects the
        eigenvectors onto the ice-basis rows, Loewdin-orthonormalizes, and
        returns  H_eff = T diag(E) T^dagger  in the ice basis (n_ice x n_ice,
        complex, Hermitian), plus the eigenvalues used.
        """
        n_band = n_band or len(self.ice_rows)
        E, Psi = eigsh(H, k=n_band, which="SA")
        order = np.argsort(E)
        E, Psi = E[order], Psi[:, order]
        # ARPACK can return imperfectly orthonormal vectors inside degenerate
        # multiplets; re-orthonormalize within each energy cluster (QR mixes
        # only states of equal energy, so the (E_i, psi_i) pairing survives)
        edges = np.flatnonzero(np.diff(E) > 1e-9) + 1
        for lo, hi in zip(np.r_[0, edges], np.r_[edges, len(E)]):
            if hi - lo > 1:
                Psi[:, lo:hi] = np.linalg.qr(Psi[:, lo:hi])[0]
        gram_err = np.max(np.abs(Psi.conj().T @ Psi - np.eye(n_band)))
        if gram_err > 1e-8:
            print(f"  [downfold] warning: band Gram error {gram_err:.2e}")
        A = Psi[self.ice_rows, :]                # (n_ice, n_band) overlap
        # Loewdin: T = A (A^dag A)^(-1/2)  -- columns = orthonormalized images
        U, s, Vh = np.linalg.svd(A, full_matrices=False)
        if s.min() < 1e-6:
            print(f"  [downfold] warning: smallest overlap sv = {s.min():.2e}")
        T = U @ Vh                               # unitary part of A
        return (T * E) @ T.conj().T, E
