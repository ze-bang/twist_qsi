"""Check that the Loewdin corner map factorizes through the frame overlap."""
import sys
sys.path.insert(0, 'src')
sys.path.insert(0, 'notes')
import numpy as np
from scipy.linalg import eigh, polar
import recompute_finite_size_artifact as geometry
from qsi_campaign.exact_band import (
    basis_character_phases, fixed_magnetization_basis,
    microscopic_character_hamiltonian, translation_sector_bases,
    cubic16_translation_permutations, uniform_character_grid)

cl = geometry.build_cluster('cubic', (1, 1, 1))
st = fixed_magnetization_basis(16, 8)
idx = {int(s): i for i, s in enumerate(st)}
sec = translation_sector_bases(st, cubic16_translation_permutations(cl))
H = microscopic_character_hamiltonian(cl, st, -0.05, np.zeros(3))
vals, vecs = [], []
for b in sec.values():
    blk = (b.conj().T @ (H @ b)).toarray()
    blk = np.ascontiguousarray((0.5 * (blk + blk.conj().T)).real)
    v, w = eigh(blk, subset_by_index=(0, 89))
    vals.append(v)
    vecs.append(np.asarray(b @ w))
vals = np.concatenate(vals)
vecs = np.concatenate(vecs, axis=1)
o = np.argsort(vals)[:90]
Psi = vecs[:, o]
print('Psi isometry defect:', np.linalg.norm(Psi.conj().T @ Psi - np.eye(90)),
      flush=True)
h0 = Psi.conj().T @ (H @ Psi)
thetas = list(uniform_character_grid(2))
corners = [basis_character_phases(cl, st, np.asarray(t)) for t in thetas]
sourced = [microscopic_character_hamiltonian(cl, st, -0.05, np.asarray(t))
           for t in thetas]
wp = wh = wu = 0.0
for ph, Hc in zip(corners, sourced):
    Ac = Psi.conj().T @ (ph[:, None] * Psi)
    Vl = polar(Ac, side='left')[0]                 # Ac = |Ac^dag| Vl
    K = Ac @ Ac.conj().T
    w, u = np.linalg.eigh(0.5 * (K + K.conj().T))
    Kinv_half = (u / np.sqrt(w)) @ u.conj().T
    wp = max(wp, np.linalg.norm(Ac.conj().T @ Kinv_half - Vl.conj().T))
    wu = max(wu, np.linalg.norm(Vl.conj().T @ Vl - np.eye(90)))
    Phi = ph[:, None] * Psi                        # the corner band U_c Psi
    hc = Vl @ (Phi.conj().T @ (Hc @ Phi)) @ Vl.conj().T
    wh = max(wh, np.linalg.norm(hc - Vl @ h0 @ Vl.conj().T))
    wt = np.linalg.norm(Phi.conj().T @ (Hc @ Phi) - h0)
    print('   corner: ||Psi^dag U_c^dag H(theta_c) U_c Psi - h0|| =', wt,
          flush=True)
print('max || A_c^dag (A_cA_c^dag)^-1/2 - V_c^dag ||   =', wp)
print('max || V_c^dag V_c - 1 ||                        =', wu)
print('max || pulled-back corner op - V_c h0 V_c^dag || =', wh)

ice = np.zeros((len(st), 90))
for k, s in enumerate(cl.ice_states):
    ice[idx[int(s)], k] = 1.0
for n, ph in enumerate(corners[:4]):
    A = ice.T @ (ph[:, None] * ice)
    print(f'  ice frame corner {n}: |offdiag A_c| =',
          np.abs(A - np.diag(np.diag(A))).max(),
          ', unitarity defect =',
          np.linalg.norm(A @ A.conj().T - np.eye(90)))
