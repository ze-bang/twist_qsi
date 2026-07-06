"""Build a twisted-boundary pyrochlore XYZ Hamiltonian for the C++ ED engine.

The bare XYZ Hamiltonian on the pyrochlore is

    H = J_zz sum_<ij> S^z_i S^z_j  -  J_pm sum_<ij> ( S^+_i S^-_j + h.c. )

where J_pm = -(J_xx + J_yy)/4. Periodic boundary conditions on a finite torus
introduce non-contractible loops in the nearest-neighbour graph; in the spin-ice
regime these short cycles generate spurious ring-exchange channels that
contaminate the photon Green's function and the low-T specific heat.

Twist (Aharonov-Bohm) averaging removes them exactly: every NN bond crossing the
boundary in direction alpha picks up a U(1) phase exp(i phi_alpha), so a loop
with winding (w_x, w_y, w_z) acquires phase exp(i w . phi). Averaging
observables over phi in [0, 2*pi)^3 projects onto w = 0, killing every
non-contractible cycle while leaving every contractible (bulk) cycle untouched.

This module:
  * Generates the pyrochlore lattice (reusing the existing helper).
  * Computes the integer wrap vector n_ij in {-1, 0, +1}^3 for every NN bond,
    via the minimum-image convention.
  * Writes InterAll.dat / Trans.dat / positions.dat / *_nn_list.dat in the
    legacy format expected by the C++ ED binary, with S^+_i S^-_j coefficient
    multiplied by exp(-i phi . n_ij) (and the conjugate on S^-_i S^+_j).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

ED_ROOT = Path(__file__).resolve().parents[2] / "QED"
sys.path.insert(0, str(ED_ROOT / "python" / "edlib"))

import helper_pyrochlore_super as helper  # noqa: E402


def _wrap_vector(pos_i: np.ndarray, pos_j: np.ndarray,
                 box: np.ndarray) -> Tuple[int, int, int]:
    """Return n in Z^3 such that pos_j - pos_i - n*box is the minimum image."""
    delta = pos_j - pos_i
    n = np.zeros(3, dtype=int)
    for k in range(3):
        if delta[k] > box[k] / 2.0:
            n[k] = 1
        elif delta[k] < -box[k] / 2.0:
            n[k] = -1
    return tuple(int(x) for x in n)


def build_twisted_hamiltonian(
    output_dir: str | Path,
    dim: Tuple[int, int, int] = (1, 1, 1),
    Jxx: float = 0.2,
    Jyy: float = 0.2,
    Jzz: float = 1.0,
    twist_phi: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    non_kramer: bool = False,
) -> Dict[str, object]:
    """Generate Hamiltonian files for the twisted pyrochlore XYZ model.

    Args:
        output_dir: Destination directory; created if missing.
        dim: (dim1, dim2, dim3) cluster shape in conventional unit cells.
            (1, 1, 1) yields 16 sites; (2, 1, 1) yields 32.
        Jxx, Jyy, Jzz: XYZ exchange couplings on every NN bond.
        twist_phi: (phi_x, phi_y, phi_z) twist angles in radians.
        non_kramer: include the non-Kramers J_pmpm channel (default False;
            keep False unless you are studying the Kadowaki phase tables).

    Returns:
        Dict with the metadata needed by the runner (cluster name, num_sites,
        twist, edge wrap vectors, ...).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dim1, dim2, dim3 = dim
    use_pbc = True

    vertices, edges, tetrahedra, node_mapping, vertex_to_cell = (
        helper.generate_pyrochlore_super_cluster(dim1, dim2, dim3, use_pbc=use_pbc)
    )
    nn_list, positions, sublattice_indices = helper.create_nn_lists(
        edges, node_mapping, vertices, vertex_to_cell
    )

    pbc_str = "pbc" if use_pbc else "obc"
    nk_str = "non_kramer" if non_kramer else "kramer"
    cluster_name = f"pyrochlore_super_{dim1}x{dim2}x{dim3}_{pbc_str}_{nk_str}"

    helper.write_cluster_nn_list(
        str(output_dir), cluster_name, nn_list, positions, sublattice_indices,
        node_mapping, vertex_to_cell,
    )

    box = np.array([dim1, dim2, dim3], dtype=float)
    edge_wrap: Dict[Tuple[int, int], Tuple[int, int, int]] = {}
    for v1, v2 in edges:
        n_ij = _wrap_vector(np.array(vertices[v1]), np.array(vertices[v2]), box)
        edge_wrap[(v1, v2)] = n_ij

    Jpm = -(Jxx + Jyy) / 4.0
    Jpmpm = (Jxx - Jyy) / 4.0

    phi = np.array(twist_phi, dtype=float)

    interALL: List[List[float]] = []
    transfer: List[List[float]] = []

    gamma = np.exp(1j * 2.0 * np.pi / 3.0)
    nk_factor = np.array([
        [0,        1,        gamma,    gamma**2],
        [1,        0,        gamma**2, gamma   ],
        [gamma,    gamma**2, 0,        1       ],
        [gamma**2, gamma,    1,        0       ],
    ])

    for site_id in sorted(nn_list.keys()):
        i = node_mapping[site_id]
        for neighbor_id in nn_list[site_id]:
            if site_id >= neighbor_id:
                continue
            j = node_mapping[neighbor_id]
            n_ij = edge_wrap[(site_id, neighbor_id)]
            phase_arg = -float(np.dot(phi, np.array(n_ij, dtype=float)))
            phase = np.exp(1j * phase_arg)

            interALL.append([2, i, 2, j, Jzz, 0.0])

            sp_sm = -Jpm * phase
            interALL.append([0, i, 1, j, float(np.real(sp_sm)),
                             float(np.imag(sp_sm))])

            sm_sp = -Jpm * np.conj(phase)
            interALL.append([1, i, 0, j, float(np.real(sm_sp)),
                             float(np.imag(sm_sp))])

            if non_kramer:
                Jpmpm_ = Jpmpm * nk_factor[
                    sublattice_indices[i], sublattice_indices[j]
                ]
            else:
                Jpmpm_ = Jpmpm

            sm_sm = Jpmpm_ * phase
            interALL.append([1, i, 1, j, float(np.real(sm_sm)),
                             float(np.imag(sm_sm))])
            sp_sp = np.conj(Jpmpm_) * np.conj(phase)
            interALL.append([0, i, 0, j, float(np.real(sp_sp)),
                             float(np.imag(sp_sp))])

    interALL_arr = np.array(interALL) if interALL else np.empty((0, 6))
    transfer_arr = np.array(transfer) if transfer else np.empty((0, 4))

    helper.write_interALL(str(output_dir), interALL_arr, "InterAll.dat")
    helper.write_transfer(str(output_dir), transfer_arr, "Trans.dat")

    np.savetxt(str(output_dir / "field_strength.dat"), np.array([[0.0]]))

    max_site = len(nn_list)
    opname = ['S+', 'S-', 'Sz']
    for a in range(3):
        helper.write_one_body_correlations(
            str(output_dir), a, max_site,
            f"one_body_correlations{opname[a]}.dat",
        )
        for b in range(3):
            helper.write_two_body_correlations(
                str(output_dir), a, b, max_site,
                f"two_body_correlations{opname[a]}{opname[b]}.dat",
            )

    n_wrap_bonds = sum(1 for n in edge_wrap.values() if n != (0, 0, 0))

    return {
        "cluster_name": cluster_name,
        "num_sites": len(vertices),
        "num_bonds": len(edges),
        "num_wrap_bonds": n_wrap_bonds,
        "twist_phi": tuple(float(x) for x in twist_phi),
        "Jxx": Jxx, "Jyy": Jyy, "Jzz": Jzz,
        "Jpm": Jpm, "Jpmpm": Jpmpm,
        "dim": tuple(dim),
        "edge_wrap": edge_wrap,
    }


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("output_dir")
    p.add_argument("--Jxx", type=float, default=0.2)
    p.add_argument("--Jyy", type=float, default=0.2)
    p.add_argument("--Jzz", type=float, default=1.0)
    p.add_argument("--phi_x", type=float, default=0.0)
    p.add_argument("--phi_y", type=float, default=0.0)
    p.add_argument("--phi_z", type=float, default=0.0)
    p.add_argument("--dim", nargs=3, type=int, default=[1, 1, 1])
    args = p.parse_args()

    info = build_twisted_hamiltonian(
        args.output_dir,
        dim=tuple(args.dim),
        Jxx=args.Jxx, Jyy=args.Jyy, Jzz=args.Jzz,
        twist_phi=(args.phi_x, args.phi_y, args.phi_z),
    )
    print(f"Built {info['cluster_name']}: {info['num_sites']} sites, "
          f"{info['num_bonds']} bonds, {info['num_wrap_bonds']} wrap bonds, "
          f"twist=({args.phi_x:.4f}, {args.phi_y:.4f}, {args.phi_z:.4f})")
