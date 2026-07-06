"""
Twist-aware Hamiltonian builder for the 16-site pyrochlore (1x1x1) cluster.

Reuses generate_pyrochlore_super_cluster() from the existing edlib helper for
the geometry, then writes Trans.dat / InterAll.dat with U(1) twist phases on
NN bonds that cross the periodic-boundary cell.

Convention
----------
Lattice vectors are the cubic conventional unit cell (a, a, a) with a = 1.
For each NN bond (i, j) we compute the integer wrap vector

        n_ij  =  round( (r_j - r_i) / L )       (componentwise)

where L = (dim1, dim2, dim3). Wrap vector is zero for bulk bonds and
non-zero for any bond that crosses a periodic boundary.

A twist vector phi = (phi_x, phi_y, phi_z) in [0, 2*pi)^3 is applied as

        S^+_i S^-_j  ->  S^+_i S^-_j  *  exp(-i * n_ij . phi)
        S^-_i S^+_j  ->  S^-_i S^+_j  *  exp(+i * n_ij . phi)

S^z S^z and the on-site (Trans.dat) terms are unaffected by the twist (they
have no winding). S^+ S^+ and S^- S^- (J_pmpm) terms are not generated here
because we restrict to the non-Kramers Jpmpm = 0 case for the demo (Jxx = Jyy
=> Jpmpm = 0 by construction).

Reuses the existing edlib I/O writers so the produced files are byte-compatible
with the C++ ED binary's loadFromInterAllFile / loadFromFile parsers.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

ED_HELPER_PATH = Path(__file__).resolve().parents[2] / "QED" / "python" / "edlib"
sys.path.insert(0, str(ED_HELPER_PATH))
from helper_pyrochlore_super import (  # noqa: E402
    generate_pyrochlore_super_cluster,
    create_nn_lists,
    write_cluster_nn_list,
    write_interALL,
    write_transfer,
    write_one_body_correlations,
    write_two_body_correlations,
)


def compute_bond_wrap_vectors(
    edges: Sequence[Tuple[int, int]],
    vertices: Dict[int, Tuple[float, float, float]],
    dims: Tuple[int, int, int],
) -> Dict[Tuple[int, int], Tuple[int, int, int]]:
    """For each bond (v1, v2) in the canonical PBC NN list, return the integer
    wrap vector n such that r_{v2} - r_{v1} - n*L is the minimum-image
    displacement. Bulk bonds get n = (0, 0, 0)."""
    L = np.array(dims, dtype=float)
    bond_wrap: Dict[Tuple[int, int], Tuple[int, int, int]] = {}
    for v1, v2 in edges:
        dr = np.asarray(vertices[v2]) - np.asarray(vertices[v1])
        n = np.round(dr / L).astype(int)
        bond_wrap[(v1, v2)] = (int(n[0]), int(n[1]), int(n[2]))
    return bond_wrap


def write_pyrochlore_xxz_with_twist(
    output_dir: str,
    Jxx: float,
    Jyy: float,
    Jzz: float,
    twist: Tuple[float, float, float],
    dim1: int = 1,
    dim2: int = 1,
    dim3: int = 1,
    h_field: float = 0.0,
    field_dir: Sequence[float] = (1.0, 1.0, 1.0),
    h_perp: float = 0.0,
):
    """Generate the full set of input files (Trans.dat, InterAll.dat,
    positions.dat, *_nn_list.dat, two-body correlation maps) for the
    16-site pyrochlore cluster at twist `twist` = (phi_x, phi_y, phi_z).

    Restricts to the non-Kramers Jpmpm = 0 case (Jxx = Jyy required when
    Jpmpm sector is enabled in the parent helper). Here we set the Jpmpm
    coefficients to zero unconditionally.

    `h_perp` adds a uniform LOCAL transverse field h_perp * S^x_i on every
    site (same local x-axis convention at every sublattice, i.e.
    h_perp/2 * (S+_i + S-_i)). Unlike `h_field` (a local-S^z / global-[111]
    longitudinal term, which commutes with S^z_tot), this breaks S^z_tot
    conservation -- the resulting Hamiltonian must be diagonalized in the
    full Hilbert space, not a fixed-Sz block.
    """
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    pbc = True
    vertices, edges, tetrahedra, node_mapping, vertex_to_cell = generate_pyrochlore_super_cluster(
        dim1, dim2, dim3, use_pbc=pbc
    )
    nn_list, positions, sublattice_indices = create_nn_lists(edges, node_mapping, vertices, vertex_to_cell)
    cluster_name = f"pyrochlore_super_{dim1}x{dim2}x{dim3}_pbc"
    write_cluster_nn_list(output_dir, cluster_name, nn_list, positions, sublattice_indices, node_mapping, vertex_to_cell)

    bond_wrap = compute_bond_wrap_vectors(edges, vertices, (dim1, dim2, dim3))

    Jpm = -(Jxx + Jyy) / 4.0
    phi = np.asarray(twist, dtype=float)

    field_dir_arr = np.asarray(field_dir, dtype=float)
    nrm = np.linalg.norm(field_dir_arr)
    if nrm > 1e-15:
        field_dir_arr = field_dir_arr / nrm
    z_local = np.array([
        np.array([1, 1, 1]) / np.sqrt(3),
        np.array([1, -1, -1]) / np.sqrt(3),
        np.array([-1, 1, -1]) / np.sqrt(3),
        np.array([-1, -1, 1]) / np.sqrt(3),
    ])

    transfer = []
    for site_id in sorted(nn_list.keys()):
        sub_idx = sublattice_indices[site_id]
        local_field_z = h_field * float(np.dot(field_dir_arr, z_local[sub_idx]))
        if abs(local_field_z) > 1e-15:
            transfer.append([2, node_mapping[site_id], -local_field_z, 0.0])
        if abs(h_perp) > 1e-15:
            # -h_perp * S^x_i = -h_perp/2 * (S+_i + S-_i), local frame, same
            # convention on every sublattice (op codes: 0=S+, 1=S-, 2=Sz)
            transfer.append([0, node_mapping[site_id], -h_perp / 2.0, 0.0])
            transfer.append([1, node_mapping[site_id], -h_perp / 2.0, 0.0])

    interALL: List[List[float]] = []
    for v1, v2 in sorted(edges):
        i, j = node_mapping[v1], node_mapping[v2]
        nij = np.array(bond_wrap[(v1, v2)], dtype=float)
        phase_dot = float(np.dot(nij, phi))
        cphase = np.cos(phase_dot)
        sphase = np.sin(phase_dot)

        interALL.append([2, i, 2, j, Jzz, 0.0])

        re_pm = -Jpm * cphase
        im_pm = -Jpm * (-sphase)
        interALL.append([0, i, 1, j, re_pm, im_pm])

        re_mp = -Jpm * cphase
        im_mp = -Jpm * (+sphase)
        interALL.append([1, i, 0, j, re_mp, im_mp])

    interALL_arr = np.array(interALL) if interALL else np.empty((0, 6))
    transfer_arr = np.array(transfer) if transfer else np.empty((0, 4))

    write_interALL(output_dir, interALL_arr, "InterAll.dat")
    write_transfer(output_dir, transfer_arr, "Trans.dat")

    np.savetxt(os.path.join(output_dir, "field_strength.dat"), np.array([[h_field]]))

    n_sites = len(nn_list)
    op_names = ["S+", "S-", "Sz"]
    for a in range(3):
        write_one_body_correlations(output_dir, a, n_sites, f"one_body_correlations{op_names[a]}.dat")
        for b in range(3):
            write_two_body_correlations(output_dir, a, b, n_sites, f"two_body_correlations{op_names[a]}{op_names[b]}.dat")

    return {
        "n_sites": n_sites,
        "n_bonds": len(edges),
        "n_tetrahedra": len(tetrahedra),
        "wrap_summary": _summarise_wraps(bond_wrap),
        "twist": tuple(float(x) for x in phi),
    }


def _summarise_wraps(bond_wrap: Dict[Tuple[int, int], Tuple[int, int, int]]):
    """Count bonds by wrap vector for diagnostics."""
    counter: Dict[Tuple[int, int, int], int] = {}
    for n in bond_wrap.values():
        counter[n] = counter.get(n, 0) + 1
    return counter


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True, help="output directory")
    p.add_argument("--Jxx", type=float, default=0.2)
    p.add_argument("--Jyy", type=float, default=0.2)
    p.add_argument("--Jzz", type=float, default=1.0)
    p.add_argument("--phi", type=float, nargs=3, default=[0.0, 0.0, 0.0])
    args = p.parse_args()
    info = write_pyrochlore_xxz_with_twist(
        args.out, args.Jxx, args.Jyy, args.Jzz, tuple(args.phi)
    )
    print("Wrote 16-site pyrochlore Hamiltonian with twist", info["twist"])
    print(f"  n_sites = {info['n_sites']}")
    print(f"  n_bonds = {info['n_bonds']}")
    print(f"  wrap summary (n_x, n_y, n_z) -> count:")
    for n, c in sorted(info["wrap_summary"].items()):
        print(f"    {n}: {c}")
