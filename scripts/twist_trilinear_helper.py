"""
Twisted XXZ (+ optional Kadowaki trilinear) Hamiltonian on the pyrochlore
super-cluster, using the same U(1) Peierls phases as twist_helper.py on
nearest-neighbour XY hops.

Two-body (matches twist_helper / build_twisted_hamiltonian):
    S^+_i S^-_j  * exp(-i n_ij · φ)   with n_ij the integer wrap of the minimum
    image vector from site i to site j (same convention as twist_helper).

Three-body (Peierls ansatz for σ^z σ^+ σ^z terms):
    Each Kadowaki row with S^+ on site j is multiplied by
        exp(-i φ · (n_{j→i} + n_{j→k}))
    where i,k are the σ^z neighbours in the file order
    [Sz_i, S^+_j, Sz_k] (op codes 2,0,2). The Hermitian-conjugate S^- row is
    multiplied by the complex conjugate so the operator stays Hermitian.

When all directed wraps vanish (e.g. open cluster or bulk-only triangles) the
phase factor is unity and the trilinear term reduces to the bare Kadowaki form.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

ED_HELPER = Path(__file__).resolve().parents[2] / "QED" / "python" / "edlib"
sys.path.insert(0, str(ED_HELPER))

import helper_pyrochlore_super as helper  # noqa: E402
import twist_helper as th  # noqa: E402


def _normalise_j3(three_spin_coeff: float | Sequence[float]) -> Tuple[float, float, float]:
    if np.isscalar(three_spin_coeff):
        v = float(three_spin_coeff)
        return (v, v, v)
    seq = tuple(float(x) for x in three_spin_coeff)
    if len(seq) != 3:
        raise ValueError("three_spin_coeff must be scalar or length-3 tuple")
    return seq


def directed_wrap(
    va: int,
    vb: int,
    vertices: Dict[int, Tuple[float, float, float]],
    dims: Tuple[int, int, int],
) -> np.ndarray:
    """Integer wrap vector for r_{vb} - r_{va} (same rounding as twist_helper)."""
    L = np.asarray(dims, dtype=float)
    dr = np.asarray(vertices[vb]) - np.asarray(vertices[va])
    return np.round(dr / L).astype(int)


def write_pyrochlore_twisted_xxz_trilinear(
    output_dir: str,
    Jxx: float,
    Jyy: float,
    Jzz: float,
    twist: Tuple[float, float, float],
    three_spin_coeff: float | Sequence[float] = 0.0,
    dim1: int = 1,
    dim2: int = 1,
    dim3: int = 1,
    h_field: float = 0.0,
    field_dir: Sequence[float] = (1.0, 1.0, 1.0),
    apply_threebody_twist: bool = True,
) -> Dict[str, object]:
    """Write Trans.dat, InterAll.dat, optional ThreeBodyG.dat (+ correlation maps).

    Parameters mirror ``twist_helper.write_pyrochlore_xxz_with_twist`` with an
    extra scalar or length-3 tuple ``three_spin_coeff`` for
    (J_{3s,1}, J_{3s,2}, J_{3s,3}) in the PRB 105, 014439 convention.
    """
    output_dir = str(output_dir)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    pbc = True
    vertices, edges, tetrahedra, node_mapping, vertex_to_cell = (
        helper.generate_pyrochlore_super_cluster(dim1, dim2, dim3, use_pbc=pbc)
    )
    nn_list, positions, sublattice_indices = helper.create_nn_lists(
        edges, node_mapping, vertices, vertex_to_cell
    )
    cluster_name = f"pyrochlore_super_{dim1}x{dim2}x{dim3}_pbc"
    helper.write_cluster_nn_list(
        output_dir, cluster_name, nn_list, positions, sublattice_indices,
        node_mapping, vertex_to_cell,
    )

    dims = (dim1, dim2, dim3)
    bond_wrap = th.compute_bond_wrap_vectors(edges, vertices, dims)

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

    transfer: List[List[float]] = []
    for site_id in sorted(nn_list.keys()):
        sub_idx = sublattice_indices[site_id]
        local_field_z = h_field * float(np.dot(field_dir_arr, z_local[sub_idx]))
        if abs(local_field_z) > 1e-15:
            transfer.append([2, node_mapping[site_id], -local_field_z, 0.0])

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

    helper.write_interALL(output_dir, interALL_arr, "InterAll.dat")
    helper.write_transfer(output_dir, transfer_arr, "Trans.dat")
    np.savetxt(os.path.join(output_dir, "field_strength.dat"), np.array([[h_field]]))

    j3s = _normalise_j3(three_spin_coeff)
    if any(abs(j) > 1e-15 for j in j3s):
        raw_terms = helper.generate_three_spin_terms(
            nn_list, node_mapping, j3s, sublattice_indices,
            vertex_to_cell=vertex_to_cell,
        )
        twisted: List[List[float]] = []
        m2v = {int(node_mapping[v]): int(v) for v in node_mapping}
        for row in raw_terms:
            op1, s1, op2, s2, op3, s3, cr, ci = row
            if not apply_threebody_twist:
                twisted.append(row)
                continue
            # Identify the S^+ / S^- site (op 0 / 1)
            if op2 == 0:
                vj = m2v[int(s2)]
                vi = m2v[int(s1)]
                vk = m2v[int(s3)]
                nji = directed_wrap(vj, vi, vertices, dims)
                njk = directed_wrap(vj, vk, vertices, dims)
                phase = np.exp(-1j * float(np.dot(phi, nji + njk)))
                c = (cr + 1j * ci) * phase
                twisted.append([op1, s1, op2, s2, op3, s3, c.real, c.imag])
            elif op2 == 1:
                vj = m2v[int(s2)]
                vi = m2v[int(s1)]
                vk = m2v[int(s3)]
                nji = directed_wrap(vj, vi, vertices, dims)
                njk = directed_wrap(vj, vk, vertices, dims)
                phase = np.exp(-1j * float(np.dot(phi, nji + njk)))
                c = (cr + 1j * ci) * np.conj(phase)
                twisted.append([op1, s1, op2, s2, op3, s3, c.real, c.imag])
            else:
                twisted.append(row)

        helper.write_three_spin_terms(output_dir, twisted, "ThreeBodyG.dat")
    else:
        # Remove stale three-body file if present
        p = Path(output_dir) / "ThreeBodyG.dat"
        if p.exists():
            p.unlink()

    n_sites = len(nn_list)
    op_names = ["S+", "S-", "Sz"]
    for a in range(3):
        helper.write_one_body_correlations(
            output_dir, a, n_sites, f"one_body_correlations{op_names[a]}.dat"
        )
        for b in range(3):
            helper.write_two_body_correlations(
                output_dir, a, b, n_sites,
                f"two_body_correlations{op_names[a]}{op_names[b]}.dat",
            )

    return {
        "n_sites": n_sites,
        "n_bonds": len(edges),
        "n_tetrahedra": len(tetrahedra),
        "wrap_summary": th._summarise_wraps(bond_wrap),
        "twist": tuple(float(x) for x in phi),
        "Jpm": Jpm,
        "three_spin_coeffs": j3s,
    }
