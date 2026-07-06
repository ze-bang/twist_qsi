"""
Pyrochlore-on-FCC-primitive cluster generator with twist-aware Hamiltonian
output.

The 4-sublattice pyrochlore lattice is FCC-Bravais. Per primitive cell it
contains 4 sites; the cluster shape is parameterised by integers (L1, L2, L3)
giving a parallelepiped of L1 a1 x L2 a2 x L3 a3 with primitive vectors

    a1 = (1, 1, 0) / 2 ,  a2 = (1, 0, 1) / 2 ,  a3 = (0, 1, 1) / 2

and N = 4 L1 L2 L3 sites. The (2,2,2) FCC cluster of N=32 sites has the full
pyrochlore O_h point group, in contrast to the (2,1,1) cubic 32-site brick
which has only C_2v. Twist phases on boundary-crossing NN bonds are written
in the same convention as twist_helper.py:

    S^+_i S^-_j  ->  S^+_i S^-_j  *  exp(-i * n_ij . phi)

with n_ij the FCC-primitive wrap vector (an integer vector in the (a1,a2,a3)
basis).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

ED_HELPER_PATH = Path(__file__).resolve().parents[2] / "QED" / "python" / "edlib"
sys.path.insert(0, str(ED_HELPER_PATH))
from helper_pyrochlore_super import (  # noqa: E402
    write_interALL,
    write_transfer,
    write_one_body_correlations,
    write_two_body_correlations,
)

A1 = np.array([1.0, 1.0, 0.0]) / 2.0
A2 = np.array([1.0, 0.0, 1.0]) / 2.0
A3 = np.array([0.0, 1.0, 1.0]) / 2.0
A_BASIS = np.column_stack([A1, A2, A3])
A_INV = np.linalg.inv(A_BASIS)
SUBLAT = np.array(
    [
        [0.0, 0.0, 0.0],
        [0.0, 0.25, 0.25],
        [0.25, 0.0, 0.25],
        [0.25, 0.25, 0.0],
    ]
)
NN_THRESH_LOW = 0.01
NN_THRESH_HIGH = 0.4


def generate_fcc_cluster(L1: int, L2: int, L3: int, strict: bool = True):
    """Return (vertices, edges, tetrahedra, bond_wrap) for a pyrochlore
    cluster on an L1 x L2 x L3 FCC-primitive parallelepiped with PBC.

    bond_wrap[(u, v)] is the integer wrap vector in (a1, a2, a3) units.

    The FCC primitive translations have length |a_k| = sqrt(2)/2, just twice
    the pyrochlore NN distance d_NN = sqrt(2)/4. As a consequence, any FCC
    primitive cluster with min(L1, L2, L3) < 2 is *geometrically degenerate*:
    some pairs of sites have multiple distinct minimum-image NN bonds and
    some sites end up with coordination < 6, so the cluster does not
    faithfully represent the bulk pyrochlore lattice. We refuse to build
    such clusters by default; set ``strict=False`` to override (e.g. for
    diagnostic experiments). The minimum non-degenerate FCC primitive
    cluster is (2, 2, 2) at N = 32 sites; for 16-site experiments use the
    cubic (1, 1, 1) cluster instead.
    """
    if strict and min(L1, L2, L3) < 2:
        raise ValueError(
            f"FCC primitive cluster ({L1},{L2},{L3}) is geometrically "
            "degenerate: at least one direction is shorter than 2|a_k| "
            "= 2 * NN spacing, so the cluster fundamental domain "
            "self-overlaps and the NN graph is not faithful to the bulk "
            "pyrochlore lattice. Use (L1, L2, L3) with min(L_k) >= 2, "
            "or pass strict=False to override."
        )
    Ls = np.array([L1, L2, L3], dtype=float)
    vertices: Dict[int, Tuple[float, float, float]] = {}
    vid = 0
    for i in range(L1):
        for j in range(L2):
            for k in range(L3):
                origin = i * A1 + j * A2 + k * A3
                for s in range(4):
                    pos = origin + SUBLAT[s]
                    vertices[vid] = tuple(pos)
                    vid += 1
    n_sites = vid

    edges: set = set()
    bond_wrap: Dict[Tuple[int, int], Tuple[int, int, int]] = {}
    for u in range(n_sites):
        ru = np.asarray(vertices[u])
        for v in range(u + 1, n_sites):
            rv = np.asarray(vertices[v])
            dr = rv - ru
            frac = A_INV @ dr
            n = np.round(frac / Ls).astype(int)
            min_image_frac = frac - n * Ls
            min_image_dr = A_BASIS @ min_image_frac
            dist = float(np.linalg.norm(min_image_dr))
            if NN_THRESH_LOW < dist < NN_THRESH_HIGH:
                edges.add((u, v))
                bond_wrap[(u, v)] = tuple(int(x) for x in n)
                bond_wrap[(v, u)] = tuple(-int(x) for x in n)

    # Tetrahedra are the 4-cliques of the NN graph.
    nbrs_set = {u: set() for u in vertices}
    for u, v in edges:
        nbrs_set[u].add(v)
        nbrs_set[v].add(u)
    tets: List[Tuple[int, int, int, int]] = []
    seen = set()
    for u in range(n_sites):
        nbrs_u = sorted(v for v in nbrs_set[u] if v > u)
        m = len(nbrs_u)
        for i in range(m):
            for j in range(i + 1, m):
                v, w = nbrs_u[i], nbrs_u[j]
                if w not in nbrs_set[v]:
                    continue
                for k in range(j + 1, m):
                    x = nbrs_u[k]
                    if x in nbrs_set[v] and x in nbrs_set[w]:
                        key = tuple(sorted((u, v, w, x)))
                        if key not in seen:
                            seen.add(key)
                            tets.append(key)

    return vertices, list(edges), tets, bond_wrap


def write_pyrochlore_xxz_with_twist_fcc(
    output_dir: str,
    Jxx: float,
    Jyy: float,
    Jzz: float,
    twist: Tuple[float, float, float],
    L1: int = 2,
    L2: int = 2,
    L3: int = 2,
    h_field: float = 0.0,
    field_dir: Sequence[float] = (1.0, 1.0, 1.0),
):
    """Write Trans.dat and InterAll.dat for a pyrochlore XXZ Hamiltonian on
    an L1 x L2 x L3 FCC-primitive cluster, with U(1) twist phases on
    boundary-crossing NN bonds.

    Convention matches twist_helper.py: Jzz, Jzz->S+S- + S-S+ couplings via
    Jpm = -(Jxx+Jyy)/4, Jpmpm = -(Jxx-Jyy)/4. The XXZ point Jxx=Jyy gives
    Jpmpm=0 so we only emit Jpm-type S+S- and S-S+ terms. Twist phases:

        S^+_i S^-_j  ->  S^+_i S^-_j  *  exp(-i * n_ij . phi)
        S^-_i S^+_j  ->  S^-_i S^+_j  *  exp(+i * n_ij . phi)
    """
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    vertices, edges, tets, bond_wrap = generate_fcc_cluster(L1, L2, L3)
    n_sites = len(vertices)

    Jpm = -(Jxx + Jyy) / 4.0
    Jpmpm = -(Jxx - Jyy) / 4.0
    if abs(Jpmpm) > 1e-12:
        raise NotImplementedError(
            "Jpmpm != 0 (Jxx != Jyy) is not currently emitted by this writer; "
            "the demo restricts to the XXZ point."
        )

    phi = np.asarray(twist, dtype=float)

    # Local easy axes for [111] (non-Kramers) field projection. The 4
    # sublattice indices are 0,1,2,3 mod 4 by construction of the generator.
    z_local = np.array(
        [
            np.array([1, 1, 1]) / np.sqrt(3),
            np.array([1, -1, -1]) / np.sqrt(3),
            np.array([-1, 1, -1]) / np.sqrt(3),
            np.array([-1, -1, 1]) / np.sqrt(3),
        ]
    )
    field_dir_arr = np.asarray(field_dir, dtype=float)
    nrm = np.linalg.norm(field_dir_arr)
    if nrm > 1e-15:
        field_dir_arr = field_dir_arr / nrm

    transfer: List[List[float]] = []
    for s in range(n_sites):
        sub_idx = s % 4
        h_loc = h_field * float(np.dot(field_dir_arr, z_local[sub_idx]))
        if abs(h_loc) > 1e-15:
            transfer.append([2, s, -h_loc, 0.0])

    interALL: List[List[float]] = []
    for (u, v) in sorted(edges):
        n = np.array(bond_wrap[(u, v)], dtype=float)
        phase_dot = float(np.dot(n, phi))
        cphase = np.cos(phase_dot)
        sphase = np.sin(phase_dot)
        # Jzz S^z S^z
        interALL.append([2, u, 2, v, Jzz, 0.0])
        # Jpm (S+_u S-_v + h.c.) with twist phase
        re_pm = -Jpm * cphase
        im_pm = -Jpm * (-sphase)
        interALL.append([0, u, 1, v, re_pm, im_pm])
        re_mp = -Jpm * cphase
        im_mp = -Jpm * (+sphase)
        interALL.append([1, u, 0, v, re_mp, im_mp])

    interALL_arr = np.array(interALL) if interALL else np.empty((0, 6))
    transfer_arr = np.array(transfer) if transfer else np.empty((0, 4))
    write_interALL(output_dir, interALL_arr, "InterAll.dat")
    write_transfer(output_dir, transfer_arr, "Trans.dat")

    np.savetxt(os.path.join(output_dir, "field_strength.dat"), np.array([[h_field]]))

    op_names = ["S+", "S-", "Sz"]
    for a in range(3):
        write_one_body_correlations(
            output_dir, a, n_sites, f"one_body_correlations{op_names[a]}.dat"
        )
        for b in range(3):
            write_two_body_correlations(
                output_dir,
                a,
                b,
                n_sites,
                f"two_body_correlations{op_names[a]}{op_names[b]}.dat",
            )

    return n_sites, len(edges), len(tets)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--output", default="./output_fcc")
    p.add_argument("--L1", type=int, default=2)
    p.add_argument("--L2", type=int, default=2)
    p.add_argument("--L3", type=int, default=2)
    p.add_argument("--Jxx", type=float, default=0.2)
    p.add_argument("--Jyy", type=float, default=0.2)
    p.add_argument("--Jzz", type=float, default=1.0)
    p.add_argument("--phi-x", type=float, default=0.0)
    p.add_argument("--phi-y", type=float, default=0.0)
    p.add_argument("--phi-z", type=float, default=0.0)
    args = p.parse_args()

    n_sites, n_bonds, n_tets = write_pyrochlore_xxz_with_twist_fcc(
        args.output,
        args.Jxx,
        args.Jyy,
        args.Jzz,
        (args.phi_x, args.phi_y, args.phi_z),
        args.L1,
        args.L2,
        args.L3,
    )
    print(
        f"FCC ({args.L1},{args.L2},{args.L3}) cluster: "
        f"{n_sites} sites, {n_bonds} NN bonds, {n_tets} tetrahedra"
    )
    print(f"output: {args.output}")
