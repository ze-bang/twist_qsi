#!/usr/bin/env python3
"""High-order degenerate perturbation theory on the ice manifold (cubic-16).

Independent convergence check of the winding-free suppression curve.  The
wave operator is built by the standard Bloch recursion on the fixed-Sz
space,

    Omega_0 = P,
    E_k     = P V Omega_{k-1},
    Omega_k = R [ V Omega_{k-1} - sum_{m=1}^{k-1} Omega_{k-m} E_m ],

with R the reduced resolvent of the Ising part at the ice energy.  The
order-k blocks are coupling-independent, so one recursion serves every Jpm.
The effective ice-sector Hamiltonian at order n is then formed EXACTLY the
way the production protocol forms it from the exact band: Loewdin
orthonormalization of the (truncated) wave operator followed by the
polarization-block mask,

    H_n(Jpm) = G^{-1/2} Omega^dag H Omega G^{-1/2},   G = Omega^dag Omega,
    h_n      = sector_project(H_n),

so PT order n and the exact winding-free construction differ only in the
subspace fed to the same pullback.  The band-only heat-capacity peak of h_n
is compared with that of the exact winding-free band from the scan cache at
every available coupling.

Diagnostics verified on the fly: E_1 = 0, E_2 proportional to the identity
(uniform second-order shift), and the third-order off-diagonal amplitude
equal to the Hermele-Fisher-Balents hexagon coefficient 12 (in units of
Jpm^3/Jzz^2).  A small minimum eigenvalue of G flags the breakdown of the
perturbative subspace.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "notes"))

import recompute_finite_size_artifact as geometry  # noqa: E402
from qsi_campaign.exact_band import (  # noqa: E402
    fixed_magnetization_basis,
    microscopic_character_hamiltonian,
)
from qsi_campaign.protocol import (  # noqa: E402
    polarization_sector_labels,
    sector_project,
)

CACHE = ROOT / "campaign" / "cache" / "ce2hf2o7_points"
OUTPUT = ROOT / "campaign" / "outputs"
MAX_ORDER = 8


def band_peak(levels: np.ndarray, temperatures: np.ndarray) -> float:
    """Position of the heat-capacity maximum of a finite level set."""
    shifted = levels - levels.min()
    beta = 1.0 / temperatures[:, None]
    weights = np.exp(-beta * shifted[None, :])
    z = weights.sum(axis=1)
    mean = (weights * shifted[None, :]).sum(axis=1) / z
    second = (weights * shifted[None, :] ** 2).sum(axis=1) / z
    heat = (second - mean**2) / temperatures**2
    return float(temperatures[int(np.argmax(heat))])


def main() -> None:
    cluster = geometry.build_cluster("cubic", (1, 1, 1))
    states = fixed_magnetization_basis(cluster.n_sites, cluster.n_sites // 2)
    h0 = microscopic_character_hamiltonian(cluster, states, 0.0, np.zeros(3)).tocsr()
    v = (microscopic_character_hamiltonian(cluster, states, 1.0, np.zeros(3)) - h0).tocsr()
    h0 = h0.real
    v = v.real

    index = {int(s): i for i, s in enumerate(states)}
    ice = np.asarray([index[int(s)] for s in cluster.ice_states], dtype=np.int64)
    d_ice = len(ice)
    dim = len(states)
    labels = polarization_sector_labels(cluster)

    diag = np.asarray(h0.diagonal(), dtype=float)
    if np.max(np.abs(diag[ice])) > 1e-12:
        raise RuntimeError("ice states are not at zero Ising energy")
    resolvent = np.zeros(dim)
    excited = np.abs(diag) > 1e-9
    resolvent[excited] = -1.0 / diag[excited]
    if int((~excited).sum()) != d_ice:
        raise RuntimeError("zero-energy states beyond the ice manifold")

    # Bloch recursion; Omega_k are dense (dim x d_ice) blocks.
    omega = [np.zeros((dim, d_ice))]
    omega[0][ice, np.arange(d_ice)] = 1.0
    e_coef: list[np.ndarray | None] = [None]
    for order in range(1, MAX_ORDER + 1):
        w = v @ omega[order - 1]
        e_coef.append(w[ice, :].copy())
        for m in range(1, order):
            w -= omega[order - m] @ e_coef[m]
        omega.append(resolvent[:, None] * w)

    # The raw coefficients contain the torus winding channels (the paper's
    # cluster artifact): the second-order off-diagonal amplitude is the
    # winding four-cycle g4 = 4 Jpm^2/Jzz of Eq. (3), and higher raw orders
    # carry further winding processes.  The polarization mask removes them,
    # so the physical checks act on the masked coefficients: masked order 2
    # must be a uniform shift, and the masked order-3 off-diagonal amplitude
    # must equal the Hermele-Fisher-Balents hexagon coefficient 12.
    masked2 = sector_project(e_coef[2], labels)
    masked3 = sector_project(e_coef[3], labels)
    checks = {
        "order1_norm": float(np.max(np.abs(e_coef[1]))),
        "raw_order2_winding_g4": float(np.max(np.abs(
            e_coef[2] - np.diag(np.diag(e_coef[2]))))),
        "masked_order2_uniformity": float(np.std(np.diag(masked2))),
        "masked_order2_offdiag": float(np.max(np.abs(
            masked2 - np.diag(np.diag(masked2))))),
        "masked_order3_hexagon": float(np.max(np.abs(
            masked3 - np.diag(np.diag(masked3))))),
    }
    print("PT diagnostics:", json.dumps(checks), flush=True)
    if checks["order1_norm"] > 1e-10 or checks["masked_order2_offdiag"] > 1e-10:
        raise RuntimeError("low-order structure violates ice-manifold PT")
    if abs(checks["raw_order2_winding_g4"] - 4.0) > 1e-9:
        raise RuntimeError("winding four-cycle amplitude does not match g4 = 4")
    if abs(checks["masked_order3_hexagon"] - 12.0) > 1e-9:
        raise RuntimeError("hexagon amplitude does not match the third-order value 12")

    # Masked, symmetrized strict-series coefficients: the order-n effective
    # ring Hamiltonian is sum_k lambda^k E_k.  Saved for the figure pipeline,
    # which draws the peak of the truncated series as a perturbative guide.
    coefficient_blocks = {}
    for order in range(2, MAX_ORDER + 1):
        masked = sector_project(e_coef[order], labels)
        coefficient_blocks[f"E{order}"] = np.asarray(
            0.5 * (masked + masked.T.conj()), dtype=float)
    np.savez_compressed(OUTPUT / "ice_pt_coefficients.npz", **coefficient_blocks)
    print(f"wrote {OUTPUT / 'ice_pt_coefficients.npz'}")

    temperatures = np.geomspace(1e-5, 1.0, 3000)
    points = []
    for path in sorted(CACHE.glob("cubic16_xyz_jpm*_pp+0p000000.npz")):
        blob = np.load(path)
        if not bool(blob["well_defined"]):
            continue
        jpm = float(blob["jpm"])
        g6 = 12.0 * abs(jpm) ** 3
        exact_peak = band_peak(np.asarray(blob["winding_free_band"], float), temperatures)
        record = {"jpm": jpm, "g6": g6, "exact_peak_over_g6": exact_peak / g6}
        for order in (3, 4, 5, 6, 7, 8):
            om = sum(jpm**k * omega[k] for k in range(order + 1))
            gram = om.T @ om
            gram_eigs, gram_vecs = np.linalg.eigh(gram)
            record[f"gram_min_order{order}"] = float(gram_eigs.min())
            if gram_eigs.min() < 1e-8:
                record[f"order{order}"] = None
                continue
            inv_sqrt = gram_vecs @ np.diag(gram_eigs**-0.5) @ gram_vecs.T
            hmat = om.T @ (h0 @ om) + jpm * (om.T @ (v @ om))
            effective = inv_sqrt @ hmat @ inv_sqrt
            masked = sector_project(effective, labels)
            masked = 0.5 * (masked + masked.T.conj())
            peak = band_peak(np.linalg.eigvalsh(masked), temperatures)
            record[f"order{order}"] = peak / g6
        points.append(record)
        shown = {k: (round(v, 4) if isinstance(v, float) else v)
                 for k, v in record.items() if not k.startswith("gram")}
        print(shown, flush=True)

    payload = {
        "construction": "Bloch wave operator + Loewdin pullback + polarization mask",
        "max_order": MAX_ORDER,
        "diagnostics": checks,
        "points": sorted(points, key=lambda p: p["jpm"]),
    }
    out = OUTPUT / "ice_pt_convergence.json"
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {out} with {len(points)} couplings")


if __name__ == "__main__":
    main()
