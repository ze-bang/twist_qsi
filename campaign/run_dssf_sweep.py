#!/usr/bin/env python3
"""DSSF at Gamma and X versus coupling: S^zz (gauge sector) and S^xx (spinons).

Two channels, two constructions, one file per coupling:

  S^zz  the frame-free ice-block route of ``run_dssf_counterterm.py``: the
        90x90 fold F(z_0) at the cached self-consistent masked anchor, with
        the mask off (``naive``: the unmasked downfolding at the anchor, NOT
        the exact periodic band) and on (``winding-free``).  Same anchor for
        both, so the panels differ by exactly the deleted matrix elements.
        Band-to-band weight only; ground-multiplet averaged; elastic dropped.

  S^xx  exact, no downfolding: S^x = (S^+ + S^-)/2 leaves the ice manifold
        entirely (a single flip changes S_z by one and makes two spinons), so
        the transverse response is spinon-pair creation.  Computed by Lanczos
        continued fraction in the S_z=+1 sector (full reorthogonalization);
        the S^- piece equals the S^+ piece by the global spin-flip symmetry
        of XXZ at zero field, so weight = 2 * (1/4)|<n|S^+_q|0>|^2.  The two
        variants differ only through the reference ground state: raw H versus
        H + C with C = -(1-Delta)F(z_0) (final states are H eigenstates in
        S_z=+1 either way -- C lives in the S_z=0 ice block).  The raw
        variant therefore measures excitation energies from the
        artifact-deepened ground state and inflates the apparent spinon gap.

Conventions follow run_dssf_counterterm.py: momenta in units of 2*pi, phases
exp(-2i pi q.r), per-sublattice sources summed over the four sublattices, per
site.  Anchors are read from ``cache/bw_greyzone`` (run_bw_greyzone.py must
have been run first).  One npz per coupling; rerun resumes missing points.

Usage: run_dssf_sweep.py [<jpm> ...]      (default: the bw_greyzone ladder)
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from scipy.sparse.linalg import eigsh

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "notes"))
sys.path.insert(0, str(ROOT / "campaign"))

import recompute_finite_size_artifact as geometry  # noqa: E402
from qsi_campaign.exact_band import (  # noqa: E402
    fixed_magnetization_basis,
    microscopic_character_hamiltonian,
)
from qsi_campaign.protocol import polarization_sector_labels  # noqa: E402
from run_winding_residual import feshbach_matrix  # noqa: E402
from run_exact_counterterm import embed  # noqa: E402
from run_dssf_counterterm import diagonal_sources, spectral_lines  # noqa: E402

OUTPUT = ROOT / "campaign" / "outputs" / "dssf_sweep"
CACHE = ROOT / "campaign" / "cache" / "bw_greyzone"
DEFAULT_LADDER = [-0.05, -0.10, -0.15, -0.18, -0.20, -0.22,
                  -0.24, -0.26, -0.28, -0.30]
MOMENTA = np.asarray(((0.0, 0.0, 0.0), (0.0, 0.0, 1.0)))   # Gamma, X_z
DEGENERACY_TOL = 1.0e-9
LANCZOS_STEPS = 160


def lanczos_spectrum(hamiltonian, phi, n_iter=LANCZOS_STEPS):
    """Ritz poles and weights of <phi| delta(w - H) |phi>."""
    norm2 = float(np.vdot(phi, phi).real)
    if norm2 < 1e-24:
        return np.zeros(0), np.zeros(0)
    krylov = np.empty((n_iter, len(phi)), dtype=phi.dtype)
    alpha, beta = [], []
    krylov[0] = phi / np.sqrt(norm2)
    r = hamiltonian @ krylov[0]
    for k in range(n_iter):
        a = float(np.vdot(krylov[k], r).real)
        alpha.append(a)
        r = r - a * krylov[k] - (beta[-1] * krylov[k - 1] if beta else 0)
        overlap = krylov[:k + 1].conj() @ r        # full reorthogonalization
        r = r - krylov[:k + 1].T @ overlap
        b = float(np.linalg.norm(r))
        if b < 1e-13 or k == n_iter - 1:
            break
        beta.append(b)
        krylov[k + 1] = r / b
        r = hamiltonian @ krylov[k + 1]
    tridiagonal = np.diag(alpha)
    if beta:
        tridiagonal += np.diag(beta, 1) + np.diag(beta, -1)
    values, vectors = np.linalg.eigh(tridiagonal)
    return values, norm2 * np.abs(vectors[0]) ** 2


def solve_coupling(cluster, jpm, geometry_tables) -> None:
    (states8, states9, ice_rows, complement, cross, lift_rows, sub_of,
     positions) = geometry_tables
    tag = f"jpm{jpm:+.6f}".replace(".", "p")
    out = OUTPUT / f"dssf_{tag}.npz"
    if out.exists():
        print(f"{jpm}: cached", flush=True)
        return
    anchor_file = CACHE / f"bw_{tag}.npz"
    if not anchor_file.exists():
        raise FileNotFoundError(
            f"no bw_greyzone anchor for {jpm}: run run_bw_greyzone.py first")
    with np.load(anchor_file) as saved:
        masked_energies = saved["masked_energies"]
    z0 = float(np.nanmin(masked_energies))
    g_masked = int(np.sum(masked_energies - z0 < DEGENERACY_TOL))

    started = time.perf_counter()
    n_sites = cluster.n_sites
    h8 = microscopic_character_hamiltonian(
        cluster, states8, jpm, np.zeros(3)).tocsc().real
    fold = feshbach_matrix(h8, ice_rows, complement, z0)
    fold = 0.5 * (fold + fold.T)
    masked_fold = np.where(cross, 0.0, fold)
    masked_fold = 0.5 * (masked_fold + masked_fold.T)
    sources = diagonal_sources(cluster, MOMENTA)
    exc_raw, w_raw, _ = spectral_lines(fold, sources, n_sites)
    exc_wf, w_wf, _ = spectral_lines(masked_fold, sources, n_sites)

    corrected = (h8 + embed(-np.where(cross, fold, 0.0), ice_rows,
                            len(states8))).tocsc()
    ev_raw, vec_raw = eigsh(h8, k=8, which="SA", tol=1e-10, ncv=80)
    ev_hc, vec_hc = eigsh(corrected, k=8, which="SA", tol=1e-10, ncv=80)
    order = np.argsort(ev_raw)
    ev_raw, vec_raw = ev_raw[order], vec_raw[:, order]
    order = np.argsort(ev_hc)
    ev_hc, vec_hc = ev_hc[order], vec_hc[:, order]
    g_raw = int(np.sum(ev_raw - ev_raw[0] < DEGENERACY_TOL))
    g_hc = min(g_masked, len(ev_hc))

    h9 = microscopic_character_hamiltonian(
        cluster, states9, jpm, np.zeros(3)).tocsr().real
    results = {}
    for name, vectors, g, reference in (("raw", vec_raw, g_raw, ev_raw[0]),
                                        ("wf", vec_hc, g_hc, z0)):
        for qi, momentum in enumerate(MOMENTA):
            phase = np.exp(-2j * np.pi * (positions @ momentum))
            poles, weights = [], []
            for gs in range(g):
                ground = vectors[:, gs]
                for mu in range(4):
                    phi = np.zeros(len(states9), dtype=complex)
                    for site in np.where(sub_of == mu)[0]:
                        src, dst = lift_rows[site]
                        np.add.at(phi, dst, phase[site] * ground[src])
                    values, wts = lanczos_spectrum(h9, phi)
                    poles.append(values - reference)
                    weights.append(wts * 0.5 / (n_sites * g))
            results[f"x_{name}_poles_q{qi}"] = np.concatenate(poles)
            results[f"x_{name}_weights_q{qi}"] = np.concatenate(weights)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out, jpm=jpm, z0=z0, g_masked=g_masked, g_raw=g_raw,
        e0_raw=ev_raw[0], momenta=MOMENTA,
        z_exc_raw=exc_raw, z_w_raw=w_raw, z_exc_wf=exc_wf, z_w_wf=w_wf,
        **results)
    print(f"{jpm}: saved ({time.perf_counter() - started:.0f}s, "
          f"multiplets raw {g_raw} / wf {g_hc})", flush=True)


def main() -> None:
    ladder = ([float(v) for v in sys.argv[1:]] if len(sys.argv) > 1
              else DEFAULT_LADDER)
    cluster = geometry.build_cluster("cubic", (1, 1, 1))
    n_sites = cluster.n_sites
    states8 = np.sort(fixed_magnetization_basis(n_sites, n_sites // 2))
    states9 = np.sort(fixed_magnetization_basis(n_sites, n_sites // 2 + 1))
    index8 = {int(s): i for i, s in enumerate(states8)}
    ice_rows = np.asarray([index8[int(s)] for s in cluster.ice_states])
    outside = np.ones(len(states8), dtype=bool)
    outside[ice_rows] = False
    complement = np.where(outside)[0]
    labels = polarization_sector_labels(cluster)
    cross = labels[:, None] != labels[None, :]
    bit8 = (states8[:, None] >> np.arange(n_sites, dtype=np.uint64)) \
        & np.uint64(1)
    lift_rows = []
    for site in range(n_sites):
        src = np.where(bit8[:, site] == 0)[0]
        dst = np.searchsorted(states9,
                              states8[src] + np.uint64(1 << site))
        lift_rows.append((src, dst))
    tables = (states8, states9, ice_rows, complement, cross, lift_rows,
              np.arange(n_sites) % 4,
              np.asarray(cluster.positions, dtype=float))
    for jpm in ladder:
        solve_coupling(cluster, jpm, tables)


if __name__ == "__main__":
    main()
