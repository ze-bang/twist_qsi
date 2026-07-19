#!/usr/bin/env python3
"""Nonperturbative validation of a wrapping-loop counterterm on cubic-16.

The production Hamiltonian is microscopic:

    H_imp = H_XXZ + kappa4 * W4 + mu4 * (F4 - <F4>_infty),

where W4 flips every alternating wrapping four-cycle and F4=W4_cycle^2
counts flippable wrapping cycles.  No effective Hamiltonian enters the
calculation.  Perturbative scales are recorded only as blind diagnostics.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh

import recompute_finite_size_artifact as R
from twist_resolved_full_band import sz0_basis

HERE = Path(__file__).resolve().parent
FIGS = HERE / "figs"
FIGS.mkdir(exist_ok=True)


def ring_operator(
    basis: np.ndarray,
    loops: list[tuple[tuple[int, ...], tuple[int, int, int]]],
) -> tuple[sp.csr_matrix, np.ndarray]:
    """Return the alternating-loop flip operator and flippability count."""
    index = {int(s): i for i, s in enumerate(basis)}
    rows: list[int] = []
    cols: list[int] = []
    flippability = np.zeros(len(basis), dtype=float)
    for col, state_u64 in enumerate(basis):
        state = int(state_u64)
        for path, _ in loops:
            bits = [(state >> int(site)) & 1 for site in path]
            if not all(bits[k] != bits[(k + 1) % len(path)] for k in range(len(path))):
                continue
            mask = sum(1 << int(site) for site in path)
            rows.append(index[state ^ mask])
            cols.append(col)
            flippability[col] += 1.0
    vals = np.ones(len(rows), dtype=float)
    op = sp.coo_matrix((vals, (rows, cols)), shape=(len(basis), len(basis))).tocsr()
    asymmetry = op - op.T
    if asymmetry.nnz and np.max(np.abs(asymmetry.data)) > 1e-12:
        raise RuntimeError("ring operator is not Hermitian")
    return op, flippability


def local_ice_projected_ring_operator(
    cluster: R.Cluster,
    basis: np.ndarray,
    loops: list[tuple[tuple[int, ...], tuple[int, int, int]]],
) -> tuple[sp.csr_matrix, np.ndarray]:
    """Flip a loop only when every tetrahedron touching it obeys the ice rule."""
    index = {int(state): position for position, state in enumerate(basis)}
    tetrahedra_by_site = R.site_to_tets(cluster.tets)
    rows: list[int] = []
    columns: list[int] = []
    flippability = np.zeros(len(basis), dtype=float)
    for path, _ in loops:
        first = (basis >> np.uint64(path[0])) & np.uint64(1)
        active = np.ones(len(basis), dtype=bool)
        previous = first
        for site in path[1:]:
            current = (basis >> np.uint64(site)) & np.uint64(1)
            active &= current != previous
            previous = current
        active &= previous != first
        touched_tetrahedra = sorted(
            {tetrahedron for site in path for tetrahedron in tetrahedra_by_site[site]}
        )
        for tetrahedron in touched_tetrahedra:
            up_count = np.bitwise_count(basis & cluster.tet_masks[tetrahedron])
            active &= up_count == 2
        active_columns = np.flatnonzero(active)
        mask = sum(1 << int(site) for site in path)
        active_rows = [index[int(basis[column]) ^ mask] for column in active_columns]
        rows.extend(active_rows)
        columns.extend(active_columns.tolist())
        flippability[active_columns] += 1.0
    operator = sp.coo_matrix(
        (np.ones(len(rows)), (rows, columns)), shape=(len(basis), len(basis))
    ).tocsr()
    asymmetry = operator - operator.T
    if asymmetry.nnz and np.max(np.abs(asymmetry.data)) > 1.0e-12:
        raise RuntimeError("local-ice-projected ring operator is not Hermitian")
    return operator, flippability


def global_ice_projected_ring_operator(
    cluster: R.Cluster,
    basis: np.ndarray,
    loops: list[tuple[tuple[int, ...], tuple[int, int, int]]],
) -> tuple[sp.csr_matrix, np.ndarray]:
    """Flip a loop only in the global two-in/two-out manifold."""
    index = {int(state): position for position, state in enumerate(basis)}
    is_ice = np.ones(len(basis), dtype=bool)
    for tetrahedron_mask in cluster.tet_masks:
        is_ice &= np.bitwise_count(basis & tetrahedron_mask) == 2

    rows: list[int] = []
    columns: list[int] = []
    flippability = np.zeros(len(basis), dtype=float)
    for path, _ in loops:
        first = (basis >> np.uint64(path[0])) & np.uint64(1)
        active = is_ice.copy()
        previous = first
        for site in path[1:]:
            current = (basis >> np.uint64(site)) & np.uint64(1)
            active &= current != previous
            previous = current
        active &= previous != first
        active_columns = np.flatnonzero(active)
        mask = sum(1 << int(site) for site in path)
        active_rows = [index[int(basis[column]) ^ mask] for column in active_columns]
        rows.extend(active_rows)
        columns.extend(active_columns.tolist())
        flippability[active_columns] += 1.0
    operator = sp.coo_matrix(
        (np.ones(len(rows)), (rows, columns)), shape=(len(basis), len(basis))
    ).tocsr()
    asymmetry = operator - operator.T
    if asymmetry.nnz and np.max(np.abs(asymmetry.data)) > 1.0e-12:
        raise RuntimeError("global-ice-projected ring operator is not Hermitian")
    return operator, flippability


def transverse_operator(cl: R.Cluster, basis: np.ndarray) -> sp.csr_matrix:
    index = {int(s): i for i, s in enumerate(basis)}
    rows: list[int] = []
    cols: list[int] = []
    for col, state_u64 in enumerate(basis):
        state = int(state_u64)
        for i_raw, j_raw in cl.bonds:
            i, j = int(i_raw), int(j_raw)
            if ((state >> i) & 1) == ((state >> j) & 1):
                continue
            rows.append(index[state ^ (1 << i) ^ (1 << j)])
            cols.append(col)
    vals = np.ones(len(rows), dtype=float)
    return sp.coo_matrix((vals, (rows, cols)), shape=(len(basis), len(basis))).tocsr()


def flux_labels(cl: R.Cluster) -> np.ndarray:
    """Label ice states by the conserved three-component spin-ice flux."""
    ice = np.asarray(cl.ice_states, dtype=np.uint64)
    bits = (ice[:, None] >> np.arange(cl.n_sites, dtype=np.uint64)) & np.uint64(1)
    sigma = 2 * bits.astype(int) - 1
    local_axes = np.array(
        [[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]],
        dtype=int,
    )
    flux = sigma @ local_axes[np.arange(cl.n_sites) % 4]
    _, labels = np.unique(flux, axis=0, return_inverse=True)
    return labels


def transfer_mixing(
    evals: np.ndarray,
    evecs: np.ndarray,
    ice_indices: np.ndarray,
    labels: np.ndarray,
    betas: np.ndarray,
) -> np.ndarray:
    """Fraction of the ice-to-ice transfer norm crossing parity sectors."""
    projected = evecs[ice_indices, :]
    diff = labels[:, None] != labels[None, :]
    out = []
    energies = evals - evals[0]
    for beta in betas:
        weighted = projected * np.exp(-0.5 * beta * energies)[None, :]
        kernel = weighted @ weighted.conj().T
        norm2 = float(np.sum(np.abs(kernel) ** 2))
        out.append(
            float(np.sum(np.abs(kernel[diff]) ** 2) / max(norm2, 1e-300))
        )
    return np.asarray(out)


def expectation(vec: np.ndarray, op: sp.csr_matrix) -> float:
    return float(np.real(np.vdot(vec, op @ vec)))


def multiplet_expectation(
    evals: np.ndarray,
    evecs: np.ndarray,
    op: sp.csr_matrix,
    atol: float = 1.0e-8,
) -> tuple[float, int]:
    """Basis-independent expectation averaged over the ground multiplet."""
    count = int(np.count_nonzero(evals - evals[0] <= atol))
    values = [expectation(evecs[:, n], op) for n in range(count)]
    return float(np.mean(values)), count


def centered_second_moment(
    cl: R.Cluster,
    jpm: float,
    kappa4: float,
    mu4: float,
    full_components: tuple[np.ndarray, sp.csr_matrix, sp.csr_matrix, np.ndarray],
) -> float:
    h0, transverse, w4, f4 = full_components
    fcenter = f4 - np.mean(f4)
    diag = h0 + mu4 * fcenter
    off = (-jpm) * transverse + kappa4 * w4
    mean = float(np.mean(diag))
    trace_h2 = float(np.dot(diag, diag) + np.sum(np.abs(off.data) ** 2))
    return trace_h2 / len(h0) - mean * mean


def low_peak(evals: np.ndarray, temps: np.ndarray, n_ice: int) -> float:
    curve = R.specific_heat(np.sort(evals)[:n_ice], temps)
    return float(R.refined_peak(temps, curve))


def verify_ice_band(evals: np.ndarray, n_ice: int) -> float:
    """Require all ice descendants to precede the order-Jzz defect band."""
    if len(evals) <= n_ice:
        raise RuntimeError("need at least n_ice + 1 eigenvalues to verify the ice band")
    search_stop = min(len(evals) - 1, n_ice + 30)
    gap_index = int(np.argmax(np.diff(evals[: search_stop + 1])))
    if gap_index != n_ice - 1:
        raise RuntimeError(
            f"incomplete ice band: largest low-spectrum gap follows state "
            f"{gap_index + 1}, expected {n_ice}; increase --n-low/ARPACK space"
        )
    return float(evals[n_ice] - evals[n_ice - 1])


def parse_grid(text: str) -> np.ndarray:
    start, stop, count = text.split(":")
    return np.linspace(float(start), float(stop), int(count))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jpm", type=float, default=-0.05)
    ap.add_argument("--kappa-grid", default="-0.004:0.020:13")
    ap.add_argument("--mu-grid", default="0.0:0.0:1")
    ap.add_argument("--n-low", type=int, default=180)
    ap.add_argument("--betas", default="20,50,100")
    ap.add_argument("--tol", type=float, default=1e-9)
    ap.add_argument("--out", type=Path, default=HERE / "winding_counterterm_16site.npz")
    args = ap.parse_args()

    cl = R.build_cluster("cubic", (1, 1, 1))
    loops4 = [(p, w) for p, w in cl.loops4 if tuple(w) != (0, 0, 0)]
    hex_contract = [(p, w) for p, w in cl.hexes if tuple(w) == (0, 0, 0)]
    hex_wrap = [(p, w) for p, w in cl.hexes if tuple(w) != (0, 0, 0)]
    if len(loops4) != 36 or len(hex_contract) != 16:
        raise RuntimeError("unexpected cubic-16 loop census")

    basis = sz0_basis(cl.n_sites)
    state_index = {int(s): i for i, s in enumerate(basis)}
    ice_indices = np.asarray([state_index[int(s)] for s in cl.ice_states], dtype=int)
    labels = flux_labels(cl)
    h0 = R.ising_energy(cl, basis)
    transverse = transverse_operator(cl, basis)
    w4, f4 = ring_operator(basis, loops4)
    w6c, f6c = ring_operator(basis, hex_contract)
    w6w, _ = ring_operator(basis, hex_wrap)
    f4_center = f4 - 36.0 / 8.0

    full_basis = np.arange(1 << cl.n_sites, dtype=np.uint64)
    full_w4, full_f4 = ring_operator(full_basis, loops4)
    full_components = (
        R.ising_energy(cl, full_basis),
        transverse_operator(cl, full_basis),
        full_w4,
        full_f4,
    )
    m2_bare = centered_second_moment(cl, args.jpm, 0.0, 0.0, full_components)

    kappas = parse_grid(args.kappa_grid)
    mus = parse_grid(args.mu_grid)
    betas = np.asarray([float(x) for x in args.betas.split(",")], dtype=float)
    temps = np.geomspace(2.0e-5, 1.2e-1, 1000)
    records = []
    spectra = []
    v0 = None

    for mu4 in mus:
        for kappa4 in kappas:
            h = (
                sp.diags(h0 + mu4 * f4_center)
                - args.jpm * transverse
                + kappa4 * w4
            ).tocsr()
            t0 = time.time()
            evals, evecs = eigsh(
                h,
                k=args.n_low,
                which="SA",
                tol=args.tol,
                ncv=max(2 * args.n_low + 20, 240),
                v0=v0,
            )
            order = np.argsort(evals)
            evals = np.asarray(evals[order], dtype=float)
            evecs = np.asarray(evecs[:, order], dtype=float)
            v0 = evecs[:, 0]
            defect_gap = verify_ice_band(evals, cl.n_ice)
            mixing = transfer_mixing(evals, evecs, ice_indices, labels, betas)
            m2 = centered_second_moment(
                cl, args.jpm, float(kappa4), float(mu4), full_components
            )
            w4_ground, ground_multiplicity = multiplet_expectation(
                evals, evecs, w4
            )
            w6c_ground, _ = multiplet_expectation(evals, evecs, w6c)
            w6w_ground, _ = multiplet_expectation(evals, evecs, w6w)
            f4_ground, _ = multiplet_expectation(evals, evecs, sp.diags(f4))
            f6c_ground, _ = multiplet_expectation(evals, evecs, sp.diags(f6c))
            rec = {
                "kappa4": float(kappa4),
                "mu4": float(mu4),
                "elapsed_s": float(time.time() - t0),
                "E0": float(evals[0]),
                "E89": float(evals[cl.n_ice - 1]),
                "defect_gap": defect_gap,
                "ground_multiplicity": ground_multiplicity,
                "ice_overlap_ground": float(np.sum(evecs[ice_indices, 0] ** 2)),
                "mixing": [float(x) for x in mixing],
                "mixing_mean": float(np.mean(mixing)),
                "T_peak_low_band": low_peak(evals, temps, cl.n_ice),
                "W4_ground": w4_ground,
                "W6_contract_ground": w6c_ground,
                "W6_wrap_ground": w6w_ground,
                "F4_ground": f4_ground,
                "F6_contract_ground": f6c_ground,
                "m2_infinite_T": float(m2),
                "m2_relative_change": float((m2 - m2_bare) / m2_bare),
            }
            records.append(rec)
            spectra.append(evals)
            print(
                f"k4={kappa4:+.6f} mu4={mu4:+.6f} "
                f"mix={rec['mixing_mean']:.3e} "
                f"Tpk={rec['T_peak_low_band']:.6g} "
                f"dm2={rec['m2_relative_change']:+.3e} "
                f"W6c={rec['W6_contract_ground']:+.4f} "
                f"({rec['elapsed_s']:.1f}s)",
                flush=True,
            )

    m2_scale = max(2.0e-3, max(abs(r["m2_relative_change"]) for r in records))
    for rec in records:
        rec["selection_score"] = (
            rec["mixing_mean"]
            + 0.05 * (rec["m2_relative_change"] / m2_scale) ** 2
        )
    best = min(records, key=lambda r: r["selection_score"])

    summary = {
        "method": "full microscopic ED with symmetry-complete winding counterterms",
        "cluster": {
            "n_sites": int(cl.n_sites),
            "sz0_dim": int(len(basis)),
            "ice_dim": int(cl.n_ice),
            "wrapping_four_loops": int(len(loops4)),
            "contractible_hexagons": int(len(hex_contract)),
            "wrapping_hexagons": int(len(hex_wrap)),
            "flux_sectors": int(len(np.unique(labels))),
        },
        "jpm": float(args.jpm),
        "betas": [float(x) for x in betas],
        "n_low": int(args.n_low),
        "bare_m2_infinite_T": float(m2_bare),
        "blind_perturbative_kappa4": float(4.0 * args.jpm**2),
        "ghex": float(12.0 * abs(args.jpm) ** 3),
        "best": best,
        "records": records,
    }
    args.out.with_suffix(".json").write_text(json.dumps(summary, indent=2))
    np.savez_compressed(
        args.out,
        kappas=kappas,
        mus=mus,
        betas=betas,
        spectra=np.asarray(spectra),
        summary=json.dumps(summary),
    )

    zero_mu = [r for r in records if abs(r["mu4"]) < 1e-15]
    zero_mu.sort(key=lambda r: r["kappa4"])
    x = np.asarray([r["kappa4"] for r in zero_mu])
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.35))
    axes[0].semilogy(x, [r["mixing_mean"] for r in zero_mu], "o-", color="#007c83")
    axes[0].axvline(4.0 * args.jpm**2, color="0.4", ls=":", label="blind $g_4$")
    axes[0].set_ylabel("winding transfer fraction")
    axes[0].set_xlabel(r"$\kappa_4/J_{zz}$")
    axes[0].legend(frameon=False, fontsize=8)

    axes[1].plot(
        x,
        np.asarray([r["T_peak_low_band"] for r in zero_mu])
        / (12.0 * abs(args.jpm) ** 3),
        "o-",
        color="#7b3294",
    )
    axes[1].axvline(4.0 * args.jpm**2, color="0.4", ls=":")
    axes[1].set_ylabel(r"$T_{\rm peak}/g_{\rm hex}$")
    axes[1].set_xlabel(r"$\kappa_4/J_{zz}$")

    axes[2].plot(
        x,
        [r["W6_contract_ground"] for r in zero_mu],
        "o-",
        color="#00875a",
        label="contractible hex.",
    )
    axes[2].plot(
        x,
        [r["W6_wrap_ground"] for r in zero_mu],
        "s--",
        color="#d55e00",
        label="wrapping hex.",
    )
    axes[2].set_ylabel("ground-state ring expectation")
    axes[2].set_xlabel(r"$\kappa_4/J_{zz}$")
    axes[2].legend(frameon=False, fontsize=8)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        rf"Microscopic cubic-16 counterterm validation, $J_\pm={args.jpm:+.3f}$",
        fontsize=10,
    )
    fig.tight_layout()
    figure_stem = f"fig_{args.out.stem}"
    for ext in ("pdf", "png"):
        fig.savefig(FIGS / f"{figure_stem}.{ext}", dpi=220)
    plt.close(fig)

    print(json.dumps({"best": best}, indent=2))
    print(f"wrote {args.out}")
    print(f"wrote {args.out.with_suffix('.json')}")
    print(f"wrote {FIGS / (figure_stem + '.pdf')}")


if __name__ == "__main__":
    main()
