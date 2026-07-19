#!/usr/bin/env python3
"""Test the direct microscopic character-average protocol on cubic 16.

Candidate A in the simulation plan is

    H_avg = (1 / |G|) sum_theta H_mic^{2 delta}(theta)

followed by ED on H_avg.  This script checks whether that direct full
Hamiltonian average removes the winding four-loop artifact or merely changes
its coefficient.  The diagnostic is the scaling of the low specific-heat peak:
the physical hexagon process scales as |Jpm|^3, while the spurious four-loop
scale is quadratic in Jpm.
"""
from __future__ import annotations

import json
import time
from itertools import product
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import eigsh

import recompute_finite_size_artifact as R
from twist_resolved_full_band import sz0_basis

HERE = Path(__file__).resolve().parent
FIGS = HERE / "figs"
FIGS.mkdir(exist_ok=True)


def precompute_dipole2_transitions(cl: R.Cluster, states: np.ndarray):
    """Precompute matrix transitions and averaged 2-delta character factors."""
    state_index = {int(s): i for i, s in enumerate(states)}
    one = np.uint64(1)
    twists = list(product([0.0, np.pi], repeat=3))
    transitions = []
    for col, st in enumerate(states):
        for (i, j), n in zip(cl.bonds, cl.bond_wrap):
            bi = one << np.uint64(i)
            bj = one << np.uint64(j)
            up_i = (st & bi) != 0
            up_j = (st & bj) != 0
            if up_i == up_j:
                continue
            d_ij = cl.positions[j] - cl.positions[i] - np.asarray(n) @ cl.Lvecs
            q = 2.0 * d_ij
            avg = np.mean([np.exp(1j * float(np.dot(phi, q))) for phi in twists])
            row = state_index[int(st ^ (bi | bj))]
            is_splus_i_sminus_j = (not up_i) and up_j
            transitions.append((row, col, avg, is_splus_i_sminus_j))
    return transitions


def direct_average_hamiltonian(cl, states, transitions, jpm):
    dim = len(states)
    rows = list(range(dim))
    cols = list(range(dim))
    vals = list(R.ising_energy(cl, states).astype(complex))
    for row, col, avg, is_splus_i_sminus_j in transitions:
        coeff = -jpm * avg
        amp = coeff if is_splus_i_sminus_j else np.conjugate(coeff)
        rows.append(row)
        cols.append(col)
        vals.append(amp)
    return coo_matrix((np.asarray(vals, complex), (rows, cols)), shape=(dim, dim)).tocsr()


def low_peak(evals, temps):
    curve = R.specific_heat(np.sort(evals)[:90], temps)
    return float(R.refined_peak(temps, curve)), curve


def main():
    cl = R.build_cluster("cubic", (1, 1, 1))
    states = sz0_basis(cl.n_sites)
    state_index = {int(s): i for i, s in enumerate(states)}
    ice_indices = np.asarray([state_index[int(s)] for s in cl.ice_states], dtype=np.int64)
    transitions = precompute_dipole2_transitions(cl, states)
    temps = np.geomspace(3.0e-5, 8.0e-2, 1000)
    jpms = [-0.02, -0.03, -0.04, -0.05, -0.06]

    rows = []
    curves = {}
    for jpm in jpms:
        h = direct_average_hamiltonian(cl, states, transitions, jpm)
        t0 = time.time()
        evals = np.sort(
            eigsh(h, k=120, which="SA", return_eigenvectors=False, tol=1e-9, ncv=280).real
        )
        elapsed = time.time() - t0
        t_peak, curve = low_peak(evals, temps)
        curves[str(jpm)] = curve
        g4 = 4.0 * jpm * jpm
        ghex = 12.0 * abs(jpm) ** 3
        rows.append(
            {
                "Jpm": float(jpm),
                "E0": float(evals[0]),
                "E89": float(evals[89]),
                "E119": float(evals[-1]),
                "T_peak_low90": t_peak,
                "g4": g4,
                "ghex": ghex,
                "T_peak_over_g4": t_peak / g4,
                "T_peak_over_ghex": t_peak / ghex,
                "elapsed_s": elapsed,
            }
        )

    # Ice-overlap diagnostic at the representative Jpm used in the notes.
    jpm0 = -0.05
    h0 = direct_average_hamiltonian(cl, states, transitions, jpm0)
    evals0, vecs0 = eigsh(h0, k=120, which="SA", tol=1e-10, ncv=300)
    order = np.argsort(evals0)
    evals0 = evals0[order].real
    vecs0 = vecs0[:, order]
    overlap = np.sum(np.abs(vecs0[ice_indices, :]) ** 2, axis=0)
    overlap_summary = {
        "Jpm": jpm0,
        "low90_min": float(np.min(overlap[:90])),
        "low90_mean": float(np.mean(overlap[:90])),
        "low90_max": float(np.max(overlap[:90])),
        "states90to119_min": float(np.min(overlap[90:])),
        "states90to119_mean": float(np.mean(overlap[90:])),
        "states90to119_max": float(np.max(overlap[90:])),
    }

    out = {
        "method": "direct microscopic 2delta M=2 character average, then ED",
        "cluster": {"basis": "cubic", "shape": [1, 1, 1], "n_sites": cl.n_sites},
        "sz0_dim": int(len(states)),
        "ice_dim": int(cl.n_ice),
        "n_transitions": int(len(transitions)),
        "result": "fails hexagon scaling: low peak follows the quadratic four-loop scale",
        "rows": rows,
        "ice_overlap_representative": overlap_summary,
    }
    out_path = HERE / "direct_microscopic_average_16site_scaling.json"
    out_path.write_text(json.dumps(out, indent=2))

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(8.4, 3.4))
    for rec in rows:
        jpm = rec["Jpm"]
        ax0.plot(temps, curves[str(jpm)] / cl.n_sites, lw=1.4, label=f"{jpm:+.2f}")
    ax0.set_xscale("log")
    ax0.set_xlabel("$T/J_{zz}$")
    ax0.set_ylabel("$C(T)/N$")
    ax0.set_title("Direct microscopic average")
    ax0.legend(title="$J_\\pm/J_{zz}$", frameon=False, fontsize=7)
    ax0.spines[["top", "right"]].set_visible(False)

    jabs = np.asarray([abs(r["Jpm"]) for r in rows])
    tpk = np.asarray([r["T_peak_low90"] for r in rows])
    ax1.loglog(jabs, tpk, "o-", color="#1f77b4", label="direct avg.")
    ref = tpk[-1] * (jabs / jabs[-1]) ** 2
    ax1.loglog(jabs, ref, "--", color="#d95f02", label="$\\propto |J_\\pm|^2$")
    ref3 = tpk[-1] * (jabs / jabs[-1]) ** 3
    ax1.loglog(jabs, ref3, ":", color="#7570b3", label="$\\propto |J_\\pm|^3$")
    ax1.set_xlabel("$|J_\\pm|/J_{zz}$")
    ax1.set_ylabel("$T_{\\rm peak}/J_{zz}$")
    ax1.set_title("Scale test")
    ax1.legend(frameon=False, fontsize=8)
    ax1.spines[["top", "right"]].set_visible(False)
    for ext in ("pdf", "png"):
        fig.savefig(FIGS / f"fig_direct_microscopic_average_16site_scaling.{ext}",
                    bbox_inches="tight", dpi=220)
    plt.close(fig)

    print(json.dumps(out, indent=2))
    print(f"wrote {out_path}")
    print(f"wrote {FIGS / 'fig_direct_microscopic_average_16site_scaling.pdf'}")


if __name__ == "__main__":
    main()
