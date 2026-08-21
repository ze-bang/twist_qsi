"""What IS the roton, concretely, in real space?

The roton at L is (mostly) the state the neutron makes,
    |psi> = P_roton S^z(L)|0> / ||...||,
where P_roton projects onto the degenerate multiplet carrying the line.
We ask three concrete questions about that state:

 1. WHICH HEXAGONS does the probe couple to at L?  Each pyrochlore hexagon
    lies in a plane perpendicular to one of the four <111> axes.  The form
    factor |g_p(L)|^2 of the analytic sum rule is orientation resolved: if
    it concentrates on the hexagons perpendicular to the [111] direction of
    L itself, the mode selects one hexagon family, and "why L" acquires a
    picture rather than a number.

 2. WHICH HEXAGONS RESONATE MORE?  The Rokhsar-Kivelson slope says the
    roton is resonance rich.  Resolve that per hexagon family:
        Delta_p = <psi|W_p + W_p^dag|psi> - <0|W_p + W_p^dag|0>
    and the flippability <n_p^flip> likewise.  This says, concretely, that
    the roton is a rearrangement of the spins into a pattern the ring
    exchange likes better -- which is why it costs less than a photon.

 3. HOW IS IT MODULATED IN SPACE?  Project the flippability change onto
    the L wavevector and onto each hexagon family, to see the standing
    wave: which (111) planes gain resonance and which lose it.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "notes"))
sys.path.insert(0, str(ROOT / "campaign"))

import recompute_finite_size_artifact as geometry               # noqa: E402
from run_momentum_resolved_spectrum import star                 # noqa: E402
from run_ring48_dssf import contractible                        # noqa: E402
from ring48_momenta_fix import fcc_momenta                      # noqa: E402
from ring64_pipeline import (fast_labels, fast_ring,            # noqa: E402
                             hex_masks_patterns, probe_columns)

OUT = ROOT / "campaign" / "outputs" / "ring_model_dssf"
EDEGEN = 1e-8
AXES = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]],
                dtype=float) / np.sqrt(3.0)


def hex_normals(hexes, positions):
    """Family index of each hexagon = the sublattice it omits.

    A pyrochlore hexagon visits three of the four sublattices twice each
    and lies in the plane perpendicular to the <111> axis of the fourth.
    Identifying the family combinatorially (by the missing sublattice) is
    exact and immune to periodic wrapping, which corrupts a plane fit:
    three quarters of the six-cycles cross the box boundary.
    """
    family = []
    for path in hexes:
        p = np.asarray(path, dtype=int)
        missing = sorted(set(range(4)) - set((p % 4).tolist()))
        if len(missing) != 1:
            raise RuntimeError(f"hexagon sublattice content {sorted(p % 4)}")
        family.append(missing[0])
    family = np.array(family)
    return AXES[family], family


def plaq_expect(sec_states, hmp, psi):
    """<psi|W_p + W_p^dag|psi> and <psi|n_p^flip|psi>, per plaquette."""
    wout, nout = [], []
    for mask, p1, p2 in hmp:
        sel = sec_states & mask
        flip = (sel == p1) | (sel == p2)
        idx = np.where(flip)[0]
        nout.append(float(np.sum(np.abs(psi[idx]) ** 2)))
        if len(idx) == 0:
            wout.append(0.0)
            continue
        tgt = sec_states[idx] ^ mask
        pos = np.searchsorted(sec_states, tgt)
        ok = (pos < len(sec_states)) & (sec_states[np.minimum(
            pos, len(sec_states) - 1)] == tgt)
        wout.append(float(np.sum(psi[idx[ok]].conj() * psi[pos[ok]]).real))
    return np.array(wout), np.array(nout)


def main() -> int:
    t0 = time.perf_counter()
    rec = {}

    cl = geometry.build_cluster("fcc", (2, 2, 3))
    n = int(cl.n_sites)
    states = np.sort(np.asarray(cl.ice_states, dtype=np.uint64))
    positions = np.asarray(cl.positions, dtype=float)
    labels, spins_all = fast_labels(states, positions)
    hexes = contractible(cl)
    hmp = hex_masks_patterns(hexes)
    momenta, _ = fcc_momenta(cl)
    normals, family = hex_normals(hexes, positions)
    hcent = np.array([positions[np.asarray(h, dtype=int)].mean(axis=0)
                      for h in hexes])
    print("hexagon families (perp to <111> axes): "
          f"{np.bincount(family, minlength=4)}")

    ham, _ = fast_ring(states, hmp, 0.0)
    _, v0 = eigsh(ham, k=1, which="SA", tol=1e-11, ncv=60)
    per = {int(s): float(np.sum(v0[labels == s, 0] ** 2))
           for s in np.unique(labels)}
    sector = max(per, key=per.get)
    rows = np.where(labels == sector)[0]
    hs = np.asarray(ham[np.ix_(rows, rows)].todense())
    values, vectors = eigh(hs)
    e0 = float(values[0])
    omega = values - e0
    gs = vectors[:, 0].astype(complex)
    sec_states = states[rows]

    Lq = next(qi for qi, m in enumerate(momenta) if star(m) == "L")
    qL = np.asarray(momenta[Lq], dtype=float)
    print(f"L momentum (reduced): {qL}")

    # ---- 1. probe form factor per hexagon family, |g_p(L)|^2 ----
    phase = np.exp(-2j * np.pi * (positions @ qL))
    sub = np.arange(n) % 4
    gsq = np.zeros(len(hexes))
    for j, path in enumerate(hexes):
        p = np.asarray(path, dtype=int)
        signs = np.where(np.arange(6) % 2 == 0, 1.0, -1.0)
        z = signs * phase[p]
        for mu in range(4):
            sel = sub[p] == mu
            if sel.any():
                gsq[j] += abs(z[sel].sum()) ** 2
    print("\n1. probe form factor |g_p(L)|^2 by hexagon family:")
    for f in range(4):
        m = family == f
        print(f"   family {f} (perp to <111> axis {np.round(AXES[f]*np.sqrt(3)).astype(int).tolist()}): "
              f"n={m.sum():2d}  mean |g|^2 = {gsq[m].mean():.4f}")
    rec["gsq_by_family"] = [float(gsq[family == f].mean())
                            for f in range(4)]

    # ---- roton state: probe projected onto its degenerate multiplet ----
    cols = probe_columns(spins_all[rows], positions, qL, gs, n)
    wq = np.zeros(len(values))
    amp = np.zeros((len(values),), dtype=complex)
    for col in cols:
        pr = vectors.conj().T @ col
        wq += np.abs(pr) ** 2 / n
        amp += pr
    jb = int(np.argmax(wq))
    grp = np.where(np.abs(values - values[jb]) < EDEGEN)[0]
    print(f"\nroton line: omega = {omega[jb]:.5f} K, "
          f"degeneracy {len(grp)}, share {wq[grp].sum()/wq.sum():.3f}")
    psi = vectors[:, grp] @ amp[grp]
    psi = psi / np.linalg.norm(psi)

    # ---- 2. resonance change per hexagon family ----
    wg, ng = plaq_expect(sec_states, hmp, gs)
    wr, nr = plaq_expect(sec_states, hmp, psi.astype(complex))
    dW, dN = wr - wg, nr - ng
    print("\n2. resonance change in the roton, by hexagon family:")
    print("   family   <W+Wd> gs    roton     delta      <n_flip> delta")
    for f in range(4):
        m = family == f
        print(f"   {f}        {wg[m].mean():+.5f}   {wr[m].mean():+.5f}  "
              f"{dW[m].mean():+.5f}    {dN[m].mean():+.5f}")
    print(f"   TOTAL    {wg.sum():+.5f}   {wr.sum():+.5f}  "
          f"{dW.sum():+.5f}    {dN.sum():+.5f}")
    print(f"   (energy check: -sum dW = {-dW.sum():+.5f} K "
          f"vs omega = {omega[jb]:+.5f} K)")
    rec["dW_by_family"] = [float(dW[family == f].mean()) for f in range(4)]
    rec["dN_by_family"] = [float(dN[family == f].mean()) for f in range(4)]
    rec["omega_roton"] = float(omega[jb])
    rec["dW_total"] = float(dW.sum())

    # ---- 3. spatial modulation at the L wavevector ----
    print("\n3. modulation of the resonance change at wavevector L:")
    ph = np.exp(-2j * np.pi * (hcent @ qL))
    for f in range(4):
        m = family == f
        mod = np.abs(np.sum(dW[m] * ph[m])) / max(m.sum(), 1)
        uni = np.abs(np.mean(dW[m]))
        print(f"   family {f}: |L-component| {mod:.5f}   "
              f"|uniform| {uni:.5f}")
    rec["mod_by_family"] = [
        float(np.abs(np.sum(dW[family == f] * ph[family == f]))
              / max((family == f).sum(), 1)) for f in range(4)]

    np.savez_compressed(OUT / "roton_realspace.npz", family=family,
                        normals=normals, hcent=hcent, gsq=gsq, dW=dW,
                        dN=dN, wg=wg, ng=ng, qL=qL,
                        omega_roton=omega[jb])
    with open(OUT / "roton_realspace.json", "w") as fh:
        json.dump(rec, fh, indent=1, default=float)
    print(f"\nwrote roton_realspace.* ({time.perf_counter()-t0:.0f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
