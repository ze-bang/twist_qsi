"""The microscopic crux, checked on configurations rather than operators.

In the ice basis the probe is DIAGONAL,
    a(c) = sum_i e^{-i q.r_i} S^z_i(c),
and the ring exchange W_p maps a configuration c to c' with plaquette p
flipped.  The commutator identity [A_q, W_p] = g_p(q) W_p is then a statement
about configurations:

    a(c') - a(c) = +/- g_p(q).

So g_p is exactly the amount by which flipping plaquette p shifts the
density-wave amplitude.  For a DARK plaquette g_p = 0 and flipping it does not
change the amplitude AT ALL.

Why that makes the soft mode cheap.  The excitation is the ground state
modulated by the density wave, psi(c) ~ a(c) psi_0(c).  The energy is ring
coherence, sum over the pairs (c, c') that W_p connects:

    <psi|W_p|psi>  ~  sum  a(c')* a(c) psi_0(c') psi_0(c)

  * dark p   : a(c') = a(c), so the factor is |a(c)|^2 > 0 -- the two
               configurations still resonate IN PHASE and no coherence is lost.
  * bright p : a(c') = a(c) +/- g_p, so the two configurations carry different
               modulation factors, the resonance is dephased, and coherence
               (energy) is lost.

That is precisely why the f-sum rule reads f1 = (K/2N) sum_p <W_p+W_p'> |g_p|^2:
|g_p|^2 IS the dephasing of the ring resonance on p caused by the imposed wave.

This script verifies the configuration identity directly, and measures the
resulting dephasing angle family by family.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for sub in ("src", "notes", "campaign"):
    sys.path.insert(0, str(ROOT / sub))

import recompute_finite_size_artifact as geometry              # noqa: E402
from run_ring48_dssf import contractible                       # noqa: E402
from run_momentum_resolved_spectrum import star                # noqa: E402
from ring48_momenta_fix import fcc_momenta                     # noqa: E402
from ice_enumerate_fast import enumerate_ice_fast, bit_of      # noqa: E402
from f1_all_momenta import hex_family, geom_factor             # noqa: E402

U64 = np.uint64


def main() -> int:
    t0 = time.perf_counter()
    shape = (2, 2, 3)

    orig = geometry.enumerate_ice_states
    geometry.enumerate_ice_states = lambda a, b: np.array([0], dtype=U64)
    cl = geometry.build_cluster("fcc", shape)
    geometry.enumerate_ice_states = orig
    n = int(cl.n_sites)
    pos = np.asarray(cl.positions, dtype=float)
    tets = [tuple(int(s) for s in t) for t in cl.tets]
    hexes = contractible(cl)
    fam = hex_family(hexes, n)

    states = enumerate_ice_fast(n, tets, log=lambda m: None)
    print(f"{n} sites, {len(hexes)} hexagons, {len(states):,} ice states")

    q = [qq for qq in fcc_momenta(cl)[0] if star(qq) == "L"][0]
    ph = np.exp(-2j * np.pi * (pos @ q))
    g = geom_factor(hexes, pos, q, n)
    gf = np.array([float(g[fam == mu].sum()) for mu in range(4)])
    dark = [mu for mu in range(4) if gf[mu] < 1e-9]
    print(f"q = {np.round(q, 4)}   geometric weight by family "
          f"{np.round(gf, 1)}   dark = {dark}\n")

    # The probe is SUBLATTICE-RESOLVED: S^zz sums |<n|S^z_mu(q)|0>|^2 over mu,
    # so the operators are A^mu_q = sum_{i in mu} e^{-iq.r_i} S^z_i, and the
    # relevant curl is g^mu_p.  (The sublattice-SUMMED A_q commutes with every
    # ring exchange at L -- its total alternating sum vanishes on every
    # hexagon -- so it is not the right operator here.)
    subl = np.arange(n) % 4
    gp = np.zeros((len(hexes), 4), dtype=complex)
    for i, h in enumerate(hexes):
        for j, sIdx in enumerate(h):
            gp[i, subl[int(sIdx)]] += ((-1) ** j) * ph[int(sIdx)]
    tot = np.abs(gp).sum(axis=1)
    print(f"check: sum_mu |g^mu_p|^2 by family = "
          f"{[round(float((np.abs(gp[fam==mu])**2).sum()), 1) for mu in range(4)]}"
          f"   (must match the geometric weights above)\n")

    # decode a batch of configurations as +/-1/2 spins.  States are stored as
    # (lo, hi) uint64 pairs, so go through bit_of rather than shifting.
    def spins(w):
        b = np.zeros((len(w), n))
        for s in range(n):
            b[:, s] = np.asarray(bit_of(w, s), dtype=float) - 0.5
        return b

    sub = states[:4000]
    sp = spins(sub)
    amp = sp @ ph                       # a(c) for each configuration

    print("Flip each hexagon in every configuration where it is flippable,")
    print("and compare the change in a(c) with g_p:\n")
    print(f"  {'family':>7s} {'n_hex':>6s} {'max_mu |g^mu_p|':>16s} "
          f"{'max |da^mu -/+ g^mu_p|':>24s} {'mean |da^mu|':>13s}")
    print("  " + "-" * 72)

    for mu in range(4):
        idx = [i for i in range(len(hexes)) if fam[i] == mu]
        worst, das = 0.0, []
        for i in idx:
            h = [int(s) for s in hexes[i]]
            sgn = sp[:, h] * np.array([(-1.0) ** j for j in range(6)])
            ok = np.all(np.abs(sgn - sgn[:, [0]]) < 1e-12, axis=1)
            if not ok.any():
                continue
            old = sp[ok][:, h]
            new = -old
            # change in each sublattice-resolved amplitude
            da = np.zeros((old.shape[0], 4), dtype=complex)
            for j, sIdx in enumerate(h):
                da[:, subl[int(sIdx)]] += (new[:, j] - old[:, j]) * ph[int(sIdx)]
            das.append(np.abs(da))
            # the sign is per CONFIGURATION (which way the ring flips), so
            # test each row against +g and -g separately
            e_plus = np.abs(da - gp[i]).max(axis=1)
            e_minus = np.abs(da + gp[i]).max(axis=1)
            worst = max(worst, float(np.minimum(e_plus, e_minus).max()))
        mg = float(np.abs(gp[idx]).max())
        mda = float(np.concatenate(das).mean()) if das else 0.0
        tag = "  <== DARK" if mu in dark else ""
        print(f"  {mu:>7d} {len(idx):>6d} {mg:>16.6f} {worst:>24.3e} "
              f"{mda:>13.6f}{tag}")

    print(f"""
  So on the dark family |g_p| = 0 and flipping changes the density-wave
  amplitude by exactly nothing: the two configurations the ring exchange
  connects carry the SAME modulation factor and keep resonating in phase.
  On the three bright families every flip shifts the amplitude, the pair
  falls out of phase, and that lost coherence IS the excitation energy.
""")
    print(f"[{time.perf_counter()-t0:.1f}s] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
