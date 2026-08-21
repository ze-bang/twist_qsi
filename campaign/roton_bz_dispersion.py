"""Feynman-Bijl roton dispersion over the FULL Brillouin zone.

Two ingredients, neither of which needs a big quantum calculation:

  f_1(q) = (w/8) sum_orientations sum_mu |g^mu_o(q)|^2      [ANALYTIC]
      g^mu_o(q) = sum_{j in hexagon o, sublattice mu} delta_j e^{-i q.d_j},
      delta_j = +-1 alternating around the hexagon, d_j the site offsets.
      Translation invariance collapses the plaquette sum to the 4 hexagon
      orientations of one fcc cell.  w = <W_p+W_p^dag> = -E_0/(K N_p) is a
      single ground-state number (an overall scale: it cannot move minima).

  S(q)  = equal-time structure factor of the ICE MANIFOLD, sampled by
      classical loop Monte Carlo on an L^3 pyrochlore.  Licensed by the
      gate in ring48_roton_landing.py (classical vs quantum S(q)).

  omega_SMA(q) = f_1(q)/S(q)  is then a variational UPPER BOUND on the
  lowest C-odd excitation at every q in the zone.  Its minima are where
  rotons live.  This is exactly Feynman's argument for helium, with the
  ice manifold's S(q) in place of the liquid's.

Outputs: path Gamma-X-W-L-Gamma-K-X, an (hhl) slice, and the list of
local minima = the predicted roton momenta.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "notes"))
sys.path.insert(0, str(ROOT / "campaign"))
OUT = ROOT / "campaign" / "outputs" / "ring_model_dssf"

# cubic-cell units, a = 1
A1 = np.array([0.0, 0.5, 0.5])
A2 = np.array([0.5, 0.0, 0.5])
A3 = np.array([0.5, 0.5, 0.0])
AV = np.array([A1, A2, A3])
BASIS = np.array([[0.0, 0.0, 0.0], [0.0, 0.25, 0.25],
                  [0.25, 0.0, 0.25], [0.25, 0.25, 0.0]])
LBOX = 8
NSWEEP_EQ = 40000
NMEAS = 1500
NLOOP_PER_MEAS = 40


# ----------------------------------------------------------------- lattice
def build(L):
    cells = np.array([(i, j, k) for i in range(L) for j in range(L)
                      for k in range(L)], dtype=int)
    ncell = len(cells)
    cidx = {tuple(c): n for n, c in enumerate(cells)}
    pos = np.zeros((ncell * 4, 3))
    sub = np.zeros(ncell * 4, dtype=int)
    for n, c in enumerate(cells):
        R = c @ AV
        for mu in range(4):
            pos[4 * n + mu] = R + BASIS[mu]
            sub[4 * n + mu] = mu
    # tetrahedra: A at cell n = sites (n,mu); B at cell n = (n,0),(n-a_mu,mu)
    tetA = np.array([[4 * n + mu for mu in range(4)] for n in range(ncell)])
    tetB = np.zeros((ncell, 4), dtype=int)
    for n, c in enumerate(cells):
        row = [4 * n + 0]
        for mu in range(1, 4):
            sh = c.copy()
            sh[mu - 1] = (sh[mu - 1] - 1) % L
            row.append(4 * cidx[tuple(sh)] + mu)
        tetB[n] = row
    return pos, sub, tetA, tetB, cells, cidx


def site_tetra(tetA, tetB, nsite):
    """For each site: (A tetra, B tetra) and its slot in each."""
    ta = np.zeros(nsite, dtype=int)
    tb = np.zeros(nsite, dtype=int)
    for t, row in enumerate(tetA):
        for s in row:
            ta[s] = t
    for t, row in enumerate(tetB):
        for s in row:
            tb[s] = t
    return ta, tb


# ------------------------------------------------------------- ice + loops
def seed_ice(sub):
    """Exact ice state: sigma depends only on sublattice, +1 for mu in {0,1}.

    Both tetrahedron types contain one site of each sublattice (A trivially;
    B by construction (n,0),(n-a_mu,mu)), so the sum is 1+1-1-1 = 0 on every
    tetrahedron of either type.  Loop moves then randomize it.
    """
    return np.where(sub < 2, 1, -1).astype(np.int8)


def ice_defects(sigma, tetA, tetB):
    a = np.count_nonzero(sigma[tetA].sum(axis=1))
    b = np.count_nonzero(sigma[tetB].sum(axis=1))
    return a, b


def loop_update(sigma, tetA, tetB, ta, tb, rng):
    """Standard short-loop move: preserves 2-in-2-out on every tetra."""
    nsite = len(sigma)
    start = int(rng.integers(nsite))
    path = [start]
    seen = {start: 0}
    site = start
    on_A = True                      # move into the site's A tetra first
    for _ in range(4 * nsite):
        t = ta[site] if on_A else tb[site]
        row = tetA[t] if on_A else tetB[t]
        cin = sigma[site] if on_A else -sigma[site]
        cand = [s for s in row if s != site and
                (sigma[s] if on_A else -sigma[s]) == -cin]
        if not cand:
            return 0
        nxt = int(cand[rng.integers(len(cand))])
        if nxt in seen:
            # cycle s_k..s_m closes through tetra T_m; site s_k then has
            # neighbours via T_k and T_m, which must be of OPPOSITE type
            # for both of its tetrahedra to stay balanced -> (m-k) odd.
            k = seen[nxt]
            m = len(path) - 1
            if (m - k) % 2 == 0:
                return 0                      # invalid closure, abandon
            loop = path[k:]
            sigma[loop] *= -1
            return len(loop)
        seen[nxt] = len(path)
        path.append(nxt)
        site = nxt
        on_A = not on_A
    return 0


# -------------------------------------------------------------- f1 analytic
def hexagon_templates():
    """4 hexagon orientations: site offsets, sublattices, alternating signs.

    Built from the 48-site cluster's own contractible hexagons so that the
    template provably matches the Hamiltonian actually diagonalized.
    """
    import recompute_finite_size_artifact as geometry
    from run_ring48_dssf import contractible
    cl = geometry.build_cluster("fcc", (2, 2, 3))
    pos = np.asarray(cl.positions, dtype=float)
    hexes = contractible(cl)
    temps, seen = [], set()
    for path in hexes:
        p = np.asarray(path, dtype=int)
        sub = p % 4
        off = pos[p] - pos[p[0]]
        key = tuple(np.round(np.sort(sub), 3))
        missing = tuple(sorted(set(range(4)) - set(sub.tolist())))
        if missing in seen:
            continue
        seen.add(missing)
        signs = np.where(np.arange(6) % 2 == 0, 1.0, -1.0)
        temps.append({"sub": sub, "off": off, "signs": signs,
                      "missing": missing})
    return temps


def f1_of_q(qs, temps, w):
    """f_1(q) = (w/8) sum_o sum_mu |g^mu_o(q)|^2 ; qs in cubic 2pi units."""
    out = np.zeros(len(qs))
    for t in temps:
        ph = np.exp(-2j * np.pi * (qs @ t["off"].T))     # (Nq, 6)
        z = ph * t["signs"][None, :]
        for mu in range(4):
            sel = t["sub"] == mu
            if sel.any():
                out += np.abs(z[:, sel].sum(axis=1)) ** 2
    return w * out / 8.0


# ------------------------------------------------------------------- paths
def bz_path(npts=60):
    G = np.array([0, 0, 0.0])
    X = np.array([0, 0, 1.0])
    W = np.array([0.5, 0, 1.0])
    L = np.array([0.5, 0.5, 0.5])
    K = np.array([0.75, 0.75, 0.0])
    segs = [(G, X, "$\\Gamma$", "X"), (X, W, "X", "W"), (W, L, "W", "L"),
            (L, G, "L", "$\\Gamma$"), (G, K, "$\\Gamma$", "K"),
            (K, X, "K", "X")]
    qs, ticks, lab, acc = [], [0.0], ["$\\Gamma$"], 0.0
    for a, b, la, lb in segs:
        d = np.linalg.norm(b - a)
        for i in range(npts):
            qs.append(a + (b - a) * i / npts)
        acc += d
        ticks.append(acc)
        lab.append(lb)
    qs.append(segs[-1][1])
    return np.array(qs), np.array(ticks), lab


def main() -> int:
    t0 = time.perf_counter()
    rng = np.random.default_rng(20260820)
    rec = {}

    temps = hexagon_templates()
    print(f"hexagon orientations: {len(temps)}  "
          f"(missing sublattice each: {[t['missing'] for t in temps]})")

    # scale w from the 48-site ground state
    land = OUT / "ring48_roton_landing.json"
    w = 10.556 / 48.0
    if land.exists():
        w = float(json.load(open(land))["w_plaquette"])
    print(f"w = <W_p+W_p^dag> = {w:.6f}")
    rec["w"] = w

    # ---- gate: reproduce the 48-site cluster f_1 at its own momenta ----
    if land.exists():
        tab = json.load(open(land))["momenta"]
        import recompute_finite_size_artifact as geometry
        cl = geometry.build_cluster("fcc", (2, 2, 3))
        Lv = np.asarray(cl.lattice_vectors, dtype=float) \
            if hasattr(cl, "lattice_vectors") else None
        print("\nGATE C  (template f_1 vs cluster f_1):")
        for row in tab:
            qcart = np.asarray(row["q"], dtype=float)
            fq = float(f1_of_q(np.array([qcart]), temps, w)[0])
            print(f"  {row['star']:<18s} template {fq:.5f}  "
                  f"cluster {row['f1_exact']:.5f}  "
                  f"ratio {fq/row['f1_exact']:.4f}")
        rec["gate_C"] = "see log"

    # ---------------- classical ice Monte Carlo ----------------
    pos, sub, tetA, tetB, cells, cidx = build(LBOX)
    nsite = len(pos)
    ta, tb = site_tetra(tetA, tetB, nsite)
    sigma = seed_ice(sub)
    da, db = ice_defects(sigma, tetA, tetB)
    print(f"\nlattice L={LBOX}: {nsite} sites, ice defects "
          f"A={da} B={db}")
    if da or db:
        print("  seeding failed -> abort")
        return 1

    acc = 0
    for _ in range(NSWEEP_EQ):
        acc += 1 if loop_update(sigma, tetA, tetB, ta, tb, rng) else 0
    da, db = ice_defects(sigma, tetA, tetB)
    frac = float(np.mean(sigma > 0))
    print(f"equilibrated: accepted {acc}/{NSWEEP_EQ} loops, "
          f"ice defects A={da} B={db}, up-fraction {frac:.4f} "
          f"({time.perf_counter()-t0:.0f} s)", flush=True)
    if da or db:
        print("  GATE FAILED: loop update broke the ice rule -> abort")
        return 1

    qpath, ticks, labels = bz_path()
    phase = np.exp(-2j * np.pi * (qpath @ pos.T))         # (Nq, nsite)
    masks = [sub == mu for mu in range(4)]
    Spath = np.zeros(len(qpath))
    for m in range(NMEAS):  # measurement loop
        for _ in range(NLOOP_PER_MEAS):
            loop_update(sigma, tetA, tetB, ta, tb, rng)
        s = sigma.astype(float) * 0.5
        for mk in masks:
            Spath += np.abs(phase[:, mk] @ s[mk]) ** 2
    Spath /= (NMEAS * nsite)
    print(f"measured path ({time.perf_counter()-t0:.0f} s)", flush=True)

    f1path = f1_of_q(qpath, temps, w)
    with np.errstate(divide="ignore", invalid="ignore"):
        omega = np.where(Spath > 1e-9, f1path / Spath, np.nan)

    # local minima along the path
    mins = []
    for i in range(1, len(omega) - 1):
        if (np.isfinite(omega[i]) and np.isfinite(omega[i - 1])
                and np.isfinite(omega[i + 1])
                and omega[i] < omega[i - 1] and omega[i] <= omega[i + 1]):
            mins.append({"q": [float(x) for x in qpath[i]],
                         "omega_sma": float(omega[i]),
                         "S": float(Spath[i]), "f1": float(f1path[i])})
    mins.sort(key=lambda r: r["omega_sma"])
    print("\nlocal minima of omega_SMA along the path (lowest first):")
    for r in mins[:12]:
        q = r["q"]
        print(f"  q=({q[0]:+.3f},{q[1]:+.3f},{q[2]:+.3f})  "
              f"omega_SMA={r['omega_sma']:.4f}  S={r['S']:.4f}  "
              f"f1={r['f1']:.4f}")

    np.savez_compressed(OUT / "roton_bz.npz", qpath=qpath, Spath=Spath,
                        f1path=f1path, omega=omega, ticks=ticks,
                        labels=np.array(labels, dtype=object))
    rec["minima"] = mins[:20]
    rec["L"] = LBOX
    rec["nmeas"] = NMEAS
    with open(OUT / "roton_bz.json", "w") as fh:
        json.dump(rec, fh, indent=1, default=float)
    print(f"\nwrote roton_bz.npz / roton_bz.json "
          f"({time.perf_counter()-t0:.0f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
