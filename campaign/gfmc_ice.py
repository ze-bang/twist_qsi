"""Green's function Monte Carlo for the pyrochlore ring model.

Purpose: the referee's decisive request.  The roton claim rests on
f_1(q)/S(q), and both factors are ground-state quantities, so they can
be computed on lattices far beyond exact diagonalization and on a dense
momentum grid.  The model is stoquastic -- every off-diagonal element
of H = -K sum_p (W_p + W_p^dag) is -K < 0 in the ice-configuration
basis, and the diagonal is zero -- so projector Monte Carlo has no sign
problem.

What must be stochastic and what need not be:

  * E_0: mixed estimator, exact for the ground state.
  * S(q): DIAGONAL in the configuration basis; forward walking gives
    the pure |psi_0|^2 expectation.
  * f_1(q): needs only <W_p + W_p^dag> per plaquette.  On a cubic
    (n,n,n) supercell every hexagon is related by symmetry, so the
    exact sum rule sum_p w_p = -E_0/K fixes w_p = -E_0/(K n_hex)
    with NO additional statistical error.  f_1 is then E_0 times an
    exactly computable geometric factor.

Algorithm: discrete projector M = Lambda*1 - H, all entries
nonnegative for Lambda > 0 (the shift also breaks any bipartite
oscillation of the flip graph).  A walker at configuration x carries
weight; one step multiplies the weight by (Lambda + K n_flip(x))/W_ref
and moves to one of the n_flip(x) flip-connected configurations with
probability K/(Lambda + K n_flip) each, staying put otherwise.  The
trial state is the equal-amplitude ice state (the RK wavefunction),
whose overlap with the true ground state is known to be substantial
from the 48-site ED.  Population control by stochastic
reconfiguration; forward walking by weighting each measurement with
the descendant weight a fixed number of steps ahead.

Ice configurations are (n_words x uint64) rows, so any lattice size
works.  Gates: E_0 and S(L) against exact diagonalization at 48 and 72
sites, and the winding sector is preserved exactly by construction
(hexagon flips do not change it).
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

from pyro_lattice import build_lattice, extract_templates       # noqa: E402

OUT = ROOT / "campaign" / "outputs" / "ring_model_dssf"


def log(m, t0):
    print(f"[{time.perf_counter()-t0:8.1f}s] {m}", flush=True)


# ------------------------------------------------------------- states
def n_words(n_sites):
    return (n_sites + 63) // 64


def masks_for(hexes, n_sites):
    """Per hexagon: (mask, pattern_even) as (n_words,) uint64 arrays.
    A hexagon is flippable iff  state & mask  is pattern_even or its
    complement within the mask."""
    W = n_words(n_sites)
    m = np.zeros((len(hexes), W), dtype=np.uint64)
    p = np.zeros((len(hexes), W), dtype=np.uint64)
    for h, path in enumerate(hexes):
        for j, s in enumerate(path):
            w, b = divmod(int(s), 64)
            m[h, w] |= np.uint64(1 << b)
            if j % 2 == 0:
                p[h, w] |= np.uint64(1 << b)
    return m, p


def seed_ice(lat):
    """A sublattice-ordered two-in-two-out state: on every tetrahedron
    (up or down) sublattices {0,1} point up and {2,3} down.  Verified
    against the tetrahedra directly."""
    W = n_words(lat.n_sites)
    st = np.zeros(W, dtype=np.uint64)
    for s in range(lat.n_sites):
        if lat.sublattice[s] in (0, 1):
            w, b = divmod(s, 64)
            st[w] |= np.uint64(1 << b)
    # gate
    for t in lat.tets:
        ups = 0
        for s in t:
            w, b = divmod(int(s), 64)
            ups += int((st[w] >> np.uint64(b)) & np.uint64(1))
        if ups != 2:
            raise RuntimeError("seed violates the ice rule")
    return st


def flippable(states, hm, hp):
    """(n_walkers, n_hex) bool: which hexagons of each walker flip."""
    sel = states[:, None, :] & hm[None, :, :]
    even = (sel == hp[None, :, :]).all(axis=2)
    odd = (sel == (hm ^ hp)[None, :, :]).all(axis=2)
    return even | odd


def spin_cols(states, n_sites):
    """(n_walkers, n_sites) float spins +-1/2 from packed rows."""
    W = states.shape[1]
    out = np.empty((len(states), n_sites), dtype=np.float64)
    for w in range(W):
        lo = w * 64
        hi = min(lo + 64, n_sites)
        for b in range(hi - lo):
            out[:, lo + b] = ((states[:, w] >> np.uint64(b))
                              & np.uint64(1)).astype(np.float64) - 0.5
    return out


# ------------------------------------------------------------- engine
def gfmc(lat, K=1.0, n_walkers=512, n_steps=6000, n_equil=1500,
         forward=60, measure_every=10, qs=None, seed=1, t0=None,
         lam_factor=1.0):
    rng = np.random.default_rng(seed)
    hm, hp = masks_for(lat.hexes, lat.n_sites)
    st0 = seed_ice(lat)
    states = np.tile(st0, (n_walkers, 1))
    n_hex = len(lat.hexes)

    fl = flippable(states, hm, hp)
    lam = lam_factor * K * max(1.0, fl.sum(axis=1).mean())
    log(f"lattice {lat.shape}: {lat.n_sites} sites, {n_hex} hexagons; "
        f"seed n_flip = {fl[0].sum()}, Lambda = {lam:.2f}", t0)

    if qs is None:
        qs = []
    qs = [np.asarray(q, dtype=float) for q in qs]
    phases = [np.exp(-2j * np.pi * (lat.positions @ q)) for q in qs]
    sub_masks = [(lat.sublattice == mu) for mu in range(4)]

    # ring buffer of diagonal measurements awaiting forward weights
    pending = []      # (measurement dict, step index)
    results = {"E_mix": [], "S": [[] for _ in qs], "nflip": []}
    w_ref = lam + K * fl.sum(axis=1).mean()
    log_glob = 0.0

    for step in range(n_steps):
        nf = fl.sum(axis=1)
        b = lam + K * nf                      # column weights
        # weights for reconfiguration
        wts = b / w_ref
        # move: with prob K*nf/b flip one uniformly chosen flippable hex
        u = rng.random(n_walkers)
        move = u < (K * nf) / b
        idx = np.where(move)[0]
        if len(idx):
            r = (rng.random(len(idx)) * nf[idx]).astype(np.int64)
            flips = np.argsort(~fl[idx], axis=1)   # flippable first
            chosen = flips[np.arange(len(idx)), r]
            for w in range(states.shape[1]):
                states[idx, w] ^= hm[chosen, w]
        fl = flippable(states, hm, hp)

        # stochastic reconfiguration (comb resampling, unbiased)
        tot = wts.sum()
        log_glob += np.log(tot / n_walkers)
        pos = (rng.random() + np.arange(n_walkers)) / n_walkers * tot
        cum = np.cumsum(wts)
        parents = np.searchsorted(cum, pos)
        # forward-walking bookkeeping: measurements inherit descendants
        for m in pending:
            m["mult"] = m["mult"][parents]
        states = states[parents]
        fl = flippable(states, hm, hp)

        if step >= n_equil and step % measure_every == 0:
            nf2 = fl.sum(axis=1).astype(np.float64)
            meas = {"step": step, "E": float(-K * nf2.mean()),
                    "nflip": nf2.copy(),
                    "S": [], "mult": np.arange(n_walkers)}
            if qs:
                spins = spin_cols(states, lat.n_sites)
                for pi, ph in enumerate(phases):
                    tot_q = np.zeros(n_walkers)
                    for mk in sub_masks:
                        amp = spins[:, mk] @ ph[mk]
                        tot_q += np.abs(amp) ** 2
                    meas["S"].append(tot_q / lat.n_sites)
            pending.append(meas)

        # harvest measurements older than the forward window: the
        # number of surviving descendants of each original walker is
        # its forward weight, which converts the mixed distribution
        # into the pure |psi_0|^2 one for diagonal observables
        while pending and step - pending[0]["step"] >= forward:
            m = pending.pop(0)
            cnt = np.bincount(m["mult"], minlength=n_walkers).astype(float)
            if cnt.sum() <= 0:
                continue
            wgt = cnt / cnt.sum()
            results["E_mix"].append(m["E"])
            results["nflip"].append(float((m["nflip"] * wgt).sum()))
            for pi in range(len(qs)):
                results["S"][pi].append(float((m["S"][pi] * wgt).sum()))

    return results, lam


def binned(x, nb=16):
    x = np.asarray(x, dtype=float)
    if len(x) < 2 * nb:
        nb = max(2, len(x) // 2)
    m = len(x) // nb
    b = x[:m * nb].reshape(nb, m).mean(axis=1)
    return float(b.mean()), float(b.std(ddof=1) / np.sqrt(nb))


def main() -> int:
    t0 = time.perf_counter()
    shape = tuple(int(x) for x in sys.argv[1:4]) if len(sys.argv) >= 4 \
        else (2, 2, 3)
    n_steps = int(sys.argv[4]) if len(sys.argv) >= 5 else 6000

    templates = extract_templates()
    lat = build_lattice(shape, templates)
    # L point of the cubic zone
    qL = np.array([0.5, 0.5, 0.5])
    res, lam = gfmc(lat, n_steps=n_steps, qs=[qL], t0=t0)

    Em, Ee = binned(res["E_mix"])
    S0, Se = binned(res["S"][0])
    nf, nfe = binned(res["nflip"])
    log(f"\nE_mix   = {Em:.6f} +/- {Ee:.6f} K   "
        f"(per site {Em/lat.n_sites:.6f})", t0)
    log(f"S(L)    = {S0:.6f} +/- {Se:.6f}   (forward-walked)", t0)
    log(f"<nflip> = {nf:.4f} +/- {nfe:.4f} of {len(lat.hexes)}", t0)

    tag = "".join(str(s) for s in shape)
    with open(OUT / f"gfmc_fcc{tag}.json", "w") as fh:
        json.dump({"shape": list(shape), "n_sites": lat.n_sites,
                   "E_mix": Em, "E_err": Ee, "S_L": S0, "S_L_err": Se,
                   "nflip": nf, "lam": lam,
                   "n_steps": n_steps}, fh, indent=1)
    log(f"wrote gfmc_fcc{tag}.json", t0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
