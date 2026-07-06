"""Exact numerical degenerate perturbation theory (Schrieffer-Wolff) for the
XXZ pyrochlore in the ice manifold, with per-virtual-path winding tracking.

H = Jzz sum_<ij> Sz_i Sz_j - Jpm sum_<ij> (S+_i S-_j + S-_i S+_j)

Working in the 2-in-2-out (ice) manifold, we compute the effective
Hamiltonian at orders 2 and 3,

    H2 = P V G V P,        H3 = P V G Q V G V P,     G = Q (E0 - H0)^-1,

*path-resolved*: every contribution is stored as a row

    (s_ice, t_ice, N, coeff)

where N in Z^3 is the accumulated bond-wrap vector of the virtual path
(the hop S+_a S-_b, i.e. raise a / lower b, contributes n_{ab} =
round((r_b - r_a)/L_frac) in cluster-translation units).  Under a U(1)
boundary twist phi (minimum-image gauge: phases only on boundary-crossing
bonds) the row's coefficient is multiplied by exp(-i N.phi).  This makes
H_eff(phi) available at any twist, in any gauge, from a single PT run:

  * minimum-image gauge, twist phi     : coeff * exp(-i N.phi)
  * 8-corner {0,pi}^3 OPERATOR average : keep rows with N == 0 (mod 2)
  * continuum OPERATOR average, min-image gauge : keep rows with N == 0
  * continuum OPERATOR average, smooth gauge (vector potential A = phi/L
    on every bond) == zero-net-transport projector: keep rows with
    delta := rho(t,s) + sum_k N_k Lvec_k == 0, where rho = sum_raised r
    - sum_lowered r (home coordinates).  delta is the gauge-invariant net
    charge-transport displacement of the process.

Energies are in units of Jzz (set Jzz=1); positions in units of the cubic
lattice constant a (quarter-integer coordinates -> everything integer
after multiplying by 4).
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cluster_geometry_audit as cg  # noqa: E402


# ----------------------------------------------------------------------------
# Cluster container
# ----------------------------------------------------------------------------
@dataclass
class Cluster:
    basis: str
    shape: tuple
    n_sites: int
    positions: np.ndarray          # (N,3) float, home coordinates
    bonds: np.ndarray              # (Nb,2) int
    bond_wrap: np.ndarray          # (Nb,3) int, n_ij for bond (i,j)
    tets: list                     # list of 4-tuples
    tet_masks: np.ndarray          # (Nt,) uint64
    Lvecs: np.ndarray              # (3,3) cluster translation vectors (rows)
    loops4: list = field(default_factory=list)   # (path, w)
    hexes: list = field(default_factory=list)    # (path, w)
    ice_states: np.ndarray = None  # sorted uint64
    ice_index: dict = None         # state -> idx

    @property
    def n_ice(self):
        return len(self.ice_states)


def build_cluster(basis: str, shape) -> Cluster:
    if basis == "cubic":
        vertices, edges, _t, bond_wrap, adj = cg.build_graph(*shape)
    elif basis == "fcc":
        vertices, edges, _t, bond_wrap, adj = cg.build_graph_fcc(*shape)
    else:
        raise ValueError(basis)
    n = len(vertices)
    tets = cg._enumerate_4_cliques(adj, n)
    site_tets = cg.site_to_tetrahedra(tets)
    pbc_vecs = cg.cluster_pbc_vectors(*shape, basis)

    bonds = np.array(sorted((min(u, v), max(u, v)) for u, v in edges), dtype=np.int64)
    wraps = np.array([bond_wrap[(int(u), int(v))] for u, v in bonds], dtype=np.int64)
    pos = np.array([vertices[i] for i in range(n)], dtype=float)
    tet_masks = np.array(
        [np.bitwise_or.reduce([np.uint64(1) << np.uint64(s) for s in t]) for t in tets],
        dtype=np.uint64,
    )

    cl = Cluster(
        basis=basis, shape=tuple(shape), n_sites=n, positions=pos,
        bonds=bonds, bond_wrap=wraps, tets=tets, tet_masks=tet_masks,
        Lvecs=np.array(pbc_vecs, dtype=float),
    )

    cycles4 = cg.enumerate_simple_cycles(adj, 4)
    cl.loops4 = [
        (s, w) for s, w in cycles4 if cg.is_ice_preserving(s, site_tets, tets)
    ]
    cycles6 = cg.enumerate_simple_cycles(adj, 6)
    cl.hexes = [
        (s, w) for s, w in cycles6
        if cg.is_ice_preserving(s, site_tets, tets)
        and cg.is_111_planar_with_wrap(s, vertices, bond_wrap, pbc_vecs)
    ]

    cl.ice_states = enumerate_ice(cl)
    cl.ice_index = {int(s): i for i, s in enumerate(cl.ice_states)}
    return cl


def enumerate_ice(cl: Cluster) -> np.ndarray:
    """All 2-in-2-out states, by backtracking over sites with per-tet counts."""
    n = cl.n_sites
    tets_of_site = [[] for _ in range(n)]
    for t_id, t in enumerate(cl.tets):
        for s in t:
            tets_of_site[s].append(t_id)
    n_t = len(cl.tets)
    tet_sizes = np.zeros(n_t, dtype=int)   # sites assigned so far
    tet_ups = np.zeros(n_t, dtype=int)
    out = []

    def rec(site, state):
        if site == n:
            out.append(state)
            return
        for spin in (0, 1):
            ok = True
            for t in tets_of_site[site]:
                ups = tet_ups[t] + spin
                downs = (tet_sizes[t] + 1) - ups
                if ups > 2 or downs > 2:
                    ok = False
                    break
            if not ok:
                continue
            for t in tets_of_site[site]:
                tet_sizes[t] += 1
                tet_ups[t] += spin
            rec(site + 1, state | (spin << site))
            for t in tets_of_site[site]:
                tet_sizes[t] -= 1
                tet_ups[t] -= spin
    rec(0, 0)
    return np.array(sorted(out), dtype=np.uint64)


# ----------------------------------------------------------------------------
# Ising energies (Jzz = 1), relative to the ice manifold (E_ice = 0)
# ----------------------------------------------------------------------------
def ising_energy(cl: Cluster, states: np.ndarray) -> np.ndarray:
    """E = 0.5 * sum_t (n_up_t - 2)^2 ; a 3:1 tetrahedron costs 1/2."""
    st = states[:, None] & cl.tet_masks[None, :]
    ups = np.bitwise_count(st).astype(np.int64)
    return 0.5 * ((ups - 2) ** 2).sum(axis=1).astype(float)


# ----------------------------------------------------------------------------
# One application of V = -Jpm sum (S+S- + S-S+), path-resolved
# ----------------------------------------------------------------------------
def apply_V(cl: Cluster, states, cols, Ns, amps, Jpm):
    """Apply V to a batch of (state, col, N, amp) rows; returns the new batch.

    Hop that raises a and lowers b contributes N += n_{ab}; for stored bond
    (i, j) with wrap n_ij (= round((r_j - r_i)/L)):
       S+_i S-_j (raise i, lower j): N += n_ij,  amp *= -Jpm
       S-_i S+_j (raise j, lower i): N += -n_ij, amp *= -Jpm
    """
    out_s, out_c, out_N, out_a = [], [], [], []
    one = np.uint64(1)
    for (i, j), nij in zip(cl.bonds, cl.bond_wrap):
        bi, bj = one << np.uint64(i), one << np.uint64(j)
        flip = np.uint64(bi | bj)
        up_i = (states & bi) != 0
        up_j = (states & bj) != 0
        # S+_i S-_j : i down, j up
        m = (~up_i) & up_j
        if m.any():
            out_s.append(states[m] ^ flip)
            out_c.append(cols[m])
            out_N.append(Ns[m] + nij)
            out_a.append(amps[m] * (-Jpm))
        # S-_i S+_j : i up, j down
        m = up_i & (~up_j)
        if m.any():
            out_s.append(states[m] ^ flip)
            out_c.append(cols[m])
            out_N.append(Ns[m] - nij)
            out_a.append(amps[m] * (-Jpm))
    if not out_s:
        return (np.empty(0, np.uint64), np.empty(0, np.int64),
                np.empty((0, 3), np.int64), np.empty(0, float))
    return (np.concatenate(out_s), np.concatenate(out_c),
            np.concatenate(out_N), np.concatenate(out_a))


def _dedupe(states, cols, Ns, amps, t_idx):
    """Combine identical (col, t_idx, N) rows (states already mapped to ice)."""
    key = np.stack([cols, t_idx, Ns[:, 0], Ns[:, 1], Ns[:, 2]], axis=1)
    uniq, inv = np.unique(key, axis=0, return_inverse=True)
    summed = np.zeros(len(uniq), dtype=float)
    np.add.at(summed, inv, amps)
    keep = np.abs(summed) > 1e-14
    uniq, summed = uniq[keep], summed[keep]
    return dict(
        s=uniq[:, 0].astype(np.int64), t=uniq[:, 1].astype(np.int64),
        N=uniq[:, 2:5].astype(np.int64), c=summed,
    )


# ----------------------------------------------------------------------------
# The PT run
# ----------------------------------------------------------------------------
def _merge(parts):
    if not parts:
        return dict(s=np.empty(0, np.int64), t=np.empty(0, np.int64),
                    N=np.empty((0, 3), np.int64), c=np.empty(0, float))
    s = np.concatenate([p["s"] for p in parts])
    t = np.concatenate([p["t"] for p in parts])
    N = np.concatenate([p["N"] for p in parts])
    c = np.concatenate([p["c"] for p in parts])
    key = np.stack([s, t, N[:, 0], N[:, 1], N[:, 2]], axis=1)
    uniq, inv = np.unique(key, axis=0, return_inverse=True)
    summed = np.zeros(len(uniq), dtype=float)
    np.add.at(summed, inv, c)
    keep = np.abs(summed) > 1e-14
    uniq, summed = uniq[keep], summed[keep]
    return dict(s=uniq[:, 0], t=uniq[:, 1], N=uniq[:, 2:5], c=summed)


def _rows_multiply(A, B):
    """Operator product A @ B of two ice-manifold row tables, with winding
    accumulation (phases multiply => N add): (AB)[t,s] = sum_m A[t,m] B[m,s].
    """
    order = np.argsort(A["s"], kind="stable")
    As, At, AN, Ac = A["s"][order], A["t"][order], A["N"][order], A["c"][order]
    parts = []
    # group B rows by target m, match to A rows with source m
    for i in range(len(B["c"])):
        m = B["t"][i]
        lo = np.searchsorted(As, m, side="left")
        hi = np.searchsorted(As, m, side="right")
        if lo == hi:
            continue
        k = hi - lo
        parts.append(dict(
            s=np.full(k, B["s"][i], dtype=np.int64),
            t=At[lo:hi],
            N=AN[lo:hi] + B["N"][i],
            c=Ac[lo:hi] * B["c"][i],
        ))
    return _merge(parts)


def _rows_scale(A, fac):
    return dict(s=A["s"], t=A["t"], N=A["N"], c=A["c"] * fac)


def _rows_add(*tabs):
    return _merge(list(tabs))


def _rows_dagger(A):
    """Hermitian conjugate: swap s<->t, negate N (coefficients are real)."""
    return dict(s=A["t"].copy(), t=A["s"].copy(), N=-A["N"], c=A["c"].copy())


def sw_effective(cl: Cluster, Jpm: float, order: int = 3, verbose=False):
    """Return dict of row tables per order: 'H2', 'H3', 'H4'.

    H2 = P V G V P
    H3 = P V G V G V P                                (PVP = 0)
    H4 = P V G V G V G V P - 1/2 {P V G^2 V P, H2}    (PVP = 0)

    Row tables: dict(s=..., t=..., N=(rows,3), c=...) with coefficients at
    phi=0 given by sum over rows (phases exp(-i N.phi) at twist phi).
    H2 ~ Jpm^2, H3 ~ Jpm^3, H4 ~ Jpm^4 exactly (Ising resolvents), so a
    single run at Jpm=1 provides every coupling.
    """
    ice = cl.ice_states
    n_ice = len(ice)
    out = {}
    H2_parts, H3_parts, H2gg_parts, X_parts = [], [], [], []
    ice_sorted = ice  # sorted

    for s0 in range(n_ice):
        st0 = np.array([ice[s0]], dtype=np.uint64)
        c0 = np.array([s0], dtype=np.int64)
        N0 = np.zeros((1, 3), dtype=np.int64)
        a0 = np.ones(1, dtype=float)

        s1, c1, N1, a1 = apply_V(cl, st0, c0, N0, a0, Jpm)
        E1 = ising_energy(cl, s1)
        a1g = a1 / (0.0 - E1)          # G: all E1 > 0 (V leaves the manifold)

        # ---- order 2: project back onto ice
        s2, c2, N2, a2 = apply_V(cl, s1, c1, N1, a1g, Jpm)
        E2 = ising_energy(cl, s2)
        in_ice = E2 == 0.0
        if in_ice.any():
            tt = np.searchsorted(ice_sorted, s2[in_ice])
            H2_parts.append(_dedupe(s2[in_ice], c2[in_ice], N2[in_ice],
                                    a2[in_ice], tt))

        if order >= 4:
            # P V G^2 V P : same chain, denominator squared
            s2b, c2b, N2b, a2b = apply_V(cl, s1, c1, N1, a1g / (0.0 - E1), Jpm)
            E2b = ising_energy(cl, s2b)
            m = E2b == 0.0
            if m.any():
                tt = np.searchsorted(ice_sorted, s2b[m])
                H2gg_parts.append(_dedupe(s2b[m], c2b[m], N2b[m], a2b[m], tt))

        if order >= 3:
            # middle Q projector: drop ice components, divide by (0 - E2)
            mid = ~in_ice
            s2q, c2q, N2q = s2[mid], c2[mid], N2[mid]
            a2q = a2[mid] / (0.0 - E2[mid])
            s3, c3, N3, a3 = apply_V(cl, s2q, c2q, N2q, a2q, Jpm)
            E3 = ising_energy(cl, s3)
            in3 = E3 == 0.0
            if in3.any():
                tt = np.searchsorted(ice_sorted, s3[in3])
                H3_parts.append(_dedupe(s3[in3], c3[in3], N3[in3],
                                        a3[in3], tt))

        if order >= 4:
            # store X = G V G V P (per column s0): rows (m_state, N, amp)
            # meet-in-middle: PVGVGVGVP = (VGVP)^dag G (VGVP) = X^dag_{denom} X
            # here keep the *two-hop* wavefront with ONE trailing G applied:
            #   X rows: state m (non-ice), amp = [G V G V P]_{m,s0}
            mid = ~in_ice
            X_parts.append(dict(
                col=np.full(int(mid.sum()), s0, dtype=np.int64),
                m=s2[mid].copy(), N=N2[mid].copy(),
                a=(a2[mid] / (0.0 - E2[mid])).copy(),
                E=E2[mid].copy(),
            ))
        if verbose and (s0 % 200 == 0):
            print(f"  col {s0}/{n_ice}", flush=True)

    out["H2"] = _merge(H2_parts)
    if order >= 3:
        out["H3"] = _merge(H3_parts)
    if order >= 4:
        # ---- P V G V G V G V P via meet-in-middle over intermediate m
        col = np.concatenate([p["col"] for p in X_parts])
        mst = np.concatenate([p["m"] for p in X_parts])
        NN = np.concatenate([p["N"] for p in X_parts])
        aa = np.concatenate([p["a"] for p in X_parts])
        EE = np.concatenate([p["E"] for p in X_parts])
        # pre-dedupe the wavefront by (m, col, N): cuts the pair expansion
        key = np.stack([mst.astype(np.int64), col,
                        NN[:, 0], NN[:, 1], NN[:, 2]], axis=1)
        uniq, inv = np.unique(key, axis=0, return_inverse=True)
        asum = np.zeros(len(uniq))
        np.add.at(asum, inv, aa)
        Esum = np.zeros(len(uniq))
        Esum[inv] = EE
        mst = uniq[:, 0]
        col = uniq[:, 1]
        NN = uniq[:, 2:5]
        aa = asum
        EE = Esum
        # amp already includes one G; the sandwich needs a single central G:
        #   H4a[t,s] = sum_m [VGVP]*_{m,t} G_m [VGVP]_{m,s}
        #            = sum_m X*[m,t] (0-E_m) X[m,s].
        order_m = np.argsort(mst, kind="stable")
        mst, col, NN, aa, EE = (mst[order_m], col[order_m], NN[order_m],
                                aa[order_m], EE[order_m])
        bounds = np.flatnonzero(np.diff(mst)) + 1
        starts = np.concatenate([[0], bounds])
        stops = np.concatenate([bounds, [len(mst)]])
        parts4, acc_rows, merged = [], 0, []
        for lo, hi in zip(starts, stops):
            k = hi - lo
            if k == 0:
                continue
            g_m = (0.0 - EE[lo])
            cs = col[lo:hi]
            Ns = NN[lo:hi]
            avs = aa[lo:hi]
            # outer product: bra (conjugate: N -> -N) x ket
            tt = np.repeat(cs, k)
            ss = np.tile(cs, k)
            Nrows = (np.tile(Ns, (k, 1)) - np.repeat(Ns, k, axis=0))
            crows = np.tile(avs, k) * np.repeat(avs, k) * g_m
            parts4.append(dict(s=ss, t=tt, N=Nrows, c=crows))
            acc_rows += k * k
            if acc_rows > 15_000_000:      # chunked merge to bound memory
                merged.append(_merge(parts4))
                parts4, acc_rows = [], 0
        if parts4:
            merged.append(_merge(parts4))
        H4a = _merge(merged)
        # ---- -1/2 { PVG^2VP , H2 }
        W = _merge(H2gg_parts)
        WH = _rows_multiply(W, out["H2"])
        HW = _rows_multiply(out["H2"], W)
        out["H4"] = _rows_add(H4a, _rows_scale(WH, -0.5), _rows_scale(HW, -0.5))
    return out


# ----------------------------------------------------------------------------
# Assembling H_eff matrices from row tables
# ----------------------------------------------------------------------------
def rows_to_matrix(cl: Cluster, rows, phi=None, select=None):
    """Dense (n_ice, n_ice) complex matrix.

    phi    : twist (3,) in the minimum-image gauge -> phase exp(-i N.phi)
    select : optional boolean mask over rows (applied before phases)
    """
    n = cl.n_ice
    H = np.zeros((n, n), dtype=complex)
    s, t, N, c = rows["s"], rows["t"], rows["N"], rows["c"]
    if select is not None:
        s, t, N, c = s[select], t[select], N[select], c[select]
    if phi is None:
        vals = c.astype(complex)
    else:
        vals = c * np.exp(-1j * (N @ np.asarray(phi, dtype=float)))
    np.add.at(H, (t, s), vals)
    return H


def transport_delta(cl: Cluster, rows):
    """Gauge-invariant net transport delta = rho(t,s) + N.Lvecs of each row,
    in units of a/4 (integer array, shape (rows, 3))."""
    pos4 = np.round(cl.positions * 4).astype(np.int64)          # (n,3)
    ice = cl.ice_states
    s_state = ice[rows["s"]]
    t_state = ice[rows["t"]]
    raised = t_state & ~s_state
    lowered = s_state & ~t_state
    site_bits = np.uint64(1) << np.arange(cl.n_sites, dtype=np.uint64)
    r_mask = (raised[:, None] & site_bits[None, :]) != 0
    l_mask = (lowered[:, None] & site_bits[None, :]) != 0
    rho4 = r_mask.astype(np.int64) @ pos4 - l_mask.astype(np.int64) @ pos4
    L4 = np.round(cl.Lvecs * 4).astype(np.int64)                # (3,3) rows
    return rho4 + rows["N"] @ L4


def select_rows(cl: Cluster, rows, mode: str):
    """Boolean masks implementing the operator-level averages.

    'all'        : every row (bare H_eff at phi=0)
    'corner_avg' : 8-corner {0,pi}^3 average, min-image gauge (N even)
    'n0'         : continuum average, min-image gauge (N == 0)
    'delta0'     : continuum average, smooth gauge == zero-net-transport
    """
    if mode == "all":
        return np.ones(len(rows["c"]), dtype=bool)
    if mode == "corner_avg":
        return (rows["N"] % 2 == 0).all(axis=1)
    if mode == "n0":
        return (rows["N"] == 0).all(axis=1)
    if mode == "delta0":
        return (transport_delta(cl, rows) == 0).all(axis=1)
    raise ValueError(mode)


# ----------------------------------------------------------------------------
# Thermodynamics from a spectrum
# ----------------------------------------------------------------------------
def save_rows(path, pt, cl, order):
    """Persist SW row tables (coupling-independent, built at Jpm=1) plus the
    cluster identity, so the expensive build runs once and is reused for the
    whole coupling sweep and both flux signs."""
    out = {"basis": cl.basis, "shape": np.array(cl.shape), "order": order,
           "n_ice": cl.n_ice}
    for k in ("H2", "H3", "H4"):
        if k in pt:
            out[f"{k}_s"] = pt[k]["s"]
            out[f"{k}_t"] = pt[k]["t"]
            out[f"{k}_N"] = pt[k]["N"]
            out[f"{k}_c"] = pt[k]["c"]
    np.savez_compressed(path, **out)


def load_rows(path):
    """Return (pt, meta) from a save_rows archive."""
    d = np.load(path, allow_pickle=False)
    pt = {}
    for k in ("H2", "H3", "H4"):
        if f"{k}_s" in d.files:
            pt[k] = dict(s=d[f"{k}_s"], t=d[f"{k}_t"],
                         N=d[f"{k}_N"], c=d[f"{k}_c"])
    meta = dict(basis=str(d["basis"]), shape=tuple(int(x) for x in d["shape"]),
                order=int(d["order"]), n_ice=int(d["n_ice"]))
    return pt, meta


def assemble(cl, pt, Jpm, order, mode="all"):
    """Assemble the effective Hamiltonian matrix at coupling Jpm from the
    Jpm=1 row tables, with optional projection mode ('all' = plain PBC,
    'delta0' = zero-transport projector). Returns a Hermitian (n_ice,n_ice)
    complex array."""
    M = np.zeros((cl.n_ice, cl.n_ice), dtype=complex)
    powers = {"H2": 2, "H3": 3, "H4": 4}
    for k in ("H2", "H3", "H4"):
        if k not in pt or powers[k] > order:
            continue
        sel = select_rows(cl, pt[k], mode) if mode != "all" else None
        M += (Jpm ** powers[k]) * rows_to_matrix(cl, pt[k], select=sel)
    return M


def C_of_T(E, T):
    E = np.asarray(E, dtype=float)
    E = E - E.min()
    T = np.atleast_1d(T)
    beta = 1.0 / T[:, None]
    w = np.exp(-beta * E[None, :])
    Z = w.sum(axis=1)
    Em = (w * E).sum(axis=1) / Z
    E2 = (w * E ** 2).sum(axis=1) / Z
    return (E2 - Em ** 2) / T ** 2


def peak_T(E, tmin=1e-4, tmax=0.2, n=1200):
    T = np.geomspace(tmin, tmax, n)
    C = C_of_T(E, T)
    return T[np.argmax(C)], C.max()


def manifold_gap(E, rtol=8.0, atol=1e-10):
    """Gap between ground manifold and first excited manifold, detecting the
    manifold edge as the first spacing much larger than the running scale."""
    E = np.sort(np.asarray(E, dtype=float))
    d = np.diff(E)
    scale = max(np.median(d[d > atol]) if (d > atol).any() else atol, atol)
    for k, dk in enumerate(d):
        if dk > max(rtol * scale, 20 * atol) and dk > 1e-6:
            return dk, k + 1     # gap, ground-manifold degeneracy
    return 0.0, len(E)
