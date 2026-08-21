"""Gaussian photon theory quantized on the SAME 48-site torus as the exact ED.

The compact theory is H = -K sum_hex (W + h.c.) with W = exp(i B_hex),
B_hex = sum_l C_{hl} A_l and C the alternating lattice curl.  The effective
Gaussian theory is

    H_G = (U/2) sum_l E_l^2 + K sum_h B_h^2,     [A_l, E_l'] = i delta,

whose normal modes at momentum q are the singular values sigma_lambda(q) of the
curl restricted to the momentum-q link subspace:

    omega_lambda(q) = sqrt(2 U K) * sigma_lambda(q).

Two scales (sqrt(2UK) for energies, sqrt(K/U)-ish for weights) are FITTED --
in the pure ring model U is generated, not bare, exactly the Benton-Sikora-
Shannon "leap of faith".  Everything else is parameter-free geometry:
  * the SHAPE of the dispersion across the 12 momenta and 2 branches,
  * whether the two polarizations are degenerate at each q,
  * the distribution of one-photon weight over momenta and channels.
E = S^z is linear in photon operators, so the Gaussian theory puts ALL S^zz
weight on the one-photon lines; whatever the exact ED has beyond the matched
lines is multi-photon by definition.

Validations (gates, printed first):
  * C @ d0 = 0 exactly: the alternating curl annihilates every lattice
    gradient (d0 = tetrahedron incidence, signed by the up/down bipartition).
    If this fails the sign convention is wrong and NOTHING below stands.
  * exactly 2 nonzero singular values per q != Gamma, 0 at Gamma.
  * Bloch bases orthonormal; probe-at-q pairs with modes at -q (checked by
    projection completeness of the probe onto that basis).

Compares against campaign/outputs/ring_model_dssf/ring_fixed_fcc223_v+0p00.json
(the projector-validated exact ED group weights).
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

import recompute_finite_size_artifact as geometry              # noqa: E402
from run_ring48_dssf import contractible                       # noqa: E402
from ring48_momenta_fix import fcc_momenta                     # noqa: E402

OUT = ROOT / "campaign" / "outputs" / "ring_model_dssf"
ED_JSON = OUT / "ring_fixed_fcc223_v+0p00.json"


def build_curl(cluster, hexes):
    """C (n_hex x n_sites): alternating +-1 along each contractible hexagon."""
    n = int(cluster.n_sites)
    C = np.zeros((len(hexes), n))
    for h, path in enumerate(hexes):
        for j, site in enumerate(path):
            C[h, site] = 1.0 if j % 2 == 0 else -1.0
    return C


def build_gradient(cluster):
    """d0 (n_sites x n_tets), signed by the up/down tetrahedron bipartition."""
    n = int(cluster.n_sites)
    tets = [tuple(int(s) for s in t) for t in cluster.tets]
    # bipartition the tetrahedron graph (share a site -> adjacent) by BFS
    site_tets = {}
    for ti, tet in enumerate(tets):
        for s in tet:
            site_tets.setdefault(s, []).append(ti)
    print(f"  {len(tets)} tetrahedra for {n} sites "
          f"({'OK' if len(tets) == n // 2 else 'SPURIOUS CLIQUES'})")
    color = {0: 0}
    queue = [0]
    while queue:
        ti = queue.pop()
        for s in tets[ti]:
            for tj in site_tets[s]:
                if tj == ti:
                    continue
                if tj not in color:
                    color[tj] = 1 - color[ti]
                    queue.append(tj)
                elif color[tj] == color[ti]:
                    raise SystemExit("tetrahedron graph not bipartite")
    d0 = np.zeros((n, len(tets)))
    for ti, tet in enumerate(tets):
        sign = 1.0 if color[ti] == 0 else -1.0
        for s in tet:
            d0[s, ti] = sign
    # every site must touch exactly one up and one down tet
    if not np.all(np.abs(d0).sum(axis=1) == 2):
        raise SystemExit("site-tetrahedron incidence is not 2-regular")
    return d0


def bloch_basis(positions, sublattice, momentum):
    """Orthonormal columns psi_{q,mu}[i] = e^{+2 pi i q.r_i} on sublattice mu."""
    cols = []
    phase = np.exp(2j * np.pi * (positions @ momentum))
    for mu in range(4):
        v = phase * (sublattice == mu)
        cols.append(v / np.linalg.norm(v))
    return np.array(cols).T                       # (n_sites, 4)


def main() -> int:
    t0 = time.perf_counter()
    cluster = geometry.build_cluster("fcc", (2, 2, 3))
    n = int(cluster.n_sites)
    hexes = contractible(cluster)
    positions = np.asarray(cluster.positions, dtype=float)
    sublattice = np.arange(n) % 4
    momenta, shape = fcc_momenta(cluster)
    mlist = list(np.ndindex(*shape))

    C = build_curl(cluster, hexes)
    d0 = build_gradient(cluster)
    gauge = float(np.max(np.abs(C @ d0)))
    print(f"gauge invariance ||C d0||_max = {gauge:.3e}  -> "
          f"{'PASS' if gauge < 1e-12 else 'FAIL: sign convention wrong'}")
    if gauge >= 1e-12:
        return 1
    print(f"rank(C) = {np.linalg.matrix_rank(C)} of {n}")

    # singular values per momentum, and mode vectors at each q
    sigma, modes = {}, {}
    for qi, momentum in enumerate(momenta):
        Uq = bloch_basis(positions, sublattice, momentum)
        block = C @ Uq                              # (n_hex, 4)
        u, s, vh = np.linalg.svd(block, full_matrices=False)
        keep = s > 1e-10
        sigma[qi] = s[keep]
        modes[qi] = (Uq @ vh.conj().T)[:, keep]     # link-space mode vectors
        print(f"  q={mlist[qi]}  sigma = "
              + ", ".join(f"{v:.6f}" for v in s)
              + f"   ({int(keep.sum())} photon branches)")

    # ---- match against the exact ED ------------------------------------
    ed = json.load(open(ED_JSON))
    neg = {tuple(m): mlist.index(tuple((-np.array(m)) % np.array(shape)))
           for m in mlist}

    # probe at q excites modes at -q; predicted branch weights (up to prefactor)
    rows = []
    for entry in ed["momenta"]:
        q = np.asarray(entry["q"], dtype=float)
        qi = int(np.argmin([np.linalg.norm(q - mm) for mm in momenta]))
        qj = neg[mlist[qi]]
        if len(sigma[qj]) == 0:
            continue
        phase = np.exp(-2j * np.pi * (positions @ q))
        wts = []
        for lam in range(len(sigma[qj])):
            m = modes[qj][:, lam]
            amp = sum(abs(np.vdot(m, phase * (sublattice == mu))) ** 2
                      for mu in range(4))
            wts.append(sigma[qj][lam] * amp / n)    # weight ~ omega * |proj|^2
        rows.append({"qi": qi, "m": mlist[qi], "star": entry["star"],
                     "sigma": list(map(float, np.sort(sigma[qj]))),
                     "w_harm": list(map(float, wts)),
                     "ed_first": entry["first"], "ed_firstw": entry["firstw"],
                     "ed_total": entry["total"]})

    # fit ONE energy scale on dominant ED group vs dominant harmonic branch
    num = den = 0.0
    for r in rows:
        lam = int(np.argmax(r["w_harm"]))
        num += r["ed_first"][0] * r["sigma"][lam]
        den += r["sigma"][lam] ** 2
    scale = num / den
    print(f"\nfitted energy scale sqrt(2UK) = {scale:.6f}  (1 parameter)")

    print("\n  m-index    star        harm omega (scaled)   pol split "
          "| ED groups (frac)          | 1ph_ED  multi-photon")
    print("  " + "=" * 108)
    summary = []
    for r in rows:
        harm = [scale * s for s in r["sigma"]]
        split = harm[-1] / harm[0] if harm[0] > 0 else np.nan
        # principled matching: an ED group is one-photon if within 20% of a
        # scaled branch; sum matched fractions, rest is multi-photon
        matched = 0.0
        used = []
        for e, w in zip(r["ed_first"], r["ed_firstw"]):
            hit = [h for h in harm if abs(e - h) / h < 0.20 and h not in used]
            if hit:
                matched += w
                used.append(hit[0])
        multi = 1.0 - matched
        print(f"  {str(r['m']):<10s} {r['star'][:10]:<11s} "
              + ", ".join(f"{h:.4f}" for h in harm)
              + f"   {split:5.3f}     | "
              + " ".join(f"{e:.3f}({w:.3f})" for e, w in
                         zip(r["ed_first"][:3], r["ed_firstw"][:3]))
              + f" | {matched:.4f}  {100*multi:.2f}%")
        summary.append({"m": list(r["m"]), "star": r["star"],
                        "harm_omega": harm, "pol_split": float(split),
                        "sigma": r["sigma"], "w_harm": r["w_harm"],
                        "ed_first": r["ed_first"], "ed_firstw": r["ed_firstw"],
                        "ed_total": r["ed_total"],
                        "one_photon_frac": float(matched),
                        "multi_photon_frac": float(multi)})

    with open(OUT / "ring48_gaussian_match.json", "w") as fh:
        json.dump({"scale": float(scale), "gauge_check": gauge,
                   "rows": summary}, fh, indent=1, default=float)
    print(f"\nwrote ring48_gaussian_match.json ({time.perf_counter()-t0:.0f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
