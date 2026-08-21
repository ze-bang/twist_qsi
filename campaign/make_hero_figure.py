"""HERO FIGURE (v4).

Left: energy vs momentum, EIGHT columns (five from the 48-site cluster,
three from the 64-site cluster, dagger-marked), sorted by |q|.  The two
inequivalent L classes appear side by side: one rotonizes, one does not
-- the mechanism's prediction, visible directly.  Free-photon curve
interpolated from the analytic dispersion along the same path.

Right: three LARGE, uncluttered illustrations.
  photon      -- a single kagome row of big plaquettes carrying a slow
                 flux wave
  roton       -- a cropped, enlarged pyrochlore slab; plaquette fill =
                 the exact wave amplitude at L (kagome plaquettes zero,
                 tilted plaquettes alternate)
  multiphoton -- the quartic vertex and the 1 -> 3 process, one anchor
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle, Polygon, RegularPolygon

matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif": ["cmr10", "Computer Modern Roman", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "axes.unicode_minus": False,
})

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "campaign" / "outputs" / "ring_model_dssf"
OUTDIR = ROOT / "paper_multiphoton" / "figs"

INK, INK2, INK3 = "#0f172a", "#475569", "#94a3b8"
PHOTON = "#4f46e5"
ROTON = "#e11d48"
MULTI = "#0d9488"
GAUSS = "#d97706"
EDGE = "#cbd5e1"
BOND = "#c7cdd6"

ROTON_E = 1.0235
WMIN = 5e-3

# 48-site momenta: sorted-|coords| key -> (label, path coordinate t)
M48 = {
    (0.3333, 0.3333, 0.3333):
        (r"$(\frac{1}{3}\frac{1}{3}\frac{1}{3})$", 2 / 3),
    (0.3333, 0.3333, 0.6667):
        (r"$(\frac{1}{3}\frac{1}{3}\frac{2}{3})$", 4 / 3),
    (0.5, 0.5, 0.5): (r"$L$", 1.0),
    (0.1667, 0.1667, 0.8333):
        (r"$(\frac{1}{6}\frac{1}{6}\frac{5}{6})$", 5 / 3),
    (0.0, 0.0, 1.0): (r"$X$", 2.0),
}


def classify48(qrow):
    key = tuple(np.round(np.sort(np.abs(qrow)), 4))
    for ref, val in M48.items():
        if np.allclose(key, ref, atol=2e-3):
            return val
    return None


def free_photon_curve(z, scale):
    """omega_G(t) along Gamma-L-X from the stored analytic dispersion."""
    sig = np.asarray(z["path_sigma"], dtype=float)
    tc = np.concatenate([np.linspace(0, 1, 41),
                         1 + np.linspace(0, 1, 41)[1:]])
    return lambda t: scale * float(np.interp(t, tc, sig))


def columns_48(z):
    ge = np.asarray(z["group_energies"], dtype=float)
    odd = np.asarray(z["odd"], dtype=float)
    even = np.asarray(z["even"], dtype=float)
    qred = np.round(np.asarray(z["qreduced"], dtype=float), 6)
    out, seen = [], set()
    for qi in range(odd.shape[1]):
        so = odd[:, qi].sum()
        if so < 1e-10:
            continue
        hit = classify48(qred[qi])
        if hit is None or hit[0] in seen:
            continue
        seen.add(hit[0])
        lab, t = hit
        qmag = float(np.linalg.norm(qred[qi]))
        lines = [(float(ge[j]), float(odd[j, qi] / so))
                 for j in np.where(odd[:, qi] > WMIN * so)[0]]
        se = even[:, qi].sum()
        elines = [(float(ge[j]), float(even[j, qi] / se))
                  for j in np.where(even[:, qi] > 2 * WMIN * se)[0]] \
            if se > 1e-12 else []
        out.append(dict(lab=lab, t=t, qmag=qmag, lines=lines,
                        elines=elines, cluster=48))
    return out


def columns_64():
    r64 = json.load(open(D / "ring64_results.json"))["fcc224"]["momenta"]
    picked, out = set(), []
    for key in sorted(r64, key=int):
        m = r64[key]
        star = str(m["star"])
        top = [(float(e), float(w)) for e, w in m["top"]]
        sig = (star, round(top[0][0], 3))
        if sig in picked:
            continue
        picked.add(sig)
        if "0.25, 0.25, 0.25]" in star:
            lab, t = r"$(\frac{1}{4}\frac{1}{4}\frac{1}{4})^{\dagger}$", 0.5
        elif "0.25, 0.25, 0.75]" in star:
            lab, t = r"$(\frac{1}{4}\frac{1}{4}\frac{3}{4})^{\dagger}$", 1.5
        elif star == "L" and top[0][0] < 1.5:
            lab, t = r"$L^{\dagger}$", 1.0        # rotonizing class
        elif star == "L":
            lab, t = r"$L'^{\,\dagger}$", 1.0     # non-rotonizing class
        else:
            continue                              # skip duplicates of 48
        out.append(dict(lab=lab, t=t, qmag=float(m["qmag"]), lines=top,
                        elines=[], cluster=64))
    return out


def spectrum_panel(ax, z, scale):
    cols = columns_48(z) + columns_64()
    order = {r"$L$": 0, r"$L^{\dagger}$": 1, r"$L'^{\,\dagger}$": 2}
    cols.sort(key=lambda c: (round(c["qmag"], 3), order.get(c["lab"], 3)))
    xs = np.arange(len(cols))
    TOP = 4.55

    fpc = free_photon_curve(z, scale)
    gline = np.array([fpc(c["t"]) for c in cols])
    ax.plot(xs, gline, "--", color=GAUSS, lw=1.8, zorder=2)

    for x, c in enumerate(cols):
        lines = sorted(c["lines"], key=lambda p: -p[1])
        photons = [p for p in lines[:2] if p[1] > 0.10]
        ec = "white" if c["cluster"] == 48 else INK3
        for e, w in c["lines"]:
            if e > TOP or e < 1e-6:
                continue
            is_roton = ((c["lab"] == r"$L$" and abs(e - ROTON_E) < 5e-3)
                        or (c["lab"] == r"$L^{\dagger}$"
                            and abs(e - 0.9545) < 5e-3))
            if is_roton:
                col, zz = ROTON, 6
            elif (e, w) in photons:
                col, zz = PHOTON, 5
            else:
                col, zz = MULTI, 4
            ax.scatter([x], [e], s=26 + 1350 * w, color=col, alpha=0.9,
                       zorder=zz, edgecolor=ec, lw=0.8)
        for e, w in c["elines"]:
            if e > TOP:
                continue
            ax.scatter([x], [e], s=22 + 330 * w, facecolor="none",
                       edgecolor=INK2, lw=1.15, marker="s", zorder=3,
                       alpha=0.85)

    iL = next(x for x, c in enumerate(cols) if c["lab"] == r"$L$")
    ax.annotate("roton", xy=(iL + 0.12, ROTON_E - 0.03),
                xytext=(iL - 2.05, 0.55), fontsize=11, color=ROTON,
                arrowprops=dict(arrowstyle="->", color=ROTON, lw=1.5))

    ax.set_xticks(xs)
    ax.set_xticklabels([c["lab"] for c in cols], fontsize=8.6)
    ax.set_xlim(-0.45, len(cols) - 0.55)
    ax.set_ylim(0, 5.95)
    ax.set_yticks([0, 1, 2, 3, 4, 5])
    ax.set_ylabel(r"energy $\ \omega/K$", fontsize=11, color=INK)
    ax.set_xlabel("momentum", fontsize=11, color=INK)
    ax.tick_params(labelsize=9.5, colors=INK2, length=3)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(EDGE)

    hc = [plt.Line2D([], [], marker="o", ls="", color=PHOTON, ms=8,
                     label="photon"),
          plt.Line2D([], [], marker="o", ls="", color=ROTON, ms=8,
                     label="roton"),
          plt.Line2D([], [], marker="o", ls="", color=MULTI, ms=8,
                     label="multiphoton"),
          plt.Line2D([], [], ls="--", color=GAUSS, lw=1.8,
                     label="free-photon theory (all weight)"),
          plt.Line2D([], [], marker="o", ls="", color=INK2, ms=8,
                     label="seen by neutrons"),
          plt.Line2D([], [], marker="s", ls="", mfc="none", mec=INK2,
                     ms=8, mew=1.2,
                     label="hidden from neutrons "
                           "$\\to$ Raman, ultrasound")]
    lg = ax.legend(handles=hc, fontsize=8.6, frameon=False, ncol=2,
                   loc="upper left", handletextpad=0.45,
                   columnspacing=1.2, labelspacing=0.4,
                   bbox_to_anchor=(-0.01, 1.005))
    for t in lg.get_texts():
        t.set_color(INK)
    ax.text(len(cols) - 0.6, 0.14,
            "area $=$ share of $S(\\mathbf{q})$;  $\\dagger$ 64-site cluster",
            fontsize=8.2, color=INK3, ha="right")


# =========================================================== pyrochlore
A1 = np.array([0.0, 0.5, 0.5])
A2 = np.array([0.5, 0.0, 0.5])
A3 = np.array([0.5, 0.5, 0.0])
BASIS = np.array([[0, 0, 0], [0, .25, .25], [.25, 0, .25], [.25, .25, 0]])
W111 = np.array([1.0, 1.0, 1.0]) / np.sqrt(3)


def pyro_fragment(nc=4, wmax=0.62, rmax=1.35):
    """Slab: wide perpendicular to [111], thin along it."""
    pos, sub = [], []
    for i, j, k in itertools.product(range(-nc, nc + 1), repeat=3):
        R = i * A1 + j * A2 + k * A3
        for mu in range(4):
            r = R + BASIS[mu]
            w = r @ W111
            rho = np.linalg.norm(r - w * W111)
            if -0.05 <= w <= wmax and rho <= rmax:
                pos.append(r)
                sub.append(mu)
    pos = np.array(pos)
    sub = np.array(sub)
    d = np.linalg.norm(pos[:, None] - pos[None, :], axis=2)
    bonds = [(i, j) for i in range(len(pos)) for j in range(i)
             if abs(d[i, j] - 0.353553) < 1e-3]
    adj = {i: set() for i in range(len(pos))}
    for i, j in bonds:
        adj[i].add(j)
        adj[j].add(i)
    # tetrahedra = 4-cliques of the bond graph
    tets = set()
    for i, j in bonds:
        for k in adj[i] & adj[j]:
            for l in adj[i] & adj[j] & adj[k]:
                tets.add(frozenset((i, j, k, l)))
    tets = [sorted(t) for t in tets]
    # hexagons (three sublattices twice each)
    hexes, seen = [], set()
    for start in range(len(pos)):
        stack = [(start, [start])]
        while stack:
            cur, path = stack.pop()
            if len(path) == 6:
                if start in adj[cur]:
                    key = frozenset(path)
                    if key not in seen and len(key) == 6:
                        subs = sorted(sub[list(path)])
                        if subs in ([0, 0, 1, 1, 2, 2], [0, 0, 1, 1, 3, 3],
                                    [0, 0, 2, 2, 3, 3], [1, 1, 2, 2, 3, 3]):
                            seen.add(key)
                            hexes.append(list(path))
                continue
            for nxt in adj[cur]:
                if nxt not in path and nxt > start - 1:
                    stack.append((nxt, path + [nxt]))
    fam = np.array([({0, 1, 2, 3} - set(sub[h].tolist())).pop()
                    for h in hexes])
    return pos, sub, bonds, hexes, fam, tets


def view_frame(tilt_deg=62, az_deg=0.0):
    """Orthographic frame: screen-y = [111] tilted toward the viewer,
    with an azimuthal rotation about [111] to unstack plaquettes."""
    ey = W111.copy()
    ex0 = np.array([1.0, -1.0, 0.0]) / np.sqrt(2)
    ez0 = np.cross(ex0, ey)
    a = np.deg2rad(az_deg)
    ex = np.cos(a) * ex0 + np.sin(a) * ez0
    ez = np.cross(ex, ey)
    t = np.deg2rad(90 - tilt_deg)
    ey2 = np.cos(t) * ey + np.sin(t) * ez
    ey2 /= np.linalg.norm(ey2)
    return ex, ey2, np.cross(ex, ey2)


def flux_face(v):
    """Sample the SAME colormap the colorbar shows."""
    v = float(np.clip(v, -1, 1))
    return tuple(FLUXCMAP(0.5 * (v + 1))[:3])

def shade(base, s):
    return tuple(float(np.clip(c * s, 0, 1)) for c in base)


def render_pyro(ax, frag, tilt=60, az=0.0, plaq_val=None, zero_label=True,
                cmap=False,
                stretch=1.0):
    """Depth-sorted render: shaded tetrahedra, translucent plaquettes,
    opacity and size falling off with distance.  plaq_val maps a hexagon
    index to a flux in [-1,1] (None = do not draw that plaquette).
    stretch > 1 gives an exploded view along [111]: the kagome layers
    separate, and the tilted plaquettes become unmistakable slanted
    connectors between them."""
    pos0, sub, bonds, hexes, fam, tets = frag
    w = pos0 @ W111
    pos = pos0 + (stretch - 1.0) * w[:, None] * W111[None, :]
    ex, ey, ez = view_frame(tilt, az)
    P = np.column_stack([pos @ ex, pos @ ey])
    Dp = pos @ ez
    dmin, dmax = Dp.min(), Dp.max()

    def near(d):
        return (d - dmin) / max(dmax - dmin, 1e-9)

    light = np.array([0.30, 0.45, 0.84])
    light = light / np.linalg.norm(light)
    R = np.vstack([ex, ey, ez])
    items = []                        # (depth, kind, payload)

    tet_base = (0.910, 0.899, 0.874)
    for t in tets:
        for f in itertools.combinations(t, 3):
            p3 = pos[list(f)]
            n = np.cross(p3[1] - p3[0], p3[2] - p3[0])
            nv = R @ n
            if nv[2] < 0:
                nv = -nv
            nrm = np.linalg.norm(nv)
            if nrm < 1e-12 or nv[2] / nrm < 0.06:
                continue              # cull back / edge-on faces
            s = 0.68 + 0.32 * max(float(nv @ light) / nrm, 0.0)
            items.append((float(np.mean(Dp[list(f)])), "tri",
                          (P[list(f)], shade(tet_base, s))))
    in_tet = sorted({i for t in tets for i in t})
    for i in in_tet:
        items.append((float(Dp[i]), "site", (P[i], sub[i] == 0)))
    if plaq_val is not None:
        for hi, h in enumerate(hexes):
            v = plaq_val(hi)
            if v is None:
                continue
            pts = P[h]
            c2 = pts.mean(axis=0)
            ang = np.arctan2(pts[:, 1] - c2[1], pts[:, 0] - c2[0])
            poly = pts[np.argsort(ang)]
            items.append((float(np.mean(Dp[h])) + 0.02, "hex",
                          (poly, float(v), c2)))

    # plaquettes drawn as a second pass on top (depth-sorted among
    # themselves): strict occlusion would hide the far flux band, which
    # is the object the panel exists to show
    items.sort(key=lambda o: (o[1] == "hex", o[0]))
    for d, kind, pl in items:
        a = 0.30 + 0.62 * near(d)
        if kind == "tri":
            poly, col = pl
            ax.add_patch(Polygon(poly, closed=True, facecolor=col,
                                 edgecolor=shade(col, 0.82), lw=0.5,
                                 alpha=a))
        elif kind == "site":
            p2, tri = pl
            r = 0.018 + 0.020 * near(d)
            ax.add_patch(Circle(p2, r, facecolor=INK2 if tri else "white",
                                edgecolor=INK2, lw=0.6, alpha=a))
        else:
            poly, v, c2 = pl
            if abs(v) < 0.05:
                ax.add_patch(Polygon(poly, closed=True, facecolor="white",
                                     edgecolor=INK2, lw=0.9,
                                     alpha=0.55 + 0.40 * near(d)))
                if zero_label and near(d) > 0.55:
                    ax.text(c2[0], c2[1], r"$0$", fontsize=6.8,
                            color=INK3, ha="center", va="center")
            else:
                ax.add_patch(Polygon(poly, closed=True,
                                     facecolor=flux_face(v),
                                     edgecolor=shade(flux_face(v), 0.55),
                                     lw=0.8, alpha=0.50 + 0.45 * near(d)))
    ax.set_aspect("equal", adjustable="datalim", anchor="W")
    ax.autoscale()
    ax.margins(0.04)
    return P, ex, ey

FLUXCMAP = matplotlib.colors.LinearSegmentedColormap.from_list(
    "flux", ["#617be0", "#ffffff", "#e64769"])


AXES4 = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]],
                 dtype=float) / np.sqrt(3.0)


def canonical_path(pos, sub, h, axis):
    """Orient the hexagon path with fixed chirality about its family
    axis and a fixed starting sublattice, so the alternating delta has a
    physical sign (not the cycle-search's arbitrary one)."""
    p = list(h)
    c = pos[p].mean(axis=0)
    s = 0.0
    for i in range(6):
        a = pos[p[i]] - c
        b = pos[p[(i + 1) % 6]] - c
        s += float(np.dot(np.cross(a, b), axis))
    if s < 0:
        p = p[::-1]
    mn = min(sub[i] for i in p)
    cands = [i for i in range(6) if sub[p[i]] == mn]
    m = min(cands, key=lambda i: sub[p[(i + 1) % 6]])
    return p[m:] + p[:m]


def driven_flux(frag, q, eps=(0.0, 1.0, -1.0, 0.0)):
    """Exact flux the wave E_mu(r) = eps_mu e^{i q.r} drives through each
    plaquette, with canonically oriented loops; one global phase chosen
    to make the pattern maximally real; normalized to max |Phi|."""
    pos, sub, bonds, hexes, fam, tets = frag
    eps = np.asarray(eps, dtype=float)
    delta = np.array([1, -1, 1, -1, 1, -1], dtype=float)
    gs = []
    for h, f in zip(hexes, fam):
        p = canonical_path(pos, sub, h, AXES4[f])
        gs.append(np.sum(delta * eps[sub[p]]
                         * np.exp(-1j * (pos[p] @ q))))
    gs = np.array(gs)
    big = np.abs(gs) > 1e-9
    phi = -0.5 * np.angle(np.sum(gs[big] ** 2)) if big.any() else 0.0
    vals = np.real(gs * np.exp(1j * phi))
    m = np.max(np.abs(vals))
    return vals / m if m > 0 else vals

def add_flux_cbar(fig, cax):
    sm = plt.cm.ScalarMappable(
        norm=matplotlib.colors.Normalize(-1, 1), cmap=FLUXCMAP)
    cb = fig.colorbar(sm, cax=cax)
    cb.set_ticks([-1, 0, 1])
    cb.ax.tick_params(labelsize=7, colors=INK2, length=2)
    cb.outline.set_edgecolor(EDGE)
    cb.ax.set_title(r"$\Phi_p/\Phi_{\max}$", fontsize=7.2, color=INK, pad=5)


def panel_photon(ax, frag):
    """Long wavelength, q in-plane.  In the q -> 0 limit the plaquette's
    internal factor is common to every same-orientation plaquette, so
    the exact flux pattern is the envelope Re[C e^{i q.r_c}]; that
    envelope is what is drawn."""
    pos, sub, bonds, hexes, fam, tets = frag
    exv = np.array([1.0, -1.0, 0.0]) / np.sqrt(2)
    cx = np.array([pos[h].mean(axis=0) @ exv for h in hexes])
    lo, hi = cx.min(), cx.max()
    lam = 1.9 * (hi - lo)

    def val(hi_):
        return float(np.cos(2 * np.pi * (cx[hi_] - lo) / lam - np.pi / 5))

    P, ex, ey = render_pyro(ax, frag, tilt=55, plaq_val=val, cmap=True,
                            zero_label=False)
    y0 = P[:, 1].min() - 0.14
    ax.annotate("", xy=(P[:, 0].max(), y0), xytext=(P[:, 0].min(), y0),
                arrowprops=dict(arrowstyle="->", color=INK2, lw=1.3))


def panel_roton(ax_lat, ax_prof, frag):
    """Flat orthographic side view, NO perspective: x along [1,-1,0],
    y along [111] (layer spacing exaggerated).  Kagome plaquettes appear
    edge-on as white slats (exactly zero flux); one tilted family is
    drawn colored by the exact driven flux, which reverses sign from
    one interlayer region to the next."""
    pos, sub, bonds, hexes, fam, tets = frag
    qL = 2 * np.pi * np.array([0.5, 0.5, 0.5])
    vals = driven_flux(frag, qL)
    m3 = np.max(np.abs(vals[fam != 0])) if (fam != 0).any() else 1.0
    vals = vals / max(m3, 1e-12)
    layer_dbg = 1.0 / np.sqrt(3)
    for f in (1, 2, 3):
        rows_ = {}
        for h, v, ff in zip(hexes, vals, fam):
            if ff != f:
                continue
            b = int(np.floor(float(pos[h].mean(axis=0) @ W111)
                             / layer_dbg + 1e-6))
            rows_.setdefault(b, []).append(float(v))
        for b in sorted(rows_):
            arr = np.array(rows_[b])
            print(f"  roton fam {f} band {b}: n={len(arr)} "
                  f"mean={arr.mean():+.3f} spread={np.ptp(arr):.3f}")

    stretch = 1.55
    exv = np.array([1.0, -1.0, 0.0]) / np.sqrt(2)
    w = pos @ W111
    P = np.column_stack([pos @ exv, stretch * w])

    segs = [[P[i], P[j]] for i, j in bonds]
    ax_lat.add_collection(LineCollection(segs, colors=BOND, lw=0.7,
                                         alpha=0.5, zorder=1))
    # projected tetrahedra: the side view of pyrochlore is rows of
    # up/down triangles -- this is what makes the lattice recognizable
    for t in tets:
        pts = P[list(t)]
        c = pts.mean(axis=0)
        ang = np.argsort(np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0]))
        ax_lat.add_patch(Polygon(pts[ang], closed=True,
                                 facecolor="#dedbd3", edgecolor="#c4c1b8",
                                 lw=0.5, alpha=0.55, zorder=1.5))
    for i in range(len(pos)):
        tri = sub[i] == 0
        ax_lat.add_patch(Circle((P[i, 0], P[i, 1]),
                                0.034 if tri else 0.026,
                                facecolor=INK2 if tri else "white",
                                edgecolor=INK2, lw=0.6, zorder=5))
    # name the planes: kagome rows (open sites) / triangular rows (filled)
    rows = {}
    for i in range(len(pos)):
        key = (round(P[i, 1], 1), sub[i] == 0)
        rows.setdefault(key, []).append(P[i, 1])
    x_lab = P[:, 0].min() - 0.16
    done = set()
    for (yk, tri), ys in sorted(rows.items()):
        band = round(yk, 1)
        if (band, tri) in done:
            continue
        done.add((band, tri))
        ax_lat.text(x_lab, float(np.mean(ys)),
                    "triangular" if tri else "kagome", fontsize=6.4,
                    color=INK3, ha="right", va="center")

    from matplotlib.patches import Ellipse
    vs, kag, slat_c, tilt_c = [], [], [], []
    for h, v, f in zip(hexes, vals, fam):
        c2 = P[h].mean(axis=0)
        wc = float(pos[h].mean(axis=0) @ W111)
        if f == 0:
            span = P[h][:, 0].max() - P[h][:, 0].min()
            ax_lat.add_patch(Ellipse(c2, span + 0.08, 0.11,
                                     facecolor="white", edgecolor=INK2,
                                     lw=1.0, zorder=3))
            ax_lat.text(c2[0], c2[1] - 0.004, r"$0$", fontsize=6.8,
                        color=INK3, ha="center", va="center", zorder=4)
            kag.append((wc, 0.0))
            slat_c.append(c2)
        else:
            # profile records all three tilted families; the lattice
            # draws only the strongest one so the bands stay crisp
            # (overlapping translucent families muddy each other)
            vs.append((wc, float(v)))
            if f == 3:
                pts = P[h]
                cc = pts.mean(axis=0)
                ang = np.arctan2(pts[:, 1] - cc[1], pts[:, 0] - cc[0])
                poly = pts[np.argsort(ang)]
                ax_lat.add_patch(Polygon(poly, closed=True,
                                         facecolor=flux_face(v),
                                         edgecolor=INK2, lw=1.0,
                                         alpha=0.95, zorder=2))
                tilt_c.append((c2, wc))

    # name the plaquettes once each
    if slat_c:
        s0 = max(slat_c, key=lambda c: c[1])
        ax_lat.annotate("hexagonal plaquette,\nedge-on "
                        r"($\perp\mathbf{q}$)",
                        xy=(s0[0], s0[1] + 0.04),
                        xytext=(s0[0] - 0.30, s0[1] + 0.42),
                        fontsize=6.4, color=INK2, ha="center",
                        arrowprops=dict(arrowstyle="->", color=INK2,
                                        lw=0.9), zorder=8)
    if tilt_c:
        t0 = max(tilt_c, key=lambda c: c[1])[0]
        ax_lat.annotate("tilted hexagonal\nplaquette",
                        xy=(t0[0], t0[1] + 0.06),
                        xytext=(t0[0] + 0.62, t0[1] + 0.40),
                        fontsize=6.4, color=INK2, ha="center",
                        arrowprops=dict(arrowstyle="->", color=INK2,
                                        lw=0.9), zorder=8)

    x1 = P[:, 0].max() + 0.12
    ax_lat.annotate("", xy=(x1, P[:, 1].max() + 0.03),
                    xytext=(x1, P[:, 1].min() - 0.03),
                    arrowprops=dict(arrowstyle="->", color=ROTON, lw=1.5))
    ax_lat.text(x1 + 0.08, P[:, 1].mean(), r"$\mathbf{q}$", fontsize=10,
                color=ROTON, va="center")
    ax_lat.set_aspect("equal", adjustable="datalim", anchor="W")
    ax_lat.autoscale()
    ax_lat.margins(0.05)

    layer_h = 1.0 / np.sqrt(3)
    wmaxv = max(w for w, _ in vs + kag)
    wgrid = np.linspace(-0.12, wmaxv + 0.12, 200)
    prof = np.sin(np.pi * wgrid / layer_h)
    ph0 = np.pi * np.mean([w for w, v in vs if v > 0]) / layer_h
    if np.sin(ph0) < 0:
        prof = -prof
    ax_prof.plot(prof, wgrid, "-", color=INK3, lw=1.1, zorder=1)
    ax_prof.plot(-0.5 * prof, wgrid, "--", color=INK3, lw=0.9,
                 alpha=0.75, zorder=1)
    ax_prof.axvline(0, color=EDGE, lw=0.8, zorder=0)
    for wv, v in vs:
        ax_prof.scatter([v], [wv], s=26, color=FLUXCMAP(0.5 * (v + 1)),
                        edgecolor=INK2, lw=0.6, zorder=3)
    for wv, v in kag:
        ax_prof.scatter([0], [wv], s=22, facecolor="white",
                        edgecolor=INK2, lw=0.8, zorder=4)
    ax_prof.set_xlim(-1.35, 1.35)
    ax_prof.set_xticks([-1, 0, 1])
    ax_prof.set_yticks([])
    ax_prof.set_xlabel(r"$\Phi_p/\Phi_{\max}$", fontsize=8, color=INK,
                       labelpad=2)
    ax_prof.set_ylabel(r"height along $[111]$", fontsize=6.8,
                       color=INK2, labelpad=1)
    ax_prof.tick_params(labelsize=7, colors=INK2, length=2)
    for sp in ("top", "right"):
        ax_prof.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax_prof.spines[sp].set_color(EDGE)


def panel_multi(ax):
    X0 = 0.02
    ax.text(X0, 0.84, r"$-2K\cos B_p=\mathrm{const}+K B_p^{2}"
            r"-\frac{K}{12}B_p^{4}+\dots$", fontsize=9.6, color=INK,
            transform=ax.transAxes)
    x0, y0 = X0 + 0.20, 0.36
    ax.plot([X0 + 0.02, x0], [y0, y0], color=PHOTON, lw=2.4,
            transform=ax.transAxes, solid_capstyle="round")
    for dy in (0.20, 0.0, -0.20):
        ax.plot([x0, x0 + 0.30], [y0, y0 + dy], color=MULTI, lw=2.0,
                transform=ax.transAxes, solid_capstyle="round")
    ax.add_patch(Circle((x0, y0), 0.020, facecolor=INK, edgecolor="none",
                        transform=ax.transAxes, zorder=5))
    ax.text(X0 + 0.58, y0 - 0.03,
            "the quartic vertex:\none photon $\\to$ three",
            fontsize=8.6, color=INK2, transform=ax.transAxes)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)


CAP_PHOTON = ("top view of one kagome layer; long wavelength\n"
              "($\\mathbf{q}\\perp[111]$): the flux varies slowly from "
              "plaquette to plaquette")
CAP_ROTON = ("$\\mathbf{q}\\parallel[111]$: normal-$\\parallel\\mathbf{q}$"
             " plaquettes: exactly zero flux;\ntilted families carry flux "
             "in ratio $(+1,-\\frac{1}{2},-\\frac{1}{2})$ (sum 0),\nevery "
             "plaquette reversing sign in the next region")
CAP_MULTI = "the interaction the free theory discards"


def main():
    z = np.load(D / "ring48_channels.npz", allow_pickle=True)
    scale = json.load(open(D / "ring48_gaussian_match.json"))["scale"]
    frag_p = pyro_fragment(5, wmax=0.62, rmax=2.15)
    frag_r = pyro_fragment(4, wmax=1.45, rmax=1.30)

    fig = plt.figure(figsize=(7.05, 4.75))
    fig.patch.set_facecolor("white")
    gs = fig.add_gridspec(2, 2, width_ratios=[1.42, 1.0],
                          height_ratios=[1.00, 1.30],
                          hspace=0.62, wspace=0.10,
                          left=0.065, right=0.965, top=0.955,
                          bottom=0.105)
    spectrum_panel(fig.add_subplot(gs[:, 0]), z, scale)

    def strip(a):
        a.set_xticks([]), a.set_yticks([])
        for s in a.spines.values():
            s.set_visible(False)
        return a

    gp = gs[0, 1].subgridspec(1, 2, width_ratios=[1.0, 0.045],
                              wspace=0.08)
    a0 = strip(fig.add_subplot(gp[0]))
    panel_photon(a0, frag_p)
    add_flux_cbar(fig, fig.add_subplot(gp[1]))
    a0.set_title("photon", fontsize=11, color=PHOTON, loc="left", pad=4)
    a0.text(0.0, -0.13, CAP_PHOTON, fontsize=7.8, color=INK2,
            transform=a0.transAxes, va="top")

    gr = gs[1, 1].subgridspec(1, 3, width_ratios=[1.25, 0.42, 0.045],
                              wspace=0.44)
    a1 = strip(fig.add_subplot(gr[0]))
    a1p = fig.add_subplot(gr[1])
    panel_roton(a1, a1p, frag_r)
    add_flux_cbar(fig, fig.add_subplot(gr[2]))
    a1.set_title("roton", fontsize=11, color=ROTON, loc="left", pad=14)
    a1.text(0.0, -0.19, CAP_ROTON, fontsize=7.6, color=INK2,
            transform=a1.transAxes, va="top")

    for ext in ("pdf", "png"):
        fig.savefig(OUTDIR / f"fig_hero.{ext}", dpi=300,
                    bbox_inches="tight")
    print("wrote", OUTDIR / "fig_hero.pdf")


if __name__ == "__main__":
    main()
