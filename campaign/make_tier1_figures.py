"""Tier-1 probe figures: exact winding-free observables vs raw ED.

Reads the tier1_* data (per-sector bordered-operator runs) and renders the
five headline figures into campaign/outputs/tier1_probes/.

Data columns and conventions:
  winding-free  #1f6fd6 (blue)     raw ED  #c02733 (red)
  neutral references (continuum edge, guide lines) in gray.
"""
from pathlib import Path
import json
import shutil

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "campaign" / "outputs" / "tier1_probes"
OUT.mkdir(parents=True, exist_ok=True)
SCRATCH = Path("/tmp/claude-1000/-home-pc-linux-exact-diagonalization-clean/"
               "a664474d-9b8f-4ea6-9aa5-a5e201426ef2/scratchpad")

WF, RAW = "#1f6fd6", "#c02733"
GRAY, INK = "#8a8a86", "#33332f"
plt.rcParams.update({
    "font.size": 9.5, "axes.edgecolor": GRAY, "axes.linewidth": 0.8,
    "axes.grid": True, "grid.color": "#e6e6e2", "grid.linewidth": 0.6,
    "axes.axisbelow": True, "figure.facecolor": "white",
    "text.color": INK, "axes.labelcolor": INK,
    "xtick.color": INK, "ytick.color": INK,
})

TAGS = [("xxz_m005", r"$J_\pm=-0.05$"), ("xxz_m020", r"$J_\pm=-0.20$"),
        ("xxz_m030", r"$J_\pm=-0.30$"), ("cho", r"CHO $(-0.125,\,0.085)$")]

# stage the data next to the figures (provenance)
for tag, _ in TAGS:
    for ext in ("json", "npz"):
        src = SCRATCH / f"tier1_{tag}.{ext}"
        if src.exists():
            shutil.copy(src, OUT / src.name)
fieldsrc = SCRATCH / "tier1_field_-0p050.json"
if fieldsrc.exists():
    shutil.copy(fieldsrc, OUT / fieldsrc.name)

D = {tag: json.load(open(OUT / f"tier1_{tag}.json")) for tag, _ in TAGS}
Z = {tag: np.load(OUT / f"tier1_{tag}.npz") for tag, _ in TAGS}

# ------------------------------------------------------------------ 1. S(q)
def smap(cij, pos, n=141, ext=2.5):
    hs = np.linspace(-ext, ext, n)
    out = np.zeros((n, n))
    for a, h in enumerate(hs):
        q1 = 2 * np.pi * np.array([h, h, 0.0])
        for b, l in enumerate(hs):
            ph = np.exp(1j * (pos @ (q1 + 2 * np.pi * np.array([0, 0, l]))))
            out[b, a] = float(np.real(ph.conj() @ cij @ ph)) / len(pos)
    return out

fig, axes = plt.subplots(2, 4, figsize=(15.5, 7.6), constrained_layout=True)
for col, (tag, name) in enumerate(TAGS):
    z = Z[tag]
    for row, (key, lab) in enumerate([("cij_masked", "winding-free"),
                                      ("cij_raw", "raw ED")]):
        M = smap(z[key], z["positions"])
        ax = axes[row, col]
        im = ax.imshow(M, origin="lower", extent=[-2.5, 2.5, -2.5, 2.5],
                       cmap="magma", vmin=0)
        ax.set_title(f"{lab}\n{name}", fontsize=9)
        ax.grid(False)
        ax.set_xlabel("$(h,h,0)$")
        if col == 0:
            ax.set_ylabel("$(0,0,l)$")
        fig.colorbar(im, ax=ax, shrink=0.72)
fig.suptitle("Equal-time $S^{zz}(\\mathbf{q})$ — winding-free ground multiplet vs raw ED",
             fontsize=11)
fig.savefig(OUT / "fig_tier1_sq_maps.png", dpi=140)
fig.savefig(OUT / "fig_tier1_sq_maps.pdf")
plt.close(fig)

# --------------------------------------------------- 2. emergent scales
jpms, wX, gap = [], [], []
for tag, _ in TAGS:
    d = D[tag]
    s = list(d["masked_klabels"])[0]
    lv = np.array(d["masked_levels"][s])
    kl = d["masked_klabels"][s]
    band = lv[: len(kl)]
    w = next((e - band[0] for e, k in zip(band, kl)
              if k.startswith("X") or k == "mix"), np.nan)
    jpms.append(d["jpm"]); wX.append(w); gap.append(d["edge"] - d["masked_gs"])
jpms, wX, gap = map(np.array, (jpms, wX, gap))

# the C(T) cross-check: 2.4*T_ring from the exact band must land between the
# first cross-sector gap and w(X) wherever the photon reading is meaningful
def band_ct_peak(levels):
    E = np.array(levels) - min(levels)
    T = np.logspace(-4, 0.5, 6000)
    C = []
    for t in T:
        w = np.exp(-E / t); z = w.sum()
        e = (E * w).sum() / z; e2 = (E * E * w).sum() / z
        C.append((e2 - e * e) / t ** 2)
    C = np.array(C)
    pk = [i for i in range(1, len(T) - 1) if C[i] > C[i - 1] and C[i] > C[i + 1]]
    return T[pk[0]] if pk else T[np.argmax(C)]

thermal = []
for bf in ("band_-0p050.json", "band_-0p200.json", "band_-0p300.json"):
    mb = json.load(open(ROOT / "campaign" / "outputs" / "masked_band_complete" / bf))
    thermal.append(2.4 * band_ct_peak(mb["levels"]))

fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.9, 3.9), constrained_layout=True)
x = np.abs(jpms)
# -0.30 is the restructured (6->2) band: the X states there are not a photon
a1.plot(x[:2], wX[:2], "o-", color=WF, lw=2, ms=7, label=r"photon $\omega(X)$")
a1.plot(x[2], wX[2], "o", color=WF, ms=8, mfc="white", mew=2)
a1.plot(x[:3], gap[:3], "s-", color=INK, lw=2, ms=6, label="spinon gap")
a1.plot(x[:3], thermal, "d--", color=GRAY, lw=1.6, ms=6,
        label=r"$2.4\,T_{\rm ring}$ from band $C(T)$")
a1.plot(x[3], wX[3], "o", color=WF, ms=9, mfc="white", mew=2)
a1.plot(x[3], gap[3], "s", color=INK, ms=8, mfc="white", mew=2)
a1.annotate("CHO", (x[3], wX[3]), textcoords="offset points",
            xytext=(6, -12), color=WF)
a1.annotate("6$\\to$2 restructured:\nnot a photon energy", (x[2], wX[2]),
            fontsize=7.5, color=GRAY, ha="right",
            textcoords="offset points", xytext=(-8, -2))
a1.set_yscale("log")
a1.set_xlabel(r"$|J_\pm|/J_{zz}$"); a1.set_ylabel(r"energy / $J_{zz}$")
a1.set_title("Emergent scales (exact, winding-free)", fontsize=10)
a1.legend(frameon=False, fontsize=8.5)
a2.plot(x[:2], wX[:2] / gap[:2], "o-", color=WF, lw=2, ms=7)
a2.plot(x[2], wX[2] / gap[2], "o", color=WF, ms=8, mfc="white", mew=2)
a2.plot(x[3], wX[3] / gap[3], "o", color=WF, ms=9, mfc="white", mew=2)
a2.annotate("CHO", (x[3], wX[3] / gap[3]), textcoords="offset points",
            xytext=(6, -12), color=WF)
a2.annotate("restructured\n(bandwidth-based)", (x[2], wX[2] / gap[2]),
            fontsize=7.5, color=GRAY, ha="right",
            textcoords="offset points", xytext=(-8, -6))
a2.axhline(1.0, color=GRAY, lw=1, ls="--")
a2.annotate("photon reaches the matter scale", (0.06, 1.02), fontsize=8,
            color=GRAY)
a2.set_xlabel(r"$|J_\pm|/J_{zz}$")
a2.set_ylabel(r"$\omega(X)\,/\,$spinon gap")
a2.set_title("Photon–matter hierarchy of the emergent QED", fontsize=10)
fig.savefig(OUT / "fig_tier1_scales.png", dpi=140)
fig.savefig(OUT / "fig_tier1_scales.pdf")
plt.close(fig)

# --------------------------------------------------- 3. selection rule
fig, ax = plt.subplots(figsize=(6.4, 3.6), constrained_layout=True)
names = [n for _, n in TAGS]
mvals = [max(D[t]["masked_rule_q0"]) for t, _ in TAGS]
rvals = [max(D[t]["raw_rule_q0"]) for t, _ in TAGS]
xp = np.arange(len(names))
ax.bar(xp - 0.19, rvals, 0.34, color=RAW, label="raw ED")
ax.bar(xp + 0.19, np.maximum(mvals, 1e-30), 0.34, color=WF,
       label="winding-free")
ax.set_yscale("log"); ax.set_ylim(1e-30, 3)
ax.set_xticks(xp, names, fontsize=8.5)
ax.set_ylabel(r"largest $|\langle n|m_\mu|0\rangle|^2$ at $q=0$")
ax.set_title("The $\\Gamma$-point longitudinal selection rule", fontsize=10)
for i, v in enumerate(mvals):
    ax.annotate(f"{v:.0e}", (xp[i] + 0.19, max(v, 1e-30)), ha="center",
                va="bottom", fontsize=7.5, color=WF)
ax.legend(frameon=False, loc="center right")
fig.savefig(OUT / "fig_tier1_rule.png", dpi=140)
fig.savefig(OUT / "fig_tier1_rule.pdf")
plt.close(fig)

# --------------------------------------------------- 4. [111] field
fd = json.load(open(OUT / "tier1_field_-0p050.json"))
rows = fd["rows"]
h = np.array([r["h"] for r in rows])
fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.6, 3.8), constrained_layout=True)
a1.plot(h, [r["raw_e"] for r in rows], "-", color=RAW, lw=2, label="raw ED ground")
a1.plot(h, [r["masked_e"] for r in rows], "-", color=WF, lw=2,
        label="winding-free band bottom")
a1.plot(h, [r["edge"] for r in rows], "--", color=GRAY, lw=1.5,
        label="spinon continuum edge")
a1.set_xlabel(r"$h_{[111]}/J_{zz}$"); a1.set_ylabel(r"energy / $J_{zz}$")
a1.set_title(r"$J_\pm=-0.05$ in a $[111]$ field", fontsize=10)
a1.legend(frameon=False, fontsize=8.5)
a2.step(h, [r["n_band_sectors"] for r in rows], where="post", color=WF, lw=2)
a2.set_xlabel(r"$h_{[111]}/J_{zz}$")
a2.set_ylabel("sectors with a root below the edge")
a2.set_ylim(0, 26)
a2.set_title("Field-driven sector staircase", fontsize=10)
fig.savefig(OUT / "fig_tier1_field.png", dpi=140)
fig.savefig(OUT / "fig_tier1_field.pdf")
plt.close(fig)

# --------------------------------------------------- 5. 't Hooft splittings
c16 = {-0.05: 0.002841, -0.15: 0.015264, -0.20: 0.013576,
       -0.22: 0.010634, -0.25: 0.001206, -0.30: 0.020638}
f32 = {}
for f in (ROOT / "campaign" / "outputs" / "fcc32_sector_grounds").glob("grounds_L*.json"):
    d = json.load(open(f))
    L = int(f.name.split("_L")[1][0])
    gr = (sorted(d["sector_grounds"].values())
          if isinstance(d["sector_grounds"], dict) else sorted(d["sector_grounds"]))
    vals = np.array(sorted(set(np.round(gr, 10))))
    f32.setdefault(L, {})[round(d["jpm"], 3)] = vals[1] - vals[0]

fig, ax = plt.subplots(figsize=(6.6, 4.0), constrained_layout=True)
xs = sorted(c16); ax.plot([abs(x) for x in xs], [c16[x] for x in xs], "o-",
                          color=INK, lw=2, ms=6, label="cubic-16 (exact)")
for L, style, shade in ((1, "^-", "#8fb8ee"), (2, "v-", WF)):
    xs = sorted(f32[L])
    ax.plot([abs(x) for x in xs], [f32[L][x] for x in xs], style, color=shade,
            lw=2, ms=6, label=f"FCC-32 (L={L})")
ax.set_yscale("log")
ax.set_xlabel(r"$|J_\pm|/J_{zz}$")
ax.set_ylabel(r"flux-sector splitting / $J_{zz}$")
ax.set_title("'t Hooft diagnostic: flux-sector splitting vs cluster size",
             fontsize=10)
ax.annotate("shrinks with size\n(deconfined signature)", (0.05, 0.0021),
            fontsize=8, color=INK, xytext=(0.07, 0.0008),
            arrowprops=dict(arrowstyle="->", color=GRAY))
ax.annotate("grows with size\n(flux rigidity; L-truncation caveat)",
            (0.26, 0.05), fontsize=8, color=INK)
ax.annotate("6$\\to$2 sector crossing:\nsplitting passes through zero",
            (0.25, 0.00121), fontsize=8, color=INK, ha="center",
            xytext=(0.21, 0.0016),
            arrowprops=dict(arrowstyle="->", color=GRAY))
ax.legend(frameon=False, fontsize=8.5, loc="lower right")
fig.savefig(OUT / "fig_tier1_thooft.png", dpi=140)
fig.savefig(OUT / "fig_tier1_thooft.pdf")
plt.close(fig)

print("figures written to", OUT)
for f in sorted(OUT.glob("fig_tier1_*.png")):
    print("  ", f.name)
