"""Contact sheet: six candidate (tilt, azimuth) views of the roton slab."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import make_hero_figure as mh

frag = mh.pyro_fragment(4, wmax=1.45, rmax=1.05)
views = [(50, 10), (50, 30), (62, 20), (62, 38), (72, 15), (72, 32)]
fig, axes = plt.subplots(2, 3, figsize=(13, 8))
for ax, (tilt, az) in zip(axes.ravel(), views):
    ax.set_xticks([]), ax.set_yticks([])
    import numpy as np
    pos, sub, bonds, hexes, fam, tets = frag
    qL = 2 * np.pi * np.array([0.5, 0.5, 0.5])
    def chan(h, mu):
        s_ = [i for i in h if sub[i] == mu]
        if len(s_) != 2:
            return 0j
        return (np.exp(-1j * (pos[s_[0]] @ qL))
                - np.exp(-1j * (pos[s_[1]] @ qL)))
    gs = np.array([chan(h, 1) - chan(h, 2) for h in hexes])
    big = np.abs(gs) > 1e-8
    phi = -0.5 * np.angle(np.sum(gs[big] ** 2)) if big.any() else 0.0
    vals = np.real(gs * np.exp(1j * phi))
    vals = vals / max(np.max(np.abs(vals)), 1e-12)
    mh.render_pyro(ax, frag, tilt=tilt, az=az,
                   plaq_val=lambda hi: float(vals[hi]))
    ax.set_title(f"tilt={tilt}  az={az}", fontsize=11)
fig.tight_layout()
fig.savefig(mh.OUTDIR / "roton_view_sheet.png", dpi=140)
print("wrote", mh.OUTDIR / "roton_view_sheet.png")
