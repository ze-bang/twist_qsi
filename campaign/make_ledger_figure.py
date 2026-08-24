"""fig_ledger: the mechanism, and the anharmonicity it produces.

Panel (a).  The exact energy ledger at L, 48 sites.  For this Hamiltonian
the excitation energy IS lost ring coherence, additively over plaquettes,
so each state's cost can be split by whether the probe drives the family:
the three families with nonzero curl (bright) against the one whose
plaquettes lie in the wavefronts and have g == 0 (dark).  The rigid
single-mode state pays 0.968 K of dephasing on the bright families; the
true soft eigenstate has recovered 89% of it while its dark-family cost
is unchanged.  The entire relaxation happens on the driven plaquettes.

Panel (b).  Gaussian lattice electrodynamics is harmonic, so the
single-mode state is exactly the eigenstate and f1/S = omega: the
overshoot H1 is 1.000 at every momentum for any U and K.  Measured, H1
ranks the five resolved momenta exactly as the softening does.

Provenance: dark_family_algebra (ledger), harmonicity_audit (H1),
ring48_polarization_all + f1_all_fcc223 (momenta).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif": ["cmr10", "Computer Modern Roman", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "axes.unicode_minus": False,
})

ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "paper_multiphoton" / "figs"
INK, INK2, EDGE = "#0f172a", "#475569", "#cbd5e1"
BRIGHT, DARK = "#1f66a0", "#c55a11"
ROTON = "#e11d48"

# (label, bright, dark)
LEDGER = [
    (r"partner",                       1.0822, 0.6481),
    (r"soft eigenstate",               0.1081, 0.9154),
    (r"rigid wave $S^z(\mathbf{q})|0\rangle$", 0.9678, 0.8778),
]

# (label, H1, softening omega_low/omega_G)
H1 = [
    (r"$L$",                                          1.678, 0.503),
    (r"$(\frac{1}{3},\frac{1}{3},\frac{2}{3})$",      1.346, 0.824),
    (r"$X$",                                          1.274, 0.952),
    (r"$(\frac{1}{3},\frac{1}{3},\frac{1}{3})$",      1.260, 1.076),
    (r"$(\frac{1}{6},\frac{1}{6},\frac{5}{6})$",      1.213, 1.114),
]


def main() -> int:
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.0, 2.75),
                                 gridspec_kw={"width_ratios": [1.32, 1.0]})

    # ---------------------------------------------------------- panel (a)
    for i, (lab, b, d) in enumerate(LEDGER):
        ax.barh(i, b, height=0.52, color=BRIGHT, edgecolor="white", lw=0.7)
        ax.barh(i, d, left=b, height=0.52, color=DARK,
                edgecolor="white", lw=0.7)
        # a narrow segment cannot hold its own label; put it above instead
        if b < 0.25:
            ax.text(b / 2, i + 0.36, f"{b:.3f}", ha="center", va="bottom",
                    color=BRIGHT, fontsize=7.2)
        else:
            ax.text(b / 2, i, f"{b:.3f}", ha="center", va="center",
                    color="white", fontsize=7.2)
        ax.text(b + d / 2, i, f"{d:.3f}", ha="center", va="center",
                color="white", fontsize=7.2)
        ax.text(b + d + 0.05, i, f"{b+d:.3f}", ha="left", va="center",
                color=INK, fontsize=7.8)

    # the relaxation, annotated in the clear band between the bar rows
    ax.annotate("", xy=(0.115, 1.30), xytext=(0.968, 1.70),
                arrowprops=dict(arrowstyle="-|>", color=ROTON, lw=1.3,
                                connectionstyle="arc3,rad=-0.30"))
    ax.text(1.16, 1.50, "relaxation $0.968\\to0.108$\n"
            "(89% of the dephasing recovered)",
            color=ROTON, fontsize=7.2, ha="left", va="center")

    ax.set_yticks(range(len(LEDGER)))
    ax.set_yticklabels([l for l, _, _ in LEDGER], fontsize=8.2)
    ax.set_xlabel(r"excitation energy  $\omega/K$", fontsize=9)
    ax.set_xlim(0, 2.42)
    ax.set_ylim(-1.02, 2.62)
    ax.set_yticks(range(len(LEDGER)))
    ax.tick_params(labelsize=8)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    for s_ in ("left", "bottom"):
        ax.spines[s_].set_color(EDGE)

    ax.barh(-10, 0, color=BRIGHT, label="3 bright families (probe drives)")
    ax.barh(-10, 0, color=DARK, label="1 dark family (probe blind)")
    ax.legend(loc="lower center", fontsize=6.9, frameon=False,
              handlelength=1.1, ncol=1, borderaxespad=0.15,
              bbox_to_anchor=(0.52, -0.02))
    ax.set_title(r"(a)  exact energy ledger at $L$", fontsize=9,
                 loc="left", color=INK)

    # ---------------------------------------------------------- panel (b)
    bx.axhline(1.0, color=EDGE, lw=1.0, ls="--", zorder=1)
    bx.text(0.60, 1.012, "harmonic: $f_1/S=\\omega$ exactly",
            fontsize=7.0, color=INK2, va="bottom")

    xs = [s for _, _, s in H1]
    ys = [h for _, h, _ in H1]
    bx.plot(xs, ys, "-", color=EDGE, lw=1.0, zorder=2)
    for lab, h, s in H1:
        col = ROTON if lab == r"$L$" else INK2
        bx.plot([s], [h], "o", ms=6.0 if lab == r"$L$" else 4.6,
                color=col, zorder=3)
    # the three high-softening points are bunched; alternate above
    # and below so no two labels share a band
    OFF = {0: (9, -3), 1: (-8, 11), 2: (-6, -16),
           3: (3, 10), 4: (5, -17)}
    for k, (lab, h, s_) in enumerate(H1):
        bx.annotate(lab, (s_, h), textcoords="offset points",
                    xytext=OFF[k], fontsize=6.9,
                    color=ROTON if lab == r"$L$" else INK)

    bx.set_xlabel(r"softening  $\omega_{\rm low}/\omega_{\rm Gauss}$",
                  fontsize=9)
    bx.set_ylabel(r"$H_1=(f_1/S)\,/\,\omega_{\rm low}$", fontsize=9)
    bx.set_xlim(0.40, 1.30)
    bx.set_ylim(0.92, 1.84)
    bx.tick_params(labelsize=8)
    for s in ("top", "right"):
        bx.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        bx.spines[s].set_color(EDGE)
    bx.set_title(r"(b)  anharmonicity tracks the softening", fontsize=9,
                 loc="left", color=INK)

    fig.tight_layout(pad=0.5, w_pad=1.6)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    out = OUTDIR / "fig_ledger.pdf"
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
