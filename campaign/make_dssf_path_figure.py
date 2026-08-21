#!/usr/bin/env python3
"""S^zz along the connected Gamma-X-W-L-Gamma circuit.

Two rows, because they answer different questions.

  top     trace channel, sum_mu |A^mu(q)|^2 -- the quantity the eight-column
          panel plots.  Finite at Gamma in the periodic theory.
  bottom  uniform channel, |sum_mu A^mu(q)|^2 -- the total longitudinal probe
          sum_i S^z_i, which commutes with H.  It is an exact zero at Gamma in
          BOTH theories, which is what U(1) requires.

Reading the two together: the periodic Gamma weight is not a violation of spin
conservation, it is the three sublattice-staggered q=0 channels, which U(1)
does not protect and the transport mask does.

Open circles on the axis mark the eight genuine FCC-32 momenta.  Everything
between them is the finite-cluster continuation -- smooth, but not independent
information, since only the commensurate points are free of the periodic-image
convention.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import PowerNorm  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DSSF = ROOT / "campaign" / "outputs" / "wp3_dssf"
FIGS = ROOT / "paper" / "figs"
INK = "#17191d"

plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "dejavuserif",
    "axes.labelsize": 8.6,
    "axes.titlesize": 8.6,
    "xtick.labelsize": 7.6,
    "ytick.labelsize": 7.4,
    "axes.linewidth": 0.7,
    "savefig.bbox": "tight",
})


def broaden(record, key, energy, width):
    """(n_energy, n_path) broadened intensity for one channel."""
    excitation = np.asarray(record["excitation"], dtype=float)
    weights = np.asarray(record[key], dtype=float)          # (n_path, n_state)
    keep = excitation > 1.0e-9
    delta = energy[:, None] - excitation[None, keep]
    kernel = np.exp(-0.5 * (delta / width) ** 2) / (np.sqrt(2 * np.pi) * width)
    return kernel @ weights[:, keep].T


def corner_ticks(record):
    positions, labels = [], []
    for distance, label in zip(record["distance"], record["corner_label"]):
        if label:
            positions.append(float(distance))
            labels.append(label)
    return positions, labels


def path_figure(coupling: float) -> None:
    tag = f"{coupling:+.3f}".replace(".", "p")
    records = {}
    for method in ("periodic", "masked"):
        path = DSSF / f"dssfpath_{method}_fcc32_L1_{tag}.json"
        if not path.exists():
            print(f"skip {coupling:+.3f}: missing {path.name}")
            return
        records[method] = json.loads(path.read_text())

    maximum = max(max(r["excitation"]) for r in records.values())
    energy = np.linspace(0.0, 1.03 * maximum, 600)
    width = maximum / 90.0
    distance = np.asarray(records["periodic"]["distance"], dtype=float)

    maps = {(method, key): broaden(records[method], key, energy, width)
            for method in records for key in ("trace_weight", "uniform_weight")}
    # One colour scale for the whole figure: the point is that the bottom row
    # is empty at Gamma, and per-panel scaling would hide it by renormalising.
    vmax = max(float(m.max()) for m in maps.values())

    figure, axes = plt.subplots(2, 2, figsize=(7.1, 5.0), sharex=True,
                               sharey=True, gridspec_kw={"wspace": 0.08,
                                                         "hspace": 0.16})
    titles = {
        ("periodic", "trace_weight"):
            r"(a) periodic, $\sum_\mu|A^\mu|^2$",
        ("masked", "trace_weight"):
            r"(b) masked, $\sum_\mu|A^\mu|^2$",
        ("periodic", "uniform_weight"):
            r"(c) periodic, $|\sum_\mu A^\mu|^2$  (U(1)-protected)",
        ("masked", "uniform_weight"):
            r"(d) masked, $|\sum_\mu A^\mu|^2$  (U(1)-protected)",
    }
    image = None
    for row, key in enumerate(("trace_weight", "uniform_weight")):
        for column, method in enumerate(("periodic", "masked")):
            axis = axes[row, column]
            image = axis.pcolormesh(
                distance, energy, maps[(method, key)], shading="nearest",
                cmap="magma", norm=PowerNorm(gamma=0.45, vmin=0.0, vmax=vmax))
            axis.set_title(titles[(method, key)], loc="left", fontsize=8.0)
            ticks, labels = corner_ticks(records[method])
            for position in ticks[1:-1]:
                axis.axvline(position, color="white", linewidth=0.6,
                             alpha=0.45, zorder=3)
            axis.set_xticks(ticks)
            axis.set_xticklabels(labels)
            # the eight commensurate momenta, the only convention-free points
            exact = np.asarray(records[method]["is_exact_cluster_momentum"])
            axis.plot(distance[exact],
                      np.full(int(exact.sum()), energy[1]),
                      "o", markerfacecolor="none", markeredgecolor="white",
                      markersize=3.2, markeredgewidth=0.8, zorder=4,
                      clip_on=False)
    for axis in axes[1, :]:
        axis.set_xlabel("wave vector along $\\Gamma$–X–W–L–$\\Gamma$")
    for axis in axes[:, 0]:
        axis.set_ylabel(r"$\omega/J_{zz}$")
    colorbar = figure.colorbar(image, ax=axes, pad=0.016, fraction=0.030)
    colorbar.set_label(r"broadened $S^{zz}(\mathbf{q},\omega)$")
    figure.suptitle(
        rf"FCC-32 longitudinal response along a connected path, "
        rf"$J_\pm/J_{{zz}}={coupling:+.3f}$, "
        rf"{records['periodic']['nstates_computed']} states",
        y=0.985, fontsize=9.0)
    stem = f"fig_fcc32_dssf_path_{tag}"
    figure.savefig(FIGS / f"{stem}.pdf")
    figure.savefig(FIGS / f"{stem}.png", dpi=220)
    plt.close(figure)

    gamma = int(np.argmin(np.linalg.norm(
        np.asarray(records["periodic"]["path"]), axis=1)))
    print(f"wrote {FIGS / (stem + '.pdf')}")
    for method in ("periodic", "masked"):
        record = records[method]
        print(f"  {method:9s} Gamma: "
              f"uniform={np.sum(record['uniform_weight'][gamma]):.4e}  "
              f"trace={np.sum(record['trace_weight'][gamma]):.4e}")


def main() -> None:
    FIGS.mkdir(parents=True, exist_ok=True)
    couplings = [float(v) for v in sys.argv[1:]] or [+0.046, -0.200]
    for coupling in couplings:
        path_figure(coupling)


if __name__ == "__main__":
    main()
