#!/usr/bin/env python3
"""FCC-32-only figures for the masked-Feshbach manuscript.

Figure 1: paired periodic/masked ice-band thermodynamics on FCC-32.
Figure 2: periodic/masked Szz(q,w) at all eight FCC-32 momenta.

The loop-ratio figure is produced separately by make_clean_paper_figures.py;
it already uses FCC-32 exclusively.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import PowerNorm  # noqa: E402
import numpy as np  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
THERMO = ROOT / "campaign" / "outputs" / "fcc32_thermodynamics"
DSSF = ROOT / "campaign" / "outputs" / "wp3_dssf"
FIGS = ROOT / "paper" / "figs"

INK = "#17191d"
GRID = "#e2e5e9"
PERIODIC = "#a95a45"
MASKED = "#356399"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "mathtext.fontset": "cm",
    "font.size": 8.5,
    "axes.labelsize": 8.8,
    "axes.titlesize": 9.0,
    "legend.fontsize": 7.4,
    "xtick.labelsize": 7.2,
    "ytick.labelsize": 7.4,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.7,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "savefig.bbox": "tight",
})


def load_thermo(coupling: float) -> tuple[dict, dict[str, np.ndarray]]:
    tag = f"{coupling:+.3f}".replace(".", "p")
    metadata = json.loads((THERMO / f"thermo_{tag}.json").read_text())
    arrays = dict(np.load(THERMO / f"thermo_{tag}.npz"))
    return metadata, arrays


def dominant_peak(record: dict, method: str) -> float:
    """Temperature of the highest heat-capacity maximum in a complete band."""
    _, arrays = load_thermo(float(record["jpm"]))
    candidates = np.asarray(record[f"{method}_peaks"], dtype=float)
    if not len(candidates):
        raise RuntimeError(f"no {method} heat-capacity peak at {record['jpm']}")
    temperature = arrays["temperature"]
    curve = arrays[f"{method}_heat_capacity_per_site"]
    indices = np.searchsorted(temperature, candidates)
    indices = np.clip(indices, 0, len(temperature) - 1)
    return float(candidates[np.argmax(curve[indices])])


def thermodynamics_figure() -> None:
    benchmark, data = load_thermo(+0.046)
    qmc = np.genfromtxt(
        ROOT / "campaign" / "data" / "huang_2018_qmc_jpm_0p046.csv",
        delimiter=",", names=True)

    scanned = []
    for path in sorted(THERMO.glob("thermo_-*.json")):
        scanned.append(json.loads(path.read_text()))
    scanned.sort(key=lambda row: abs(row["jpm"]))
    records = []
    for metadata in scanned:
        if not metadata.get("comparison_complete"):
            break
        records.append(metadata)
    if not records:
        raise RuntimeError("no complete negative-coupling FCC-32 spectra")

    figure, (left, right) = plt.subplots(
        1, 2, figsize=(7.05, 3.05), gridspec_kw={"wspace": 0.28})
    temperature = data["temperature"]
    left.plot(
        temperature, data["periodic_heat_capacity_per_site"],
        color=PERIODIC, lw=1.35, label="periodic FCC-32")
    left.plot(
        temperature, data["masked_heat_capacity_per_site"],
        color=MASKED, lw=1.5, label="masked FCC-32")
    left.plot(
        qmc["temperature_over_Jzz"], qmc["heat_capacity_per_site"],
        "o", color=INK, mfc="white", mew=0.75, ms=2.5,
        label="QMC (Huang et al.)", zorder=4)
    left.set_xscale("log")
    left.set_xlim(4.5e-4, 3.0e-2)
    left.set_ylim(bottom=0.0)
    left.set_xlabel(r"$T/J_{zz}$")
    left.set_ylabel(r"$C/N$")
    left.set_title(r"(a) sign-free benchmark, $J_\pm/J_{zz}=+0.046$", loc="left")
    left.grid(True, which="both", color=GRID, lw=0.5)
    left.legend(frameon=False, loc="upper right")

    x = np.array([abs(row["jpm"]) for row in records])
    raw = np.array([dominant_peak(row, "periodic") for row in records])
    masked = np.array([dominant_peak(row, "masked") for row in records])
    right.loglog(x, raw, "s-", color=PERIODIC, ms=3.3, lw=1.25,
                 label="periodic FCC-32")
    right.loglog(x, masked, "o-", color=MASKED, ms=3.4, lw=1.35,
                 label="masked FCC-32")
    guide = np.geomspace(max(0.008, x.min()), x.max(), 300)
    right.loglog(guide, 4 * guide**2, ":", color=PERIODIC, lw=1.0,
                 label=r"$4J_\pm^2/J_{zz}$")
    right.loglog(guide, 12 * guide**3, "--", color=MASKED, lw=1.0,
                 label=r"$12|J_\pm|^3/J_{zz}^2$")
    right.set_xlabel(r"$|J_\pm|/J_{zz}$")
    right.set_ylabel(r"$T_{\rm peak}/J_{zz}$")
    right.set_title("(b) FCC-32 coupling range", loc="left")
    right.grid(True, which="both", color=GRID, lw=0.5)
    right.legend(frameon=False, loc="upper left", fontsize=6.8)
    right.text(
        0.98, 0.04,
        rf"last complete band: $|J_\pm|/J_{{zz}}={x.max():.2f}$",
        transform=right.transAxes, ha="right", va="bottom", fontsize=6.8)

    figure.savefig(FIGS / "fig_fcc32_thermodynamics.pdf")
    figure.savefig(FIGS / "fig_fcc32_thermodynamics.png", dpi=220)
    plt.close(figure)
    print(f"wrote {FIGS / 'fig_fcc32_thermodynamics.pdf'}")
    print("benchmark:", benchmark)


def momentum_label(momentum: list[float]) -> str:
    values = tuple(float(value) for value in momentum)
    if np.allclose(values, 0.0):
        return r"$\Gamma$"
    pieces = []
    for value in values:
        if np.isclose(value, 0.5):
            pieces.append(r"\frac{1}{2}")
        elif np.isclose(value, -0.5):
            pieces.append(r"-\frac{1}{2}")
        else:
            pieces.append(str(int(round(value))))
    return r"$(" + ",".join(pieces) + r")$"


def broaden(record: dict, energy: np.ndarray, width: float) -> np.ndarray:
    intensity = np.zeros((len(energy), len(record["momenta"])))
    normalizer = 1.0 / (np.sqrt(2.0 * np.pi) * width)
    for column, momentum in enumerate(record["momenta"]):
        excitation = np.asarray(momentum["excitation"], dtype=float)
        weight = np.asarray(momentum["weight"], dtype=float)
        keep = (excitation > 1.0e-9) & (weight > 0.0)
        if np.any(keep):
            delta = energy[:, None] - excitation[None, keep]
            intensity[:, column] = (
                np.exp(-0.5 * (delta / width)**2)
                @ weight[keep]) * normalizer
    return intensity


def dssf_figure(coupling: float = -0.20) -> None:
    tag = f"{coupling:+.3f}".replace(".", "p")
    periodic = json.loads(
        (DSSF / f"dssf_periodic_fcc32_L1_{tag}.json").read_text())
    masked = json.loads(
        (DSSF / f"dssf_fcc32_L1_{tag}_pp+0p000.json").read_text())
    if len(periodic["momenta"]) != len(masked["momenta"]):
        raise RuntimeError("periodic/masked momentum counts do not match")
    for raw_q, masked_q in zip(periodic["momenta"], masked["momenta"]):
        if not np.allclose(raw_q["momentum"], masked_q["momentum"]):
            raise RuntimeError("periodic/masked momentum order does not match")

    maximum = max(
        max(max(momentum["excitation"]) for momentum in periodic["momenta"]),
        max(max(momentum["excitation"]) for momentum in masked["momenta"]))
    energy = np.linspace(0.0, 1.03 * maximum, 700)
    width = maximum / 90.0
    width_power = int(np.floor(np.log10(width)))
    width_coefficient = width / 10.0**width_power
    raw_map = broaden(periodic, energy, width)
    masked_map = broaden(masked, energy, width)
    vmax = max(float(raw_map.max()), float(masked_map.max()))
    labels = [momentum_label(row["momentum"]) for row in periodic["momenta"]]

    figure, axes = plt.subplots(
        1, 2, figsize=(7.05, 3.25), sharey=True,
        gridspec_kw={"wspace": 0.10})
    images = []
    for axis, matrix, title in zip(
            axes, (raw_map, masked_map),
            ("(a) periodic FCC-32", "(b) transport-masked FCC-32")):
        image = axis.imshow(
            matrix, origin="lower", aspect="auto", interpolation="nearest",
            extent=(-0.5, len(labels) - 0.5, energy[0], energy[-1]),
            cmap="magma", norm=PowerNorm(gamma=0.45, vmin=0.0, vmax=vmax))
        images.append(image)
        axis.set_xticks(np.arange(len(labels)))
        axis.set_xticklabels(labels, rotation=35, ha="right")
        axis.set_xlabel("FCC-32 momentum (cubic r.l.u.)")
        axis.set_title(title, loc="left")
        axis.spines["top"].set_visible(True)
        axis.spines["right"].set_visible(True)
    axes[0].set_ylabel(r"$\omega/J_{zz}$")
    colorbar = figure.colorbar(images[-1], ax=axes, pad=0.018, fraction=0.035)
    colorbar.set_label(r"broadened $S^{zz}(\mathbf{q},\omega)$")
    figure.suptitle(
        rf"Longitudinal response at $J_\pm/J_{{zz}}={coupling:+.2f}$; "
        rf"Gaussian width $\eta/J_{{zz}}="
        rf"{width_coefficient:.1f}\times10^{{{width_power}}}$",
        y=0.995, fontsize=9.0)
    figure.subplots_adjust(left=0.075, right=0.90, bottom=0.20, top=0.88)
    figure.savefig(FIGS / "fig_fcc32_dssf.pdf")
    figure.savefig(FIGS / "fig_fcc32_dssf.png", dpi=220)
    plt.close(figure)
    print(f"wrote {FIGS / 'fig_fcc32_dssf.pdf'}")


def main() -> None:
    FIGS.mkdir(parents=True, exist_ok=True)
    thermodynamics_figure()
    dssf_figure()


if __name__ == "__main__":
    main()
