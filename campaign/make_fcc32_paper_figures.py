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


# The eight FCC-32 momenta are not eight unrelated points: they are three
# high-symmetry stars of the fcc Brillouin zone -- one Gamma, three X, four L.
# `allowed_momenta` returns them in the order the cluster enumerates them,
# which interleaves the stars (Gamma, L, L, X, L, X, X, L) and makes the
# response look like eight arbitrary columns.  Ordering them along the
# canonical L-Gamma-X traversal instead turns the panel into a path through
# the zone, and puts Gamma -- where the mask sends the weight to zero -- in the
# middle where the contrast is legible.
STAR_ORDER = ("L", r"$\Gamma$", "X")


def momentum_star(momentum) -> str:
    """Which high-symmetry star of the fcc zone this cluster momentum sits on.

    Classified by the multiset of |components| in cubic r.l.u., which is the
    invariant the cubic point group preserves: (0,0,0) is Gamma, one unit
    component is X, three half components is L.  Anything else (W, K, U --
    absent on this cluster, but a larger one would have them) is returned
    verbatim so it groups separately rather than being silently mislabelled.
    """
    magnitudes = sorted(abs(float(value)) for value in momentum)
    if np.allclose(magnitudes, [0.0, 0.0, 0.0]):
        return r"$\Gamma$"
    if np.allclose(magnitudes, [0.0, 0.0, 1.0]):
        return "X"
    if np.allclose(magnitudes, [0.5, 0.5, 0.5]):
        return "L"
    return "?"


def path_order(momenta) -> list[int]:
    """Indices of `momenta` sorted along the L-Gamma-X path.

    Within a star the members are symmetry-equivalent and their order carries
    no meaning, so they are sorted by coordinate purely for reproducibility --
    the same input always yields the same figure.  Unrecognised stars are
    appended at the end rather than dropped.
    """
    def key(index):
        star = momentum_star(momenta[index]["momentum"])
        rank = (STAR_ORDER.index(star) if star in STAR_ORDER
                else len(STAR_ORDER))
        return (rank, tuple(float(v) for v in momenta[index]["momentum"]))
    return sorted(range(len(momenta)), key=key)


def star_groups(momenta, order) -> list[tuple[str, int, int]]:
    """(star, first column, last column) for each contiguous group."""
    groups: list[tuple[str, int, int]] = []
    for column, index in enumerate(order):
        star = momentum_star(momenta[index]["momentum"])
        if groups and groups[-1][0] == star:
            groups[-1] = (star, groups[-1][1], column)
        else:
            groups.append((star, column, column))
    return groups


# The circuit the panel walks.  Gamma appears twice because the path closes;
# both columns are the same data and are meant to be.
PATH_STARS = (r"$\Gamma$", "X", "L", r"$\Gamma$")


def star_columns(record, energy, width):
    """One broadened column per high-symmetry star, averaged over its members.

    FCC-32 has eight momenta but only three stars, and the members of a star
    are related by the cubic point group -- exactly degenerate when the
    calculation is converged.  Averaging them is therefore free of information
    loss and the spread across members is a convergence gauge, so it is
    returned rather than discarded.  Nothing is interpolated: a cluster with
    three stars gets three distinct columns and no more.
    """
    columns, spreads = {}, {}
    per_momentum = broaden(record, energy, width)          # (n_energy, 8)
    for index, entry in enumerate(record["momenta"]):
        columns.setdefault(momentum_star(entry["momentum"]), []).append(index)
    averaged, report = {}, {}
    for star, members in columns.items():
        block = per_momentum[:, members]
        averaged[star] = block.mean(axis=1)
        totals = np.array([record["momenta"][m]["total_inelastic_weight"]
                           for m in members], dtype=float)
        scale = float(np.mean(np.abs(totals)))
        spreads[star] = (float(totals.max() - totals.min()) / scale
                         if scale > 1e-14 else 0.0)
        report[star] = (len(members), spreads[star])
    return averaged, report


def broaden(record: dict, energy: np.ndarray, width: float,
            order=None) -> np.ndarray:
    momenta = record["momenta"]
    if order is None:
        order = range(len(momenta))
    order = list(order)
    intensity = np.zeros((len(energy), len(order)))
    normalizer = 1.0 / (np.sqrt(2.0 * np.pi) * width)
    for column, index in enumerate(order):
        momentum = momenta[index]
        excitation = np.asarray(momentum["excitation"], dtype=float)
        weight = np.asarray(momentum["weight"], dtype=float)
        keep = (excitation > 1.0e-9) & (weight > 0.0)
        if np.any(keep):
            delta = energy[:, None] - excitation[None, keep]
            intensity[:, column] = (
                np.exp(-0.5 * (delta / width)**2)
                @ weight[keep]) * normalizer
    return intensity


def dssf_figure(coupling: float = -0.20, stem: str = "fig_fcc32_dssf",
                periodic_path: Path | None = None,
                masked_path: Path | None = None) -> None:
    tag = f"{coupling:+.3f}".replace(".", "p")
    periodic_path = periodic_path or DSSF / f"dssf_periodic_fcc32_L1_{tag}.json"
    masked_path = masked_path or DSSF / f"dssf_fcc32_L1_{tag}_pp+0p000.json"
    periodic = json.loads(periodic_path.read_text())
    masked = json.loads(masked_path.read_text())
    if len(periodic["momenta"]) != len(masked["momenta"]):
        raise RuntimeError("periodic/masked momentum counts do not match")
    for raw_q, masked_q in zip(periodic["momenta"], masked["momenta"]):
        if not np.allclose(raw_q["momentum"], masked_q["momentum"]):
            raise RuntimeError("periodic/masked momentum order does not match")
    # The two panels must be the same calculation up to the mask.  A state
    # count mismatch silently changes the integrated weight per momentum, so
    # it is a hard error rather than a caption footnote.
    states = (periodic.get("nstates_computed") or
              periodic.get("nstates_requested"))
    masked_states = (masked.get("nstates_computed") or
                     masked.get("nstates_requested"))
    if states != masked_states:
        raise RuntimeError(
            f"periodic uses {states} states, masked {masked_states}; "
            "the panels would not be comparable")

    maximum = max(
        max(max(momentum["excitation"]) for momentum in periodic["momenta"]),
        max(max(momentum["excitation"]) for momentum in masked["momenta"]))
    energy = np.linspace(0.0, 1.03 * maximum, 700)
    width = maximum / 90.0
    width_power = int(np.floor(np.log10(width)))
    width_coefficient = width / 10.0**width_power

    raw_stars, raw_report = star_columns(periodic, energy, width)
    masked_stars, masked_report = star_columns(masked, energy, width)
    raw_map = np.stack([raw_stars[s] for s in PATH_STARS], axis=1)
    masked_map = np.stack([masked_stars[s] for s in PATH_STARS], axis=1)
    vmax = max(float(raw_map.max()), float(masked_map.max()))

    figure, axes = plt.subplots(
        1, 2, figsize=(6.6, 3.3), sharey=True,
        gridspec_kw={"wspace": 0.09})
    images = []
    for axis, matrix, title in zip(
            axes, (raw_map, masked_map),
            ("(a) periodic FCC-32", "(b) transport-masked FCC-32")):
        # Blocky on purpose: FCC-32 resolves exactly these momenta, so each
        # column is one measured point and the sharp edge between columns is
        # the truth.  A smooth image here would draw a dispersion the cluster
        # cannot see -- the nearest momenta to Gamma are the zone-boundary X
        # and L, so there is no small-q information to interpolate through.
        image = axis.imshow(
            matrix, origin="lower", aspect="auto", interpolation="nearest",
            extent=(0.0, len(PATH_STARS), energy[0], energy[-1]),
            cmap="magma", norm=PowerNorm(gamma=0.45, vmin=0.0, vmax=vmax))
        images.append(image)
        for edge in range(1, len(PATH_STARS)):
            axis.axvline(edge, color="white", linewidth=0.9, alpha=0.8,
                         zorder=3)
        axis.set_xticks(np.arange(len(PATH_STARS)) + 0.5)
        axis.set_xticklabels(PATH_STARS)
        axis.set_xlabel("cluster momentum")
        axis.set_title(title, loc="left")
        axis.spines["top"].set_visible(True)
        axis.spines["right"].set_visible(True)
    axes[0].set_ylabel(r"$\omega/J_{zz}$")
    for name, report in (("periodic", raw_report), ("masked", masked_report)):
        for star, (members, spread) in sorted(report.items()):
            print(f"    {name:8s} {star:10s} {members} member(s), "
                  f"weight spread {spread:.2e}")
    colorbar = figure.colorbar(images[-1], ax=axes, pad=0.018, fraction=0.035)
    colorbar.set_label(r"broadened $S^{zz}(\mathbf{q},\omega)$")
    figure.suptitle(
        rf"Longitudinal response at $J_\pm/J_{{zz}}={coupling:+.3f}$, "
        rf"{states} states; Gaussian width $\eta/J_{{zz}}="
        rf"{width_coefficient:.1f}\times10^{{{width_power}}}$",
        y=0.995, fontsize=9.0)
    figure.subplots_adjust(left=0.085, right=0.90, bottom=0.155, top=0.855)
    figure.savefig(FIGS / f"{stem}.pdf")
    figure.savefig(FIGS / f"{stem}.png", dpi=220)
    plt.close(figure)
    print(f"wrote {FIGS / (stem + '.pdf')}  path={list(PATH_STARS)}")


def main() -> None:
    FIGS.mkdir(parents=True, exist_ok=True)
    thermodynamics_figure()
    dssf_figure()                                  # Fig. 2, Jpm/Jzz = -0.20

    # The QMC-comparable coupling.  Its masked run must come from the clean
    # 144-state window: at 120 states the window cuts a degenerate multiplet
    # and the four L-point weights, which cubic symmetry forces to be equal,
    # split by 5.6%.  See campaign/run_dssf_window_check.sbatch.
    scan = DSSF / "window_scan"
    for coupling, stem, masked_file in (
            (+0.046, "fig_fcc32_dssf_qmc",
             scan / "dssf_fcc32_L1_+0p046_n144.json"),
            (-0.046, "fig_fcc32_dssf_signcheck",
             DSSF / "dssf_fcc32_L1_-0p046_pp+0p000.json")):
        tag = f"{coupling:+.3f}".replace(".", "p")
        periodic_file = DSSF / f"dssf_periodic_fcc32_L1_{tag}.json"
        if not (masked_file.exists() and periodic_file.exists()):
            print(f"skip {stem}: missing input")
            continue
        try:
            dssf_figure(coupling, stem=stem, periodic_path=periodic_file,
                        masked_path=masked_file)
        except RuntimeError as error:
            print(f"skip {stem}: {error}")


if __name__ == "__main__":
    main()
