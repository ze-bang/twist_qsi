"""The causal operator ladder: do the longer loops change any observable?

The loop tomography establishes that the winding-free ice-block Hamiltonian
contains simple contractible loops of perimeter 6, 8, 10 and 12 with an
approximately exponential amplitude hierarchy.  That is a statement about
OPERATOR CONTENT.  It does not by itself establish that the longer loops
matter, and the manuscript is careful not to claim that it does.

This script supplies the missing causal step.  From the cached masked map it
builds the truncated Hamiltonians

    H6            = diagonal + simple L=6 entries
    H6+H8         = ... + simple L=8
    H6+H8+H10     = ... + simple L=10
    H6+H8+H10+H12 = ... + simple L=12
    full          = the complete masked map (all classes, including
                    vertex-joined composites and perimeters beyond 12)

and compares them on observables rather than on amplitudes:

  * the band spectrum and its density of states;
  * the ring-scale heat-capacity maximum of the band, which is the quantity
    the paper's scaling claim is about;
  * the transport sector owning the band bottom, and the band gap;
  * a sector-resolved longitudinal response, using the fact that
    ``P S^z_q P`` is exactly diagonal in the Ising basis so that no frame
    enters the comparison.

Reading the result.  If the ladder converges rapidly -- if H6 alone already
reproduces the full masked map on these observables -- then the longer loops
are formally present but numerically inert, and "breakdown of the ring-only
theory" must stay an operator-level statement.  If it converges slowly, the
breakdown is observable and can be quantified.  Either outcome is publishable;
only one of them licenses upgrading the claim.

No refolding is required: ``f_masked`` is cached per coupling by
``run_wp1_tomography.py``, so each point is a few dense 2970x2970
diagonalizations and costs seconds.

Restartable: one JSON per coupling, skipped if present unless ``--force``.

Usage: run_causal_ladder.py [--force] [<jpm> ...]     (default: all cached)
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

import recompute_finite_size_artifact as geometry  # noqa: E402
from qsi_campaign.protocol import polarization_sector_labels  # noqa: E402
from qsi_campaign.tomography import (  # noqa: E402
    classify_flip,
    graph_distance_matrix,
    minimum_image_distance_matrix,
)

TOMO = ROOT / "campaign" / "outputs" / "wp1_tomography"
OUTPUT = ROOT / "campaign" / "outputs" / "causal_ladder"
PERIMETERS = (6, 8, 10, 12)
LADDER_ORDER = ("H6", "H6+H8", "H6+H8+H10", "H6+H8+H10+H12", "full")


def class_layers(cluster, labels, graph_distance, real_distance,
                 max_perimeter: int = 12):
    """Boolean matrices selecting the genuine simple loops of each perimeter.

    Only sets in which every site has degree exactly two inside the induced
    subgraph are counted.  Vertex-joined hexagon pairs are graph-connected but
    factorise into products of shorter processes; pooling them with genuine
    long loops is the single largest hazard in this measurement (on FCC-32 they
    are 87% of the connected L=12 entries), so they are excluded here exactly
    as they are excluded from the K_L fit.
    """
    states = np.asarray(cluster.ice_states, dtype=np.uint64)
    difference = states[:, None] ^ states[None, :]
    perimeter = np.bitwise_count(difference)
    same_sector = np.asarray(labels)[:, None] == np.asarray(labels)[None, :]
    short = (perimeter > 0) & (perimeter <= max_perimeter)

    flat = difference[short]
    unique_masks, inverse = np.unique(flat, return_inverse=True)
    keys = [classify_flip(cluster, int(m), graph_distance, real_distance)
            for m in unique_masks]
    length_of = np.array([k[0] for k in keys])
    simple_of = np.array([bool(k[2]) and int(k[1]) == 1 for k in keys])

    layers = {}
    for length in PERIMETERS:
        keep = simple_of & (length_of == length)
        layer = np.zeros(short.shape, dtype=bool)
        layer[short] = keep[inverse]
        layers[length] = layer & same_sector

    # Everything the ladder does NOT contain, split so that a non-convergent
    # ladder can be diagnosed rather than merely reported: vertex-joined or
    # disconnected sets at perimeter <= 12 (product processes, whose amplitudes
    # factorise into shorter ones), and everything of longer perimeter.
    composite = np.zeros(short.shape, dtype=bool)
    composite[short] = (~simple_of)[inverse]
    extras = {
        "composite": composite & same_sector,
        "long": (perimeter > max_perimeter) & same_sector,
    }
    return layers, extras


def band_heat_peak(levels, lo=1e-6, hi=0.5, n=4000):
    """Position of the band heat-capacity maximum: the ring-scale observable."""
    shifted = np.sort(np.asarray(levels, float))
    shifted = shifted - shifted.min()
    grid = np.geomspace(lo, hi, n)
    beta = 1.0 / grid[:, None]
    weight = np.exp(-beta * shifted[None, :])
    partition = weight.sum(axis=1)
    mean = (weight * shifted[None, :]).sum(axis=1) / partition
    second = (weight * shifted[None, :] ** 2).sum(axis=1) / partition
    heat = (second - mean * mean) / grid**2
    return float(grid[int(np.argmax(heat))])


def longitudinal_weight(cluster, vectors, energies, momentum):
    """Sector-diagonal longitudinal weight from the band eigenvectors."""
    states = np.asarray(cluster.ice_states, dtype=np.uint64)
    n_sites = int(cluster.n_sites)
    bits = (states[:, None] >> np.arange(n_sites, dtype=np.uint64)) & np.uint64(1)
    spins = bits.astype(float) - 0.5
    sublattice = np.arange(n_sites) % 4
    phase = np.exp(-2j * np.pi * (np.asarray(cluster.positions) @ momentum))
    excitation = energies - energies[0]
    weights = np.zeros(len(energies))
    for mu in range(4):
        diagonal = spins @ (phase * (sublattice == mu))
        overlap = vectors.conj().T @ (diagonal[:, None] * vectors[:, :1])
        weights += np.abs(overlap[:, 0]) ** 2 / n_sites
    weights[excitation <= 1e-12] = 0.0
    return float(weights.sum())


def norm_budget(f_masked, layers, extras):
    """Where the off-diagonal weight of the masked map actually sits.

    If the ladder fails to converge, the answer is here: either the simple
    loops carry most of the off-diagonal norm and the failure is a cancellation
    effect, or they do not and the missing weight is in product processes.
    """
    off = np.array(f_masked, dtype=float, copy=True)
    np.fill_diagonal(off, 0.0)
    total = float(np.linalg.norm(off))
    budget = {"total": total}
    accounted = np.zeros_like(off, dtype=bool)
    for length in PERIMETERS:
        budget[f"simple_L{length}"] = float(
            np.linalg.norm(np.where(layers[length], off, 0.0)))
        accounted |= layers[length]
    for name, selector in extras.items():
        budget[name] = float(np.linalg.norm(np.where(selector, off, 0.0)))
        accounted |= selector
    budget["unaccounted"] = float(
        np.linalg.norm(np.where(~accounted, off, 0.0)))
    budget["simple_fraction"] = float(
        np.sqrt(sum(budget[f"simple_L{p}"] ** 2 for p in PERIMETERS))
        / max(total, 1e-300))
    return budget


def analyse(jpm, cluster, labels, layers, extras):
    tag = f"{jpm:+.3f}".replace(".", "p")
    cached = TOMO / f"tomo_fcc32_L2_{tag}.npz"
    if not cached.exists():
        raise FileNotFoundError(f"no cached masked map for {jpm}: {cached}")
    with np.load(cached) as data:
        f_masked = np.asarray(data["f_masked"], dtype=float)

    started = time.perf_counter()
    momentum = np.array([0.5, 0.5, 0.5])

    built = {}
    running = np.diag(np.diag(f_masked))
    for index, length in enumerate(PERIMETERS):
        running = running + np.where(layers[length], f_masked, 0.0)
        built[LADDER_ORDER[index]] = running.copy()
    built["full"] = f_masked

    results = {}
    for name in LADDER_ORDER:
        energies, vectors = np.linalg.eigh(built[name])
        results[name] = {
            "ground_energy": float(energies[0]),
            "bandwidth": float(energies[-1] - energies[0]),
            "band_gap": float(energies[1] - energies[0]),
            "heat_peak": band_heat_peak(energies),
            "bottom_sector": int(np.asarray(labels)[
                int(np.argmax(np.abs(vectors[:, 0]) ** 2))]),
            "longitudinal_total_weight": longitudinal_weight(
                cluster, vectors, energies, momentum),
            "spectrum": [float(x) for x in energies],
        }

    full = results["full"]
    reference_spectrum = np.asarray(full["spectrum"])
    span = max(float(np.ptp(reference_spectrum)), 1e-300)
    for name in LADDER_ORDER:
        entry = results[name]
        spectrum = np.asarray(entry["spectrum"])
        entry["heat_peak_rel_error"] = float(
            abs(entry["heat_peak"] - full["heat_peak"])
            / max(abs(full["heat_peak"]), 1e-300))
        entry["ground_rel_error"] = float(
            abs(entry["ground_energy"] - full["ground_energy"])
            / max(abs(full["ground_energy"]), 1e-300))
        shift = (spectrum - spectrum[0]) - (reference_spectrum
                                            - reference_spectrum[0])
        entry["spectrum_rms_rel"] = float(np.sqrt(np.mean(shift**2)) / span)
        entry["weight_rel_error"] = float(
            abs(entry["longitudinal_total_weight"]
                - full["longitudinal_total_weight"])
            / max(abs(full["longitudinal_total_weight"]), 1e-300))

    return {
        "jpm": jpm,
        "cluster": "fcc32",
        "levels": 2,
        "source": cached.name,
        "momentum": [float(x) for x in momentum],
        "norm_budget": norm_budget(f_masked, layers, extras),
        "ladder": results,
        "runtime": time.perf_counter() - started,
    }


def main():
    force = "--force" in sys.argv
    values = [float(v) for v in sys.argv[1:] if not v.startswith("--")]
    if not values:
        found = []
        for path in sorted(TOMO.glob("tomo_fcc32_L2_-0p*.npz")):
            if "_n" in path.stem:
                continue
            with np.load(path) as data:
                found.append(float(data["jpm"]))
        values = sorted(set(found))

    OUTPUT.mkdir(parents=True, exist_ok=True)
    cluster = geometry.build_cluster("fcc", (2, 2, 2))
    labels = polarization_sector_labels(cluster)
    graph_distance = graph_distance_matrix(cluster)
    real_distance = minimum_image_distance_matrix(cluster)
    print(f"fcc32: {cluster.n_ice} ice states; classifying loop layers...",
          flush=True)
    layers, extras = class_layers(cluster, labels, graph_distance,
                                  real_distance)
    for length in PERIMETERS:
        print(f"  simple L={length}: {int(layers[length].sum())} "
              f"mask-surviving entries", flush=True)

    for jpm in values:
        tag = f"{jpm:+.3f}".replace(".", "p")
        destination = OUTPUT / f"ladder_{tag}.json"
        if destination.exists() and not force:
            print(f"jpm={jpm:+.3f} already done, skipping", flush=True)
            continue
        record = analyse(jpm, cluster, labels, layers, extras)
        destination.write_text(json.dumps(record, indent=1, default=float))
        full = record["ladder"]["full"]
        print(f"jpm={jpm:+.3f}  full peak={full['heat_peak']:.5e}", flush=True)
        for name in LADDER_ORDER[:-1]:
            entry = record["ladder"][name]
            print(f"    {name:<16} peak={entry['heat_peak']:.5e} "
                  f"({entry['heat_peak_rel_error']:+8.2%})  "
                  f"gs={entry['ground_rel_error']:.2e}  "
                  f"spec={entry['spectrum_rms_rel']:.2e}  "
                  f"Szz={entry['weight_rel_error']:+8.2%}", flush=True)
        budget = record["norm_budget"]
        print(f"    off-diagonal norm budget: simple="
              f"{budget['simple_fraction']:.3f} of total; "
              f"composite={budget['composite'] / budget['total']:.3f}; "
              f"long={budget['long'] / budget['total']:.3f}", flush=True)
        print(f"    wrote {destination.name} "
              f"({record['runtime']:.1f}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
