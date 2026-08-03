#!/usr/bin/env python3
"""The self-consistent counterterm protocol: every number in the main text.

The counterterm C = -(1 - Delta)[F(z_0)] is supported inside the ice manifold,
so it cannot dress itself and satisfies F_{H+C}(z) = F_H(z) + C at every z.  At
one energy, z = z_0, this makes F_{H+C}(z_0) = Delta[F(z_0)] exactly: the
winding-free effective Hamiltonian is realised as the Feshbach map of a genuine
Hermitian operator on the full Hilbert space.

The anchor is not a free parameter.  It is fixed by the Brillouin-Wigner
condition z_0 = E_0(H + C), and solving that condition *tightly* is the whole
game, because z_0 is the one energy at which the construction is exact.  At the
fixed point the ground multiplet of H + C is an exact eigenvalue multiplet of
the masked map, degeneracy included.

Why that matters, and what it corrects
--------------------------------------
The mask makes Delta[F(z)] block diagonal in the transport label.  On cubic-16
the 90 ice configurations split into 25 transport sectors of dimensions

    6 x 1  +  12 x 2  +  6 x 8  +  1 x 12  =  90,

and symmetry-related blocks carry identical spectra, so the masked map has exact
degeneracies -- verified here to hold in Delta[F(z_0)] to ~1e-14.  An
*under-converged* anchor breaks them, and the splitting produces a spurious
low-temperature Schottky anomaly and a violated entropy sum rule.

An earlier version of this campaign iterated the anchor at most three times,
reaching a residual of only ~1e-2 at J_pm/J_zz = -0.20, and concluded that the
splitting was an intrinsic "anchoring residue" to be repaired by averaging the
six lowest levels.  That conclusion was wrong.  The splitting is proportional
to the anchor residual (measured in run_anchor_convergence.py: it falls from
4.0e-2 to 1.2e-12 as the residual falls from 6.2e-2 to 4.9e-15), so it is a
convergence artifact, not a property of the construction.  Plain iteration
converges linearly at rate ~(1 - Z) and stalls; a secant step reaches machine
precision in about ten solves.

With the anchor converged there is nothing to repair.  This driver therefore
reports the unmodified spectrum as the result, and also computes the
symmetrised variant in order to show that it is *worse*: forcing the upper band
onto Delta[F(z_0)]'s degeneracy pattern degrades the ring anomaly from 0.5% to
2.7% of the reference at -0.20, because Delta[F(z_0)] is itself inaccurate away
from z_0.  The overlap purity reported below quantifies that: H + C's band
states are only ~57% aligned with Delta[F(z_0)]'s eigenvectors at -0.20.

Everything is checked rather than assumed: anchor self-consistency, the
exactness of the protected degeneracies, the ground multiplicity actually
realised, the separation of the ice band from the continuum, the entropy sum
rule, temperature-grid refinement, and -- where the exact masked reference
exists -- the deviation of the band and of the ring anomaly from it.

Usage: run_counterterm_spine.py [--jpm a,b,c]
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.linalg import eigh, eigvalsh
from scipy.sparse.linalg import eigsh

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "notes"))
sys.path.insert(0, str(ROOT / "campaign"))

import recompute_finite_size_artifact as geometry  # noqa: E402
from qsi_campaign.exact_band import (  # noqa: E402
    cubic16_translation_permutations,
    fixed_magnetization_basis,
    microscopic_character_hamiltonian,
    translation_sector_bases,
)
from qsi_campaign.protocol import polarization_sector_labels  # noqa: E402
from run_winding_residual import feshbach_matrix  # noqa: E402
from run_exact_counterterm import embed  # noqa: E402
from analyse_peak_labels import label_peaks  # noqa: E402

CACHE = ROOT / "campaign" / "outputs" / "method_comparison"
OUTPUT = ROOT / "campaign" / "outputs" / "counterterm_spine"
OUTPUT.mkdir(parents=True, exist_ok=True)

N_SITES = 16
N_ICE = 90
GRID = np.geomspace(1e-7, 4.0, 3000)
FINE = np.geomspace(1e-7, 4.0, 6000)
SGRID = np.geomspace(1e-8, 60.0, 6000)
DEGENERACY_TOL = 1.0e-9
ANCHOR_TOL = 1.0e-13


# --------------------------------------------------------------- thermodynamics
def heat_capacity(levels, grid=GRID):
    energies = np.sort(np.asarray(levels, dtype=float))
    energies = energies - energies[0]
    beta = 1.0 / grid[:, None]
    weight = np.exp(-beta * energies[None, :])
    partition = weight.sum(axis=1)
    mean = (weight * energies[None, :]).sum(axis=1) / partition
    second = (weight * energies[None, :] ** 2).sum(axis=1) / partition
    return (second - mean * mean) / (N_SITES * grid**2)


def entropy_curve(levels, grid=SGRID):
    """S(T)/N by integrating C/T up from the bottom, plus the frozen ln g_0."""
    energies = np.sort(np.asarray(levels, dtype=float))
    energies = energies - energies[0]
    degeneracy = int(np.sum(energies < DEGENERACY_TOL))
    curve = heat_capacity(energies, grid)
    integrand = curve / grid
    cumulative = np.concatenate(
        [[0.0], np.cumsum(0.5 * (integrand[1:] + integrand[:-1]) * np.diff(grid))]
    )
    return cumulative + np.log(degeneracy) / N_SITES, degeneracy


# ------------------------------------------------------- protected multiplets
def protected_partition(matrix, labels):
    """Multiplet partition of the masked map, read off its block structure.

    Returns the eigenvalues of Delta[F(z_0)] sorted, the sector each came
    from, the group index of each level, and the largest within-group spread.
    The spread is the check: the mask makes symmetry-related blocks share a
    spectrum exactly, so it must vanish to machine precision.
    """
    entries = []
    for sector in np.unique(labels):
        rows = np.flatnonzero(labels == sector)
        block = matrix[np.ix_(rows, rows)]
        block_values, block_vectors = eigh(block)
        for column, value in enumerate(block_values):
            vector = np.zeros(len(labels))
            vector[rows] = block_vectors[:, column]
            entries.append((float(value), int(sector), int(len(rows)),
                            vector))
    entries.sort(key=lambda item: item[0])
    values = np.asarray([item[0] for item in entries])
    sectors = np.asarray([item[1] for item in entries])
    dims = np.asarray([item[2] for item in entries])
    frame = np.column_stack([item[3] for item in entries])

    scale = max(float(np.ptp(values)), 1.0e-12)
    groups = np.zeros(len(values), dtype=int)
    current = 0
    for index in range(1, len(values)):
        if values[index] - values[index - 1] > DEGENERACY_TOL * max(1.0, scale):
            current += 1
        groups[index] = current
    spread = max(
        float(np.ptp(values[groups == g])) for g in range(groups.max() + 1)
    )
    return values, sectors, dims, groups, spread, frame


def symmetrise(levels, groups):
    """Average within each protected multiplet.  Trace preserving by design."""
    out = np.asarray(levels, dtype=float).copy()
    for group in range(groups.max() + 1):
        mask = groups == group
        if mask.sum() > 1:
            out[mask] = out[mask].mean()
    return out


def periodic_zero_spectrum(cluster, jpm, sectors):
    """The uncorrected S^z = 0 spectrum, for couplings with no cached run."""
    states = fixed_magnetization_basis(cluster.n_sites, cluster.n_sites // 2)
    hamiltonian = microscopic_character_hamiltonian(
        cluster, states, jpm, np.zeros(3))
    return np.sort(np.concatenate([
        eigvalsh(np.ascontiguousarray(
            (basis.conj().T @ (hamiltonian @ basis)).toarray().real))
        for basis in sectors.values()]))


def other_magnetization_spectrum(cluster, jpm, permutations):
    """Every magnetization sector except S^z_tot = 0, where C does not act."""
    half = cluster.n_sites // 2
    levels = []
    for n_up in range(cluster.n_sites + 1):
        if n_up == half:
            continue
        block_states = fixed_magnetization_basis(cluster.n_sites, n_up)
        hamiltonian = microscopic_character_hamiltonian(
            cluster, block_states, jpm, np.zeros(3))
        if len(block_states) < 8:
            levels.append(eigvalsh(np.asarray(hamiltonian.todense()).real))
            continue
        for basis in translation_sector_bases(
                block_states, permutations).values():
            block = (basis.conj().T @ (hamiltonian @ basis)).toarray()
            levels.append(eigvalsh(np.ascontiguousarray(
                (0.5 * (block + block.conj().T)).real)))
    return np.concatenate(levels)


# ------------------------------------------------------------------- the spine
def main() -> None:
    requested = None
    if "--jpm" in sys.argv:
        requested = [float(v) for v in
                     sys.argv[sys.argv.index("--jpm") + 1].split(",")]

    cluster = geometry.build_cluster("cubic", (1, 1, 1))
    permutations = cubic16_translation_permutations(cluster)
    states = fixed_magnetization_basis(cluster.n_sites, cluster.n_sites // 2)
    index = {int(s): i for i, s in enumerate(states)}
    ice_rows = np.asarray([index[int(s)] for s in cluster.ice_states])
    outside = np.ones(len(states), dtype=bool)
    outside[ice_rows] = False
    complement = np.where(outside)[0]
    sectors = translation_sector_bases(states, permutations)
    labels = polarization_sector_labels(cluster)
    cross = labels[:, None] != labels[None, :]

    sector_dims = np.unique(labels, return_counts=True)[1]
    print(f"transport sectors: {len(sector_dims)}, dimensions "
          f"{np.unique(sector_dims, return_counts=True)}")

    # Couplings come from the cached comparison runs by default, but a coupling
    # with no cache is allowed: the periodic spectrum is then computed here and
    # there is simply no masked reference to compare against.
    cached = {round(float(np.load(p)["jpm"]), 6): p
              for p in CACHE.glob("compare_*.npz")}
    if requested is None:
        couplings = sorted(cached, reverse=True)
    else:
        couplings = sorted((round(v, 6) for v in requested), reverse=True)

    records = []
    for jpm in couplings:
        started = time.perf_counter()
        path = cached.get(round(jpm, 6))
        if path is not None:
            data = np.load(path)
            periodic = np.sort(data["periodic_levels"])
            reference = (np.sort(data["feshbach_levels"])[:N_ICE]
                         if "feshbach_levels" in data else None)
        else:
            periodic = np.sort(np.concatenate([
                periodic_zero_spectrum(cluster, jpm, sectors),
                other_magnetization_spectrum(cluster, jpm, permutations)]))
            reference = None

        hamiltonian = microscopic_character_hamiltonian(
            cluster, states, jpm, np.zeros(3)).tocsc().real

        # --- the anchor.  The Brillouin-Wigner condition z_0 = E_0(H + C) is
        # the defining equation, and it matters that it be solved tightly: the
        # counterterm is exact at z_0, so the ground multiplet inherits the
        # masked map's protected degeneracy only to the extent that z_0 has
        # converged.  Plain iteration converges linearly at rate ~(1 - Z) and
        # stalls near 10^-7; a secant step on g(z) = E_0(H + C(z)) - z reaches
        # machine precision in a handful of solves.  Only the ground energy is
        # needed inside the loop, so it is obtained sparsely.
        def ground_energy(anchor):
            f_matrix = feshbach_matrix(hamiltonian, ice_rows, complement,
                                       anchor)
            corrected = (hamiltonian + embed(-np.where(cross, f_matrix, 0.0),
                                             ice_rows, len(states))).tocsc()
            ground = float(eigsh(corrected, k=1, which="SA", tol=1e-14,
                                 return_eigenvectors=False)[0])
            return ground, f_matrix, corrected

        z_previous = float(eigsh(hamiltonian, k=1, which="SA", tol=1e-14,
                                 return_eigenvectors=False)[0])
        ground, f_matrix, corrected = ground_energy(z_previous)
        g_previous = ground - z_previous
        z_current = ground
        history = [abs(g_previous)]
        for iteration in range(24):
            ground, f_matrix, corrected = ground_energy(z_current)
            g_current = ground - z_current
            history.append(abs(g_current))
            if abs(g_current) < ANCHOR_TOL:
                break
            denominator = g_current - g_previous
            step = (-g_current * (z_current - z_previous) / denominator
                    if abs(denominator) > 1e-16 else g_current)
            z_previous, g_previous = z_current, g_current
            z_current = float(z_current + step)
        anchor = z_current
        self_consistency = history[-1]
        masked_block = np.where(cross, 0.0, f_matrix)

        # --- the full S^z = 0 spectrum of H + C, translation-blocked, with the
        # eigenvectors of the lowest N_ICE states: they are what identifies the
        # multiplets.
        blocks = []
        for basis in sectors.values():
            dense = np.ascontiguousarray(
                (basis.conj().T @ (corrected @ basis)).toarray().real)
            blocks.append((basis, dense, eigvalsh(dense)))
        zero_levels = np.sort(np.concatenate([b[2] for b in blocks]))
        cut = zero_levels[N_ICE - 1]
        collected = []
        for basis, dense, block_values in blocks:
            take = int(np.sum(block_values <= cut))
            if take == 0:
                continue
            sub_values, sub_vectors = eigh(
                dense, subset_by_index=(0, take - 1))
            ice_rows_basis = np.asarray(basis[ice_rows].todense())
            amplitudes = ice_rows_basis @ sub_vectors      # N_ICE x take
            collected.extend((float(sub_values[i]), amplitudes[:, i])
                             for i in range(take))
        collected.sort(key=lambda item: item[0])
        low_ice = np.column_stack([item[1] for item in collected[:N_ICE]])
        rest = other_magnetization_spectrum(cluster, jpm, permutations)
        raw_full = np.sort(np.concatenate([zero_levels, rest]))

        # --- the protected multiplet structure, and the check that it is exact
        values, sec, dims, groups, spread, frame = protected_partition(
            masked_block, labels)

        # --- the ice band of H + C, and the check that it is a band
        band = zero_levels[:N_ICE]
        band_gap = float(zero_levels[N_ICE] - zero_levels[N_ICE - 1])
        band_width = float(band[-1] - band[0])

        # --- assign each band state to a protected multiplet by overlap, not
        # by energy order.  Summing |<phi|v>|^2 over a degenerate group makes
        # the weight independent of the basis chosen inside that group, so the
        # assignment is well defined even where multiplets have drawn level.
        ice_amplitudes = np.asarray(low_ice).real
        norms = np.linalg.norm(ice_amplitudes, axis=0)
        ice_amplitudes = ice_amplitudes / np.where(norms > 0, norms, 1.0)
        overlap = (frame.T @ ice_amplitudes) ** 2                # 90 x 90
        n_groups = int(groups.max() + 1)
        weight = np.stack([overlap[groups == g].sum(axis=0)
                           for g in range(n_groups)])            # groups x 90
        assigned = np.argmax(weight, axis=0)
        purity = float(np.min(weight.max(axis=0) / weight.sum(axis=0)))
        sizes_ok = all(int(np.sum(assigned == g)) == int(np.sum(groups == g))
                       for g in range(n_groups))
        ice_content = float(np.min(norms))

        # The ground multiplet is the one the anchor makes exact; report its
        # splitting separately from the worst multiplet anywhere in the band.
        ground_multiplicity = int(np.sum(groups == 0))
        ground_splitting = float(np.ptp(band[:ground_multiplicity]))
        symmetrised_band = symmetrise(band, groups)
        splitting = max(float(np.ptp(band[groups == g]))
                        for g in range(n_groups))

        # How far the overlap assignment differs from naive energy ordering.
        order_agreement = float(np.mean(assigned == groups))
        centres = np.asarray([band[assigned == g].mean()
                              for g in range(n_groups) if np.any(assigned == g)])
        between = float(np.min(np.diff(np.sort(centres)))) \
            if len(centres) > 1 else float("inf")

        symmetrised_full = np.sort(np.concatenate([symmetrised_band,
                                                   zero_levels[N_ICE:], rest]))

        # --- the 90 x 90 band operator the DSSF needs.  The eigenvectors come
        # from Delta[F(z_0)], whose block structure is exact, and the
        # eigenvalues from the repaired counterterm spectrum.  No frame is
        # involved: this operator lives in the ice configuration basis, so a
        # diagonal spin source is projected onto it exactly.
        band_operator = (frame * (band - band[0])) @ frame.T
        band_operator = 0.5 * (band_operator + band_operator.T)
        frame_error = float(np.linalg.norm(frame.T @ frame - np.eye(N_ICE)))

        # --- thermodynamics, before and after, with checks
        record = {"jpm": jpm, "g4": 4.0 * jpm * jpm,
                  "g6": 12.0 * abs(jpm) ** 3, "anchor": anchor,
                  "self_consistency": self_consistency,
                  "protected_spread": spread,
                  "n_multiplets": int(groups.max() + 1),
                  "ground_multiplet": int(np.sum(groups == 0)),
                  "band_gap": band_gap, "band_width": band_width,
                  "residual_splitting": splitting,
                  "splitting_over_width": splitting / band_width,
                  "ground_multiplicity": ground_multiplicity,
                  "ground_splitting": ground_splitting,
                  "ground_splitting_over_width": ground_splitting / band_width,
                  "spurious_schottky_T": ground_splitting / 2.4,
                  "anchor_iterations": len(history),
                  "multiplet_spacing": between,
                  "frame_orthonormality_error": 0.0,
                  "partition_margin": splitting / between,
                  "assignment_purity": purity,
                  "assignment_sizes_ok": sizes_ok,
                  "order_agreement": order_agreement,
                  "min_ice_content": ice_content,
                  "n_roots_reference": (int(len(reference))
                                        if reference is not None else 0)}

        curves = {}
        for tag, levels in (("counterterm", raw_full),
                            ("symmetrised", symmetrised_full),
                            ("periodic", periodic)):
            curve = heat_capacity(levels)
            fine = heat_capacity(levels, FINE)
            entropy, degeneracy = entropy_curve(levels)
            peaks = label_peaks(levels, has_sectors=(tag != "periodic"))
            peaks_fine = label_peaks(levels, grid=FINE,
                                     has_sectors=(tag != "periodic"))
            curves[tag] = curve
            curves[f"{tag}_entropy"] = entropy
            record[f"{tag}_peaks"] = peaks
            record[f"{tag}_entropy_total"] = float(entropy[-1])
            record[f"{tag}_entropy_error"] = float(entropy[-1] - np.log(2.0))
            record[f"{tag}_ground_degeneracy"] = degeneracy
            named = {p["label"]: p["T"] for p in peaks}
            named_fine = {p["label"]: p["T"] for p in peaks_fine}
            for name in ("sector", "ring", "gauge", "spinon"):
                record[f"{tag}_{name}"] = named.get(name)
                if name in named and name in named_fine:
                    record[f"{tag}_{name}_grid_shift"] = abs(
                        named_fine[name] / named[name] - 1.0)
            record[f"{tag}_n_peaks"] = len(peaks)

        # --- deviation from the exact masked reference, where it exists
        if reference is not None and len(reference) == N_ICE:
            ref_peaks = label_peaks(
                np.sort(np.concatenate([reference, raw_full[N_ICE:]])))
            ref_ring = next((p["T"] for p in ref_peaks
                             if p["label"] == "ring"), None)
            record["reference_ring"] = ref_ring
            record["reference_band_deviation"] = float(
                np.max(np.abs(band - reference)) / band_width)
            record["reference_ground_deviation"] = float(
                abs(band[0] - reference[0]))
            for tag in ("counterterm", "symmetrised"):
                value = record[f"{tag}_ring"]
                record[f"{tag}_ring_deviation"] = (
                    abs(value / ref_ring - 1.0)
                    if (ref_ring and value) else None)

        record["frame_orthonormality_error"] = frame_error
        record["iterations"] = len(history) - 1
        record["runtime"] = time.perf_counter() - started
        records.append(record)

        tag = f"{jpm:+.3f}".replace(".", "p")
        np.savez_compressed(
            OUTPUT / f"spine_{tag}.npz", jpm=jpm, grid=GRID, sgrid=SGRID,
            band=band, band_symmetrised=symmetrised_band,
            band_operator=band_operator, frame=frame, assigned=assigned,
            masked_block=masked_block, masked_values=values,
            masked_sectors=sec, masked_dims=dims, groups=groups,
            levels=raw_full, levels_symmetrised=symmetrised_full,
            levels_periodic=periodic,
            reference_band=(reference if reference is not None
                            else np.zeros(0)),
            **curves)
        with open(OUTPUT / f"spine_{tag}.json", "w") as handle:
            json.dump(record, handle, indent=1, default=float)

        ring = record["counterterm_ring"]
        deviation = record.get("counterterm_ring_deviation")
        deviation_sym = record.get("symmetrised_ring_deviation")
        show = lambda v: '-' if v is None else f'{v*100:.2f}%'
        print(f"jpm={jpm:+.3f} anchor={anchor:+.9f} in {len(history)} steps "
              f"(res {self_consistency:.1e}) | g0={ground_multiplicity} "
              f"split={ground_splitting:.2e} "
              f"({ground_splitting/band_width:.1e} of width) "
              f"-> spurious T={record['spurious_schottky_T']:.1e} | "
              f"peaks={record['counterterm_n_peaks']} "
              f"ring={ring if ring is None else round(ring, 7)} "
              f"dev={show(deviation)} (sym {show(deviation_sym)}) | "
              f"S/N={record['counterterm_entropy_total']:.6f} "
              f"g0_meas={record['counterterm_ground_degeneracy']} "
              f"protected={spread:.1e} purity={purity:.3f} "
              f"({record['runtime']:.0f}s)", flush=True)

    records.sort(key=lambda r: r["jpm"])
    with open(OUTPUT / "spine_summary.json", "w") as handle:
        json.dump(records, handle, indent=1, default=float)
    print(f"\nwrote {len(records)} couplings to {OUTPUT}")
    print(f"ln 2 = {np.log(2):.9f}")


if __name__ == "__main__":
    main()
