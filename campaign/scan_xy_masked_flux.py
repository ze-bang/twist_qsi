#!/usr/bin/env python3
"""Winding-free hexagon flux over the (Jx, Jy) plane, from the mask alone.

No counterterm anywhere in here.  The mask is the whole method: build the
exact Feshbach map F(z) of the ice block, delete every entry connecting
different transport sectors, and solve E = eig F(E).  The ground branch's
eigenvector is the ice-manifold component of the winding-free state, and
the contractible hexagon operator maps ice to ice, so the flux is a 90x90
expectation value.

Why the counterterm exists at all, since it is not needed here: the masked
map returns energies and ice-block vectors but no states above the band and
no wavefunctions on the full space.  When you want those -- thermodynamics,
the spinon sector, observables with Q support -- you need an operator, and
C = -(transport-changing part of F(z_0)) is the operator whose folded map
equals the masked map at the anchor z_0.  For a flux, which lives entirely
inside the ice manifold, the mask by itself is enough and exact.

Both maps are solved at every point:
  masked=True   the winding-free answer;
  masked=False  the same fold with the cross-sector entries left in, which
                is exact for the raw cluster -- so the pair isolates the
                winding contribution inside one framework, with no change
                of basis, solver or normalisation between them.

On the full 65,536-state basis the direct QHQ eigendecomposition is a single
65,446-dimensional dense solve, which overflows LAPACK's int32 indexing
(n^2 > 2^31) and segfaults.  build_blocks_klein routes around it through the
four FCC translations, which commute with H at any (Jpm, Jpmpm).

Usage: scan_xy_masked_flux.py <points.json> <task_id> <n_tasks> <out_prefix>
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
from qsi_campaign.exact_band import (  # noqa: E402
    cubic16_translation_permutations,
    full_spin_basis,
    microscopic_character_hamiltonian,
    translation_sector_bases,
)
from qsi_campaign.downfold import (  # noqa: E402
    build_blocks_klein, solve_roots)

OUTPUT = ROOT / "campaign" / "outputs" / "xy_phase_plane"
OUTPUT.mkdir(parents=True, exist_ok=True)


def hexagon_operator(cluster) -> np.ndarray:
    """Mean of <O_hex + h.c.> over contractible hexagons, in the ice basis.

    A ring flip on a flippable hexagon takes an ice state to an ice state,
    so the operator closes inside the 90-dimensional manifold and is a dense
    90x90 matrix.  Averaging over hexagons matches the convention of
    run_strong_coupling_gs.loop_expectation, so the numbers are comparable
    with the counterterm campaign's cached values.
    """
    ice = np.asarray(cluster.ice_states, dtype=np.uint64)
    hexagons = {tuple(sorted(sites)): sites
                for sites, wind in cluster.hexes if tuple(wind) == (0, 0, 0)}
    operator = np.zeros((ice.size, ice.size))
    for sites in hexagons.values():
        mask = 0
        pattern = 0
        for position, site in enumerate(sites):
            mask |= 1 << int(site)
            if position % 2 == 0:
                pattern |= 1 << int(site)
        other = mask ^ pattern
        occupied = ice & np.uint64(mask)
        active = np.where((occupied == np.uint64(pattern))
                          | (occupied == np.uint64(other)))[0]
        if active.size == 0:
            continue
        partners = np.searchsorted(ice, ice[active] ^ np.uint64(mask))
        if not np.array_equal(ice[partners], ice[active] ^ np.uint64(mask)):
            raise RuntimeError("a hexagon flip left the ice manifold")
        operator[partners, active] += 1.0
    return operator / max(len(hexagons), 1)


DEGEN_TOL = 1e-9


def branch_state(blocks, roots, operator, *, masked: bool):
    """Ground multiplet of the map: flux, ice weight, gap, multiplicity.

    The masked ground state is degenerate by theorem, not by accident -- the
    sector structure protects a multiplicity of six at weak coupling and two
    once it reorganises.  A single eigenvector of a degenerate subspace is
    whatever LAPACK returned, so every expectation value here is averaged
    over the multiplet, matching what the counterterm campaign does.
    """
    if not roots.converged.any():
        return None
    energies = np.where(roots.converged, roots.energies, np.inf)
    ground = float(np.min(energies))
    multiplet = np.where(energies <= ground + DEGEN_TOL)[0]
    above = energies[energies > ground + DEGEN_TOL]
    values, vectors = np.linalg.eigh(blocks.f_matrix(ground, masked=masked))
    fluxes, weights = [], []
    for branch in multiplet:
        vector = vectors[:, branch]
        fluxes.append(float(vector @ operator @ vector))
        weights.append(
            1.0 / (1.0 - blocks.self_energy_slope(ground, vector)))
    return {
        "energy": ground,
        "flux": float(np.mean(fluxes)),
        "flux_spread": float(np.max(fluxes) - np.min(fluxes)),
        "ice_weight": float(np.mean(weights)),
        "degeneracy": int(multiplet.size),
        # gap above the protected multiplet, within the ice-descended band
        "gap": float(above.min() - ground) if above.size else float("nan"),
        "residual": float(values[multiplet[0]] - ground),
        "levels": np.sort(energies[np.isfinite(energies)])[:10].tolist(),
        "n_converged": int(roots.converged.sum()),
    }


def main() -> None:
    arguments = [value for value in sys.argv[1:] if value != "--gpu"]
    gpu = "--gpu" in sys.argv
    points_file, task_id, n_tasks, prefix = (
        arguments[0], int(arguments[1]), int(arguments[2]), arguments[3])
    requested = json.loads(Path(points_file).read_text())["points"]
    mine = sorted(requested[task_id::n_tasks],
                  key=lambda point: (point["jx"], point["jy"]))
    print(f"task {task_id}/{n_tasks}: {len(mine)} of {len(requested)} points",
          flush=True)
    if not mine:
        return

    cluster = geometry.build_cluster("cubic", (1, 1, 1))
    states = full_spin_basis(cluster.n_sites)
    ice_rows = np.searchsorted(states, np.sort(
        np.asarray(cluster.ice_states, dtype=np.uint64)))
    sector_bases = translation_sector_bases(
        states, cubic16_translation_permutations(cluster))
    operator = hexagon_operator(cluster)
    print(f"{len(sector_bases)} Klein irreps; "
          f"hexagon operator {operator.shape}", flush=True)

    # Resume: checkpointed points are kept and skipped, so a task that hit
    # the wall is continued rather than restarted.  This replaces the
    # sbatch's file-exists guard, which with checkpointing would have
    # skipped partially finished tasks forever.
    destination = OUTPUT / f"{prefix}_{task_id:03d}.json"
    records = []
    if destination.exists():
        records = json.loads(destination.read_text())["points"]
        done = {(round(r["jx"], 9), round(r["jy"], 9)) for r in records}
        mine = [p for p in mine
                if (round(p["jx"], 9), round(p["jy"], 9)) not in done]
        print(f"resuming: {len(records)} points already on disk, "
              f"{len(mine)} to go", flush=True)

    for point in mine:
        started = time.perf_counter()
        jx, jy = float(point["jx"]), float(point["jy"])
        jpm, jpmpm = -(jx + jy) / 4.0, (jx - jy) / 4.0
        hamiltonian = microscopic_character_hamiltonian(
            cluster, states, jpm, np.zeros(3), jpmpm=jpmpm)
        entry = {"jx": jx, "jy": jy, "jpm": jpm, "jpmpm": jpmpm}
        try:
            blocks = build_blocks_klein(cluster, states, ice_rows,
                                        hamiltonian, sector_bases, gpu=gpu)
        except RuntimeError as failure:
            # build_blocks_klein's ice-kernel classification can fail to
            # separate spurious modes from genuine poles where extra
            # degeneracy makes the overlaps ambiguous -- seen first on the
            # Jpm = 0 anti-diagonal.  Record the point as unresolved and
            # keep going: one bad point must not cost the task its other
            # forty.
            entry["error"] = str(failure)
            entry["runtime"] = time.perf_counter() - started
            records.append(entry)
            print(f"jx={jx:+.3f} jy={jy:+.3f} jpm={jpm:+.4f} pp={jpmpm:+.4f} "
                  f"UNRESOLVED: {failure}", flush=True)
            with open(destination, "w") as handle:
                json.dump({"points": records}, handle, indent=1, default=float)
            continue
        for masked in (True, False):
            tag = "masked" if masked else "unmasked"
            state = branch_state(blocks, solve_roots(blocks, masked=masked),
                                 operator, masked=masked)
            if state is None:
                # no root below the continuum: the ice-descended band has
                # dissolved here, which is a result and not a failure
                entry[f"{tag}_flux"] = float("nan")
                entry[f"{tag}_energy"] = float("nan")
                entry[f"{tag}_degeneracy"] = 0
                entry[f"{tag}_n_converged"] = 0
                continue
            for name, value in state.items():
                entry[f"{tag}_{name}"] = value
        entry["runtime"] = time.perf_counter() - started
        records.append(entry)
        print(f"jx={jx:+.3f} jy={jy:+.3f} jpm={jpm:+.4f} pp={jpmpm:+.4f} "
              f"masked: E={entry['masked_energy']:+.5f} "
              f"flux={entry['masked_flux']:+.5f}"
              f"(spread {entry.get('masked_flux_spread', float('nan')):.1e}) "
              f"deg={entry.get('masked_degeneracy', 0)} "
              f"Z={entry.get('masked_ice_weight', float('nan')):.4f} "
              f"({entry['masked_n_converged']}/90) | "
              f"unmasked: flux={entry['unmasked_flux']:+.5f} "
              f"({entry['unmasked_n_converged']}/90) "
              f"[{entry['runtime']:.0f}s]", flush=True)
        # Checkpoint every point.  At ~170 s each a task carries hours of
        # work, and writing only at the end means a crash or a wall throws
        # all of it away -- which is exactly what the first launch did.
        with open(destination, "w") as handle:
            json.dump({"points": records}, handle, indent=1, default=float)

    print(f"wrote {destination} ({len(records)} points)", flush=True)


if __name__ == "__main__":
    main()
