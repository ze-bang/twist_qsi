#!/usr/bin/env python3
"""Fill the four (Jx, Jy) columns the array job never wrote.

Tasks 17, 19, 21 and 22 of array 18332139 died -- three on the three-hour
wall, one on ArpackNoConvergence -- leaving 164 of the 1681 grid points
blank at Jx = -0.15, -0.05, +0.05, +0.10.  Almost none of them need a new
solve:

  * Jx <-> Jy is an exact symmetry of the model (it sends Jpmpm -> -Jpmpm),
    measured on the completed part of this very grid at 1.5e-13 on the gap
    and 1.7e-12 on the ice violation.  A blank cell (a, b) whose transpose
    (b, a) sits in a completed column is therefore already known.
  * The dead tasks printed every point they finished before dying, and the
    printed line carries E0, the gap, the degeneracy, the ice weight and
    the ice violation -- everything the figure reads.  That partial work is
    recoverable, at the four-decimal precision of the printout.

Preference order per cell: own JSON > own log line > transposed log line >
transposed JSON.  A direct solve always beats a reflected one, because the
ice weight is an eigenvector quantity and reflection is only exact to
2.9e-3 for it (the gap and the violation reflect to machine precision).

Anything left after both routes is written to remaining_points.json for
scan_xy_hard_points.py, and any solves that script has already deposited in
extra_points_*.json are folded in here.  Provenance is recorded per point,
so nothing reflected or scraped can later be mistaken for a fresh solve.

Usage: fill_xy_phase_gaps.py [n_grid]
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCAN = ROOT / "campaign" / "outputs" / "xy_phase_plane"

LINE = re.compile(
    r"jx=(?P<jx>[-+][\d.]+)\s+jy=(?P<jy>[-+][\d.]+)\s+"
    r"E0=(?P<e0>[-+][\d.]+)\s+gap=(?P<gap>[\d.eE+-]+)\s+"
    r"deg=(?P<deg>\d+)\s+Zice=(?P<zice>[\d.]+)\s+viol=(?P<viol>[\d.]+)")

FIELDS = ("energy", "gap", "degeneracy", "ice_weight", "ice_violation")


def nearest(axis: np.ndarray, value: float) -> int:
    return int(np.argmin(np.abs(axis - value)))


def load_columns(n_grid: int):
    """Completed column files, minus any this script wrote on an earlier run."""
    columns = {}
    for path in sorted(SCAN.glob(f"xyscan_col*_n{n_grid}.json")):
        record = json.loads(path.read_text())
        if record.get("filled"):
            continue
        columns[int(record["column"])] = record
    return columns


def scrape_logs(axis: np.ndarray, solved: set[int]):
    """Points printed by tasks that died before writing their JSON."""
    found = {}
    for path in sorted(SCAN.glob("xyscan_*_*.log")):
        for match in LINE.finditer(path.read_text()):
            column = nearest(axis, float(match["jx"]))
            if column in solved:
                continue          # the JSON for this column is authoritative
            row = nearest(axis, float(match["jy"]))
            found[(row, column)] = {
                "energy": float(match["e0"]),
                "gap": float(match["gap"]),
                "degeneracy": int(match["deg"]),
                "ice_weight": float(match["zice"]),
                "ice_violation": float(match["viol"]),
                "source": f"log:{path.name}",
            }
    return found


def load_extras(axis: np.ndarray):
    """Fresh solves from scan_xy_points.py, whichever job produced them.

    The flux scan covers the whole upper triangle, so once it has run these
    supersede every log scrape: full precision, and reflection then carries
    them into the lower triangle too.
    """
    extras = {}
    for path in sorted(list(SCAN.glob("extra_points_*.json"))
                       + list(SCAN.glob("flux_points_*.json"))
                       + list(SCAN.glob("pilot_flux_*.json"))):
        for point in json.loads(path.read_text())["points"]:
            key = (nearest(axis, point["jy"]), nearest(axis, point["jx"]))
            extras[key] = {**{name: point[name] for name in FIELDS},
                           "source": f"solve:{path.name}"}
    return extras


def main() -> None:
    n_grid = int(sys.argv[1]) if len(sys.argv) > 1 else 41
    axis = np.linspace(-1.0, 1.0, n_grid)
    columns = load_columns(n_grid)
    if not columns:
        raise SystemExit(f"no completed columns under {SCAN}")

    # Precedence, strongest first: a solve at the cell, then the reflection
    # of a solve, then a log line, then the reflection of a log line.  Build
    # the solves first and reflect them BEFORE the scrapes get a chance --
    # once the flux triangle exists, a reflected solve is full precision
    # where a scrape carries four decimals, so letting the scrape win (as an
    # earlier version did) would throw precision away.
    cells: dict[tuple[int, int], dict] = {}
    cells.update(load_extras(axis))
    for column, record in columns.items():
        for point in record["points"]:
            row = nearest(axis, point["jy"])
            cells[(row, column)] = {
                "energy": float(point["levels"][0]),
                "gap": float(point["gap"]),
                "degeneracy": int(point["degeneracy"]),
                "ice_weight": float(point["ice_weight"]),
                "ice_violation": float(point["ice_violation"]),
                "source": "solve:column",
            }

    print(f"{len(columns)} completed columns; "
          f"{n_grid * n_grid - len(cells)} cells blank before reflection")

    def reflect() -> int:
        holes = [(row, column)
                 for column in range(n_grid) for row in range(n_grid)
                 if (row, column) not in cells]
        filled = 0
        for row, column in holes:
            mirror = cells.get((column, row))
            if mirror is None:
                continue
            cells[(row, column)] = {**mirror,
                                    "source": f"transpose<-{mirror['source']}"}
            filled += 1
        return filled

    print(f"reflected {reflect()} solved cells across Jx <-> Jy")
    scraped = 0
    for key, record in scrape_logs(axis, set(columns)).items():
        if key not in cells:
            cells[key] = record
            scraped += 1
    print(f"{scraped} cells from log scrapes, "
          f"{reflect()} more by reflecting those")

    remaining = [(row, column)
                 for column in range(n_grid) for row in range(n_grid)
                 if (row, column) not in cells]
    print(f"{len(remaining)} cells still need a solver")

    # Rewrite every column that has no original solve file -- not merely the
    # ones blank today.  A column filled on an earlier, thinner run leaves a
    # stale file behind, which is how a "0 cells need a solver" report ended
    # up beside a figure still reporting two holes.
    for column in [c for c in range(n_grid) if c not in columns]:
        points = []
        for row in range(n_grid):
            cell = cells.get((row, column))
            if cell is None:
                continue
            jx, jy = float(axis[column]), float(axis[row])
            points.append({
                "jx": jx, "jy": jy,
                "jpm": -(jx + jy) / 4.0, "jpmpm": (jx - jy) / 4.0,
                **cell,
            })
        tag = f"col{column:03d}_n{n_grid}"
        with open(SCAN / f"xyscan_{tag}.json", "w") as handle:
            json.dump({"n_grid": n_grid, "column": column,
                       "jx": float(axis[column]), "filled": True,
                       "points": points}, handle, indent=1, default=float)
        kinds = {}
        for point in points:
            kinds[point["source"].split(":")[0]] = 1 + kinds.get(
                point["source"].split(":")[0], 0)
        print(f"  wrote {tag}: {len(points)}/{n_grid} points {kinds}")

    # Only the upper triangle needs solving: the rest is reflection.
    unique = [(row, column) for row, column in remaining if row >= column]
    with open(SCAN / "remaining_points.json", "w") as handle:
        json.dump({"n_grid": n_grid,
                   "points": [{"jx": float(axis[column]),
                               "jy": float(axis[row])}
                              for row, column in unique]}, handle, indent=1)
    for row, column in unique:
        print(f"  remaining: jx={axis[column]:+.3f} jy={axis[row]:+.3f}")
    print(f"  ({len(remaining)} blank cells = {len(unique)} distinct solves)")

    # The flux is an eigenvector quantity and no vectors were kept, so the
    # flux map has to re-solve regardless; the upper triangle is the whole
    # job, and it returns the four stalled columns at full precision as a
    # by-product, retiring the four-decimal log scrapes above.
    triangle = [{"jx": float(axis[column]), "jy": float(axis[row])}
                for column in range(n_grid) for row in range(column, n_grid)]
    with open(SCAN / "flux_points.json", "w") as handle:
        json.dump({"n_grid": n_grid, "points": triangle}, handle, indent=1)
    print(f"wrote flux_points.json: {len(triangle)} upper-triangle points")


if __name__ == "__main__":
    main()
