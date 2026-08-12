#!/usr/bin/env python3
"""Is there a phase transition anywhere in the cubic-16 (Jpm, Jpmpm) plane?

Two spectra have to be asked separately, and they answer differently.

The *masked* band can reorganise which transport sector owns its bottom --
on the XXZ line that is the 6->2 crossing at Jpm = -0.25.  Transport sectors
are topological (flux) sectors, not symmetry sectors of a local order
parameter, so a change of owner is flux-sector energetics, not an ordering
transition.

The *raw* 65,536-state spectrum is the physical one.  A genuine finite-size
precursor of a transition would show there: a ground-state level crossing,
a degeneracy change, or a kink in E0 along a cut.  On the XXZ line the raw
ground state is known to evolve smoothly to Jpm = -1 with no crossing; this
script asks the same of the Jpmpm direction, where it has never been checked.

Reports per cell: raw E0, its degeneracy and gap, the masked band's ground
degeneracy and sector content; then, along each Jpmpm row and Jpm column,
the second difference of E0 (a kink diagnostic) normalised by the local
scale.  Smooth curvature is O(1); a level crossing spikes it.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PLANE = ROOT / "campaign" / "outputs" / "jpmpm_fold_plane"
BANDS = ROOT / "campaign" / "outputs" / "masked_band_complete"

JPMS = [-0.05, -0.09, -0.125, -0.16, -0.20, -0.24, -0.27, -0.30]
PPS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
DEGEN_TOL = 1e-9


def band_record(jpm: float, jpmpm: float):
    stem = f"{jpm:+.3f}".replace(".", "p")
    name = (f"band_{stem}.json" if abs(jpmpm) < 1e-12
            else f"band_{stem}_pp{jpmpm:+.3f}".replace(".", "p") + ".json")
    path = BANDS / name
    if not path.exists():
        return None
    record = json.loads(path.read_text())
    levels = np.asarray([np.nan if v is None else v
                         for v in record["levels"]], float)
    sectors = np.asarray(record["sector"], int)
    good = np.isfinite(levels)
    levels, sectors = levels[good], sectors[good]
    if not len(levels):
        return None
    ground = levels <= levels.min() + DEGEN_TOL
    return {"deg": int(ground.sum()),
            "sectors": sorted(set(sectors[ground].tolist()))}


def main() -> None:
    energy = np.full((len(PPS), len(JPMS)), np.nan)
    print(f"{'Jpm':>7} {'Jpmpm':>6} {'raw E0':>12} {'rawdeg':>7} "
          f"{'raw gap':>10} {'bandsdeg':>9} {'band sectors'}")
    for row, jpmpm in enumerate(PPS):
        for col, jpm in enumerate(JPMS):
            tag = f"jpm{jpm:+.4f}_pp{jpmpm:+.4f}".replace(".", "p")
            path = PLANE / f"point_{tag}_bordered.npz"
            if not path.exists():
                continue
            with np.load(path) as handle:
                raw = np.sort(np.asarray(handle["raw_levels"], float))
            energy[row, col] = raw[0]
            ground = raw <= raw[0] + DEGEN_TOL
            above = raw[~ground]
            gap = float(above.min() - raw[0]) if len(above) else np.nan
            band = band_record(jpm, jpmpm)
            print(f"{jpm:7.3f} {jpmpm:6.2f} {raw[0]:12.6f} "
                  f"{int(ground.sum()):7d} {gap:10.6f} "
                  f"{(band['deg'] if band else -1):9d} "
                  f"{band['sectors'] if band else '-'}")
        print(flush=True)

    # kink diagnostic: |second difference| of E0 relative to the mean
    # |second difference| along the same cut.  A smooth curve gives O(1);
    # a level crossing in the ground state gives a spike.
    print("\n=== kink diagnostic on the raw ground energy ===")
    for label, matrix, axis_values, other in (
            ("along Jpmpm (fixed Jpm)", energy.T, PPS, JPMS),
            ("along Jpm (fixed Jpmpm)", energy, JPMS, PPS)):
        print(f"\n{label}:")
        for index, fixed in enumerate(other):
            line = matrix[index]
            if np.count_nonzero(np.isfinite(line)) < 3:
                continue
            second = np.abs(np.diff(line, 2))
            scale = np.nanmean(second)
            worst = int(np.nanargmax(second))
            print(f"  fixed={fixed:+.3f}  max|d2E0| / mean = "
                  f"{second[worst] / scale:5.2f}  at "
                  f"{axis_values[worst + 1]:+.3f}"
                  + ("   <-- spike" if second[worst] / scale > 4.0 else ""))


if __name__ == "__main__":
    main()
