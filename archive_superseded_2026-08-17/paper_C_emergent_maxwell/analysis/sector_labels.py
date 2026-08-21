"""Join FCC-32 electric-flux sector indices to their transport vectors T,
then test the emergent-electrostatics law  dE(T) = |T|^2 / (2 eps_eff L).

This is the missing step of Paper C.  Everything it needs is pure combinatorics
on the 2970 ice configurations plus the already-computed sector ground energies
in campaign/outputs/fcc32_sector_grounds/.

It prints, per coupling and per virtual depth:

  * the shell table (multiplicity, ice dimension, |T|^2, dE)
  * the ISOTROPY test: do distinct point-group stars sharing |T|^2 have equal
    dE?  (Eq. (law) requires yes; the lattice symmetry alone does not.)
  * the QUADRATICITY test: least squares of dE against |T|^2 restricted to the
    n_shells smallest |T|, and the residual when all shells are included.
  * eps_eff from the restricted fit.

Run through the scheduler, never on a login node:
    sbatch analysis/run_sector_labels.sbatch
"""

import glob
import json
import os
import sys

import numpy as np

ROOT = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "notes"))

from recompute_finite_size_artifact import build_cluster        # noqa: E402
from qsi_campaign.protocol import polarization_sector_labels    # noqa: E402

GROUNDS = os.path.join(ROOT, "campaign", "outputs", "fcc32_sector_grounds")
# Number of smallest-|T| shells used for the Maxwell fit.  The draft argues the
# law can only hold in a weak-field window; make that window explicit here.
N_SHELLS_FIT = 3


def sector_transport(cluster):
    """Return (labels, T_by_sector) with T = 2p in lattice units.

    Reproduces protocol.polarization_sector_labels exactly (labels index the
    distinct polarizations in sorted order of the rounded tuple) and returns
    the polarization vector attached to each label.
    """
    labels = polarization_sector_labels(cluster)
    states = np.asarray(cluster.ice_states)
    n_sites = int(cluster.n_sites)
    bits = np.array([[(int(s) >> site) & 1 for site in range(n_sites)]
                     for s in states])
    pol = (bits - 0.5) @ np.asarray(cluster.positions, dtype=float)

    n_sectors = int(labels.max()) + 1
    T = np.zeros((n_sectors, pol.shape[1]))
    for lab in range(n_sectors):
        rows = pol[labels == lab]
        spread = np.ptp(rows, axis=0).max() if len(rows) > 1 else 0.0
        if spread > 1e-9:
            raise SystemExit("sector %d has non-constant polarization "
                             "(spread %.2e) -- label definition changed"
                             % (lab, spread))
        T[lab] = 2.0 * rows[0]
    return labels, T


def shells(T2, energies, tol=1e-9):
    """Group sectors by |T|^2; return list of (T2, mult, mean dE, spread)."""
    out = []
    for value in sorted(set(np.round(T2, 6))):
        sel = np.abs(T2 - value) < 1e-6
        e = energies[sel]
        out.append((value, int(sel.sum()), float(e.mean()),
                    float(e.max() - e.min())))
    return out


def analyse(path, T2, dims):
    with open(path) as fh:
        data = json.load(fh)
    n = len(data["sector_grounds"])
    energies = np.array([data["sector_grounds"][str(i)] for i in range(n)])
    dE = energies - energies.min()

    print("=" * 74)
    print("%s   Jpm/Jzz = %+0.3f   depth = %d   space = %d"
          % (os.path.basename(path), data["jpm"], data["levels"],
             data["space"]))
    print("=" * 74)

    # --- energy shells (what the raw data already shows) -------------------
    print("\nenergy shells (grouped by dE):")
    print("  mult   dim        dE/Jzz        |T|^2 values in the group")
    for value in sorted(set(np.round(dE, 12))):
        sel = np.abs(dE - value) < 1e-12
        t2s = sorted(set(np.round(T2[sel], 4)))
        print("  %4d  %4d   %12.9f     %s"
              % (sel.sum(), int(dims[sel][0]), value,
                 ", ".join("%.4f" % v for v in t2s)))

    # --- ISOTROPY: same |T|^2 must mean same dE ---------------------------
    print("\nisotropy test (Eq. law requires spread = 0 within each |T|^2):")
    worst = 0.0
    for value, mult, mean, spread in shells(T2, dE):
        worst = max(worst, spread)
        flag = "  <-- VIOLATION" if spread > 1e-9 else ""
        print("  |T|^2 = %8.4f   mult %3d   dE = %12.9f   spread %.2e%s"
              % (value, mult, mean, spread, flag))
    print("  worst spread over all |T|^2 groups: %.3e" % worst)

    # --- QUADRATICITY and eps_eff ----------------------------------------
    sh = [s for s in shells(T2, dE) if s[0] > 1e-9]
    sh.sort(key=lambda s: s[0])
    if not sh:
        print("\nno nonzero-flux shells found")
        return
    x = np.array([s[0] for s in sh])
    y = np.array([s[2] for s in sh])

    for label, k in (("weak-field window", min(N_SHELLS_FIT, len(sh))),
                     ("all shells", len(sh))):
        slope = float(np.dot(x[:k], y[:k]) / np.dot(x[:k], x[:k]))
        resid = y[:k] - slope * x[:k]
        rel = np.abs(resid).max() / max(y[:k].max(), 1e-300)
        print("\n%s (%d shells): dE = %.6e * |T|^2" % (label, k, slope))
        print("  max relative residual %.3f" % rel)
        print("  1/(2 eps_eff L) = %.6e   ->  eps_eff * L = %.4f"
              % (slope, 1.0 / (2.0 * slope)))
        if label == "weak-field window" and rel > 0.1:
            print("  WARNING: even the weak-field window is not quadratic; "
                  "do not quote eps_eff from this coupling.")


def main():
    cluster = build_cluster("fcc", (2, 2, 2))
    labels, T = sector_transport(cluster)
    T2 = np.sum(T ** 2, axis=1)
    dims = np.array([int((labels == i).sum()) for i in range(len(T))])

    print("FCC-32: %d sites, %d ice states, %d transport sectors"
          % (cluster.n_sites, len(cluster.ice_states), len(T)))
    print("distinct |T|^2 values: %d" % len(set(np.round(T2, 6))))
    print("sum of sector dims: %d" % dims.sum())

    for path in sorted(glob.glob(os.path.join(GROUNDS, "grounds_*.json"))):
        analyse(path, T2, dims)


if __name__ == "__main__":
    main()
