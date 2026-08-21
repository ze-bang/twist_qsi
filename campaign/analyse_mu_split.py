"""Split mu into (dressing of intermediates) x (ground-state shift).

The suppression K_6 = g_6 mu^2 can only come from two places in

    K_6(z) = <b| V Q (z - QHQ)^-1 Q V |a>,

namely the anchor z and the dressing of QHQ by V.  Leading perturbation theory
takes z -> 0 (unperturbed ice energy) AND QHQ -> QH_0Q.  Setting only the
anchor back to zero separates them:

    mu^2 = K_6(z0)/g_6 = [K_6(0)/g_6] * [K_6(z0)/K_6(0)]
                          ^dressing      ^ground-state shift

  dressing ~ 1  -> the whole effect is the z0 shift.  That shift is EXTENSIVE
                   (z0 scales with volume), i.e. a Brillouin-Wigner
                   size-inconsistency, and the loop-renormalization claim dies.
  shift ~ 1     -> the effect lives in the dressed intermediate spectrum and is
                   a genuine property of the theory.

Evaluated with the exact spectral data (poles E_k and couplings of QHQ) so
K_6(z) is a free sum once the fold is built.  z=0 is only meaningful while the
Q spectrum stays positive; on cubic-16 that holds to x ~ 0.10-0.12, which is
already a factor ~2.8 of suppression.  The script reports E_min and refuses to
quote K(0) once the spectrum straddles zero.

Also reports the same split for an 8-spin pair (unmasked on cubic-16, where all
L>=8 loops wind) to see whether the two factors depend on perimeter -- which is
what the hexagon-protection question needs.

Usage: analyse_mu_split.py <jpm> [<jpm> ...]
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.sparse.linalg import eigsh

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "notes"))
sys.path.insert(0, str(ROOT / "campaign"))

import recompute_finite_size_artifact as geometry              # noqa: E402
from qsi_campaign.exact_band import (                          # noqa: E402
    fixed_magnetization_basis, microscopic_character_hamiltonian)
from qsi_campaign.downfold import build_blocks                 # noqa: E402
from qsi_campaign.protocol import polarization_sector_labels   # noqa: E402
from run_feshbach_anatomy import string_classes                # noqa: E402

OUT = ROOT / "campaign" / "outputs" / "map_normalization"


def find_pair(ice_states, table, labels, want):
    states = [int(s) for s in ice_states]
    for i in range(len(states)):
        for j in range(i + 1, len(states)):
            if labels[i] != labels[j]:
                continue
            d = states[i] ^ states[j]
            if want == 6 and table.get(d) == "hexagon":
                return i, j
            if want == 8 and table.get(d) is None and bin(d).count("1") == 8:
                return i, j
    return None


def amplitude(blocks, i, j, z):
    w = np.asarray(blocks.coupling)[:, i] * np.asarray(blocks.coupling)[:, j]
    return float(np.sum(w / (z - np.asarray(blocks.poles))))


def main():
    couplings = [float(v) for v in sys.argv[1:]]
    cluster = geometry.build_cluster("cubic", (1, 1, 1))
    states = fixed_magnetization_basis(cluster.n_sites, cluster.n_sites // 2)
    index = {int(s): k for k, s in enumerate(states)}
    ice = np.asarray(cluster.ice_states, dtype=np.uint64)
    ice_idx = np.asarray([index[int(s)] for s in ice])
    labels = polarization_sector_labels(cluster)
    table = string_classes(cluster)
    pairs = {6: find_pair(ice, table, labels, 6),
             8: find_pair(ice, table, labels, 8)}
    print("pairs:", pairs, flush=True)

    recs = []
    for jpm in couplings:
        t0 = time.perf_counter()
        ham = microscopic_character_hamiltonian(cluster, states, jpm,
                                                np.zeros(3))
        z0 = float(eigsh(ham.real.tocsc(), k=1, which="SA", tol=1e-11, ncv=80,
                         return_eigenvectors=False)[0])
        blocks = build_blocks(cluster, states, ice_idx, ham)
        emin = float(np.min(blocks.poles))
        safe = emin > 0.02
        rec = {"jpm": jpm, "z0": z0, "E_min": emin, "z0_safe": safe,
               "runtime": time.perf_counter() - t0}
        print("\njpm=%+0.3f  z0=%+.5f  E_min(QHQ)=%+.4f  %s"
              % (jpm, z0, emin,
                 "" if safe else "  <-- Q spectrum straddles 0: K(0) invalid"),
              flush=True)
        for L, pair in pairs.items():
            if pair is None:
                continue
            k_z0 = amplitude(blocks, *pair, z0)
            k_0 = amplitude(blocks, *pair, 0.0) if safe else float("nan")
            # leading perturbative amplitude for this perimeter
            n = L // 2
            pert = {6: 12.0, 8: 40.0}[L] * abs(jpm) ** n
            rec["K%d_z0" % L] = k_z0
            rec["K%d_0" % L] = k_0
            rec["pert%d" % L] = pert
            tot = abs(k_z0) / pert
            dress = abs(k_0) / pert if safe else float("nan")
            shift = abs(k_z0 / k_0) if safe else float("nan")
            print("  L=%d: K(z0)=%+.4e  K(0)=%+.4e   total=%.4f"
                  "   dressing=%.4f   shift=%.4f"
                  % (L, k_z0, k_0, tot, dress, shift))
            rec["total%d" % L] = tot
            rec["dress%d" % L] = dress
            rec["shift%d" % L] = shift
        recs.append(rec)
        with open(OUT / "mu_split.json", "w") as fh:
            json.dump(recs, fh, indent=1, default=float)

    print("\n" + "=" * 74)
    print("VERDICT  (hexagon; total = mu^2 = dressing x shift)")
    print("=" * 74)
    print("   x      total    dressing   shift    E_min")
    for r in recs:
        print("  %.3f   %7.4f   %7.4f   %7.4f   %+.4f"
              % (abs(r["jpm"]), r.get("total6", np.nan),
                 r.get("dress6", np.nan), r.get("shift6", np.nan),
                 r["E_min"]))
    print("\n  dressing ~ 1 -> suppression is the extensive z0 shift (BW artifact)")
    print("  shift ~ 1    -> suppression is in the dressed intermediate spectrum")
    print("\nPerimeter dependence (does the split differ for L=6 vs L=8?):")
    print("   x     dress6   dress8   shift6   shift8")
    for r in recs:
        print("  %.3f  %7.4f  %7.4f  %7.4f  %7.4f"
              % (abs(r["jpm"]), r.get("dress6", np.nan), r.get("dress8", np.nan),
                 r.get("shift6", np.nan), r.get("shift8", np.nan)))


if __name__ == "__main__":
    main()
