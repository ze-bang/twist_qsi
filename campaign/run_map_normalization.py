"""Is the ring-exchange suppression physical, or Brillouin-Wigner normalization?

The campaign quotes K_6/g_6 falling from ~0.97 (Jpm=-0.005) to 0.070 (-0.35),
i.e. the dressed hexagon amplitude an order of magnitude below the perturbative
g_6 = 12|Jpm|^3/Jzz^2.  Every one of those numbers is a matrix element of the
Feshbach map F(z), which is BRILLOUIN-WIGNER: energy dependent, not a Hermitian
effective Hamiltonian.  Its off-diagonal entries carry the residue Z, and Z
falls 0.99 -> 0.18 across exactly the range where the claim is made.  Dividing
Z out turns "suppressed 14x" into "suppressed ~3x", so the headline is
convention sensitive and cannot be written until this is settled.

This script builds BOTH objects on cubic-16 at the same coupling and compares
them entry by entry:

  BW  : sector_project(F(z_0))            -- what the campaign has been quoting
  HERM: sector_project(polar pullback of the exact band)
        -- the des Cloizeaux / Loewdin effective Hamiltonian, manifestly
           Hermitian, residue free, and the honest object to compare with g_6.

Both are the coefficient of the same hexagon-flip operator between the same two
ice configurations, so both may be divided by g_6 with the same meaning.

Also reports K_8, K_10, K_12.  Caveat, and it is not small: on cubic-16 EVERY
simple loop with L >= 8 winds, so the mask deletes all of them and the masked
K_8,10,12 are empty there.  To test whether the BW->Hermitian conversion is
uniform across perimeter -- which is what decides whether the FCC-32 K_L/K_6
ratios inherit this problem -- the longer classes are also pooled with the
sector labels flattened, which makes the tomography average every entry rather
than only mask survivors.  Those numbers are class amplitudes of the UNMASKED
map and are used only for the uniformity test, never as physical K_L.

Expected outcomes:
  * H/BW ~ 1 uniformly            -> the suppression is physical; write it.
  * H/BW ~ Z, class uniform       -> the absolute claim shrinks to ~3x but the
                                     K_L/K_6 ratios (and the loop hierarchy)
                                     are untouched.
  * H/BW class dependent          -> the loop hierarchy is contaminated too.

The exact band must be isolated for the Hermitian object to exist at all.  On
cubic-16 that fails around Jpm <= -0.19 (bw_greyzone), i.e. possibly before the
couplings where the claim is loudest.  Failures are reported, not hidden.

Usage: run_map_normalization.py <jpm> [<jpm> ...]
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
    cubic16_translation_permutations, extract_exact_band,
    fixed_magnetization_basis, microscopic_character_hamiltonian,
    translation_sector_bases)
from qsi_campaign.downfold import build_blocks                 # noqa: E402
from qsi_campaign.protocol import (                            # noqa: E402
    polarization_sector_labels, sector_project)
from qsi_campaign.tomography import (                          # noqa: E402
    connected_class_amplitudes, graph_distance_matrix, loop_tomography,
    minimum_image_distance_matrix)
from run_feshbach_anatomy import classify_operator, string_classes  # noqa: E402

OUTPUT = ROOT / "campaign" / "outputs" / "map_normalization"
OUTPUT.mkdir(parents=True, exist_ok=True)

# Leading loop coefficients, K_L^(0) = 2 C(L-2, L/2-1) |Jpm|^n / Jzz^(n-1)
PERT = {6: 12.0, 8: 40.0, 10: 140.0, 12: 504.0}


def perturbative(length: int, jpm: float) -> float:
    return PERT[length] * abs(jpm) ** (length // 2)


def class_amplitudes(matrix, ice_states, cluster, labels, gdist, rdist):
    """K_L per perimeter from the proper degree-two simple-loop test."""
    summary = loop_tomography(matrix, ice_states, cluster, labels, gdist, rdist)
    return connected_class_amplitudes(summary)


def main() -> None:
    couplings = [float(v) for v in sys.argv[1:]]
    cluster = geometry.build_cluster("cubic", (1, 1, 1))
    states = fixed_magnetization_basis(cluster.n_sites, cluster.n_sites // 2)
    index = {int(s): i for i, s in enumerate(states)}
    ice_states = np.asarray(cluster.ice_states, dtype=np.uint64)
    ice_indices = np.asarray([index[int(s)] for s in ice_states])
    sectors = translation_sector_bases(
        states, cubic16_translation_permutations(cluster))
    labels = polarization_sector_labels(cluster)
    flat = np.zeros_like(labels)          # "no mask": every pair same sector
    table = string_classes(cluster)
    gdist = graph_distance_matrix(cluster)
    rdist = minimum_image_distance_matrix(cluster)
    n_band = len(ice_states)

    print("cubic-16: %d states, %d ice, %d sectors, n_band=%d"
          % (len(states), len(ice_states), len(np.unique(labels)), n_band),
          flush=True)

    records = []
    for jpm in couplings:
        t0 = time.perf_counter()
        print("\n" + "=" * 72, flush=True)
        print("Jpm/Jzz = %+0.4f     g6 = %.6e" % (jpm, perturbative(6, jpm)),
              flush=True)
        print("=" * 72, flush=True)
        ham = microscopic_character_hamiltonian(cluster, states, jpm,
                                                np.zeros(3))
        z_star = float(eigsh(ham.real.tocsc(), k=1, which="SA", tol=1e-11,
                             ncv=80, return_eigenvectors=False)[0])

        # ---------- Brillouin-Wigner map (what has been quoted) ----------
        blocks = build_blocks(cluster, states, ice_indices, ham)
        f_raw = blocks.f_matrix(z_star, masked=False).real
        bw_masked = np.asarray(sector_project(f_raw, labels)).real
        names, _ = classify_operator(f_raw, ice_states, table, labels)
        hexmask = names == "hexagon"
        k6_bw = -float(bw_masked[hexmask].mean())

        # ---------- Hermitian polar pullback (the honest comparison) -------
        herm_ok, k6_h, band_diag = True, float("nan"), {}
        h_raw = h_masked = None
        try:
            band = extract_exact_band(ham.tocsc(), sectors, ice_indices, n_band)
            h_raw = np.asarray(band.operator).real
            h_masked = np.asarray(sector_project(h_raw, labels)).real
            k6_h = -float(h_masked[hexmask].mean())
            band_diag = {k: float(v) for k, v in band.diagnostics.items()
                         if isinstance(v, (int, float))}
            band_diag["gap_above_band"] = float(band.fixed_sz_gap_above_band)
        except Exception as exc:                  # band not isolated, etc.
            herm_ok = False
            print("  HERMITIAN PULLBACK FAILED: %s" % exc, flush=True)

        g6 = perturbative(6, jpm)
        print("\n  hexagon amplitude, masked map:")
        print("    BW        K_6 = %+.6e    K_6/g_6 = %.4f"
              % (k6_bw, k6_bw / g6))
        if herm_ok:
            print("    Hermitian K_6 = %+.6e    K_6/g_6 = %.4f"
                  % (k6_h, k6_h / g6))
            print("    Hermitian / BW = %.4f   <-- the correction factor"
                  % (k6_h / k6_bw if k6_bw else float("nan")))
        if band_diag:
            print("    band diagnostics: " + ", ".join(
                "%s=%.4g" % (k, v) for k, v in sorted(band_diag.items())))

        # ---------- perimeter classes, masked and unmasked ----------
        rows = {}
        if True:
            rows[("BW", "masked")] = class_amplitudes(
                bw_masked, ice_states, cluster, labels, gdist, rdist)
            rows[("BW", "unmasked")] = class_amplitudes(
                np.asarray(f_raw).real, ice_states, cluster, flat, gdist, rdist)
        if herm_ok:
            rows[("HERM", "masked")] = class_amplitudes(
                h_masked, ice_states, cluster, labels, gdist, rdist)
            rows[("HERM", "unmasked")] = class_amplitudes(
                h_raw, ice_states, cluster, flat, gdist, rdist)

        print("\n  simple-loop class amplitudes K_L  (cubic-16 masks every")
        print("  L>=8 loop, so those rows are unmasked-class values only):")
        print("    L   scope      K_L^(0)      BW           HERM         "
              "HERM/BW   BW/K0    HERM/K0")
        table_rows = []
        for length in (6, 8, 10, 12):
            for scope in ("masked", "unmasked"):
                b = rows.get(("BW", scope), {}).get(length, {}).get("K")
                h = rows.get(("HERM", scope), {}).get(length, {}).get("K")
                if b is None and h is None:
                    continue
                p = perturbative(length, jpm)
                print("    %-3d %-9s %.4e   %s   %s   %s   %s   %s"
                      % (length, scope, p,
                         "%.4e" % b if b else "     --     ",
                         "%.4e" % h if h else "     --     ",
                         "%.4f" % (h / b) if (b and h) else "  --  ",
                         "%.4f" % (b / p) if b else "  --  ",
                         "%.4f" % (h / p) if h else "  --  "))
                table_rows.append({"L": length, "scope": scope,
                                   "pert": p, "bw": b, "herm": h})

        ratios = [r["herm"] / r["bw"] for r in table_rows
                  if r["bw"] and r["herm"] and r["scope"] == "unmasked"]
        if len(ratios) > 1:
            spread = ((max(ratios) - min(ratios))
                      / max(abs(float(np.mean(ratios))), 1e-30))
            print("\n  HERM/BW across perimeters (unmasked): %s"
                  % ", ".join("%.3f" % r for r in ratios))
            print("  relative spread %.3f -> %s" % (
                spread,
                "UNIFORM: K_L/K_6 ratios safe, only the absolute scale moves"
                if spread < 0.10 else
                "NOT UNIFORM: the loop hierarchy inherits the convention too"))

        rec = {"jpm": jpm, "g6": g6, "z_star": z_star,
               "k6_bw": k6_bw, "k6_herm": k6_h if herm_ok else None,
               "k6_bw_over_g6": k6_bw / g6,
               "k6_herm_over_g6": (k6_h / g6) if herm_ok else None,
               "herm_over_bw": (k6_h / k6_bw) if (herm_ok and k6_bw) else None,
               "hermitian_ok": herm_ok, "band_diagnostics": band_diag,
               "classes": table_rows, "runtime": time.perf_counter() - t0}
        records.append(rec)
        print("\n  (%.0f s)" % rec["runtime"], flush=True)

        with open(OUTPUT / "normalization.json", "w") as fh:
            json.dump(records, fh, indent=1, default=float)

    print("\n" + "=" * 72)
    print("SUMMARY: hexagon amplitude against perturbation theory")
    print("=" * 72)
    print("  Jpm       K6/g6 (BW)   K6/g6 (Herm)   Herm/BW   band isolated?")
    for r in records:
        print("  %+0.4f   %10.4f   %12s   %7s   %s"
              % (r["jpm"], r["k6_bw_over_g6"],
                 "%.4f" % r["k6_herm_over_g6"] if r["hermitian_ok"] else "--",
                 "%.4f" % r["herm_over_bw"] if r["hermitian_ok"] else "--",
                 "yes" if r["hermitian_ok"] else "NO"))
    print("\nwrote %s" % (OUTPUT / "normalization.json"))


if __name__ == "__main__":
    main()
