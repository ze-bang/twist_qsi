"""Where does mu come from?  Spectral anatomy of the loop amplitude.

The collapse says K_L = K_L^(0) mu^(L/2-1) with one mu per coupling, and that
mu is intensive.  This asks what mu IS, using the exact spectral form of the
fold.  With E_k the eigenvalues of QHQ and c_{k,a} = <k|QHP|a>, the amplitude
between two ice states a, b is

    K_ab(z) = sum_k w_k / (z - E_k),        w_k = c_{k,a} c_{k,b}.

Two mechanisms can suppress it, and they mean completely different things:

  (i)  DENOMINATOR INFLATION.  The weight sits further above z_0, so every
       1/(z_0 - E_k) is smaller.  Then mu is a renormalized virtual gap and the
       gauge theory is the perturbative one with a heavier intermediate state.
  (ii) DESTRUCTIVE INTERFERENCE.  w_k changes sign across the spinon band and
       contributions cancel.  Then sum|w_k| >> |sum w_k|, the "gap" is a
       fiction, and mu is a vertex/orthogonality effect.

Diagnostics computed per coupling, for a representative hexagon (L=6) pair and
an L=8 pair:

    W_tot     = sum_k w_k                    (coherent weight)
    W_abs     = sum_k |w_k|                  (total weight)
    cancel    = W_abs / |W_tot|              (1 = no cancellation)
    D_eff     = W_tot / K_ab(z_0)            (effective denominator)
    E_bar     = sum_k w_k E_k / W_tot        (signed centroid)
    E_bar_abs = sum_k |w_k| E_k / W_abs      (unsigned centroid)
    K_incoh   = sum_k |w_k| / (z_0 - E_k)    (what it would be with no signs)

Verdict: if D_eff(x)/D_eff(x->0) tracks 1/mu(x), mechanism (i).  If instead
`cancel` grows like 1/mu while D_eff is flat, mechanism (ii).

cubic-16, exact fold, no truncation.  Usage: analyse_mu_anatomy.py <jpm> ...
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
OUT.mkdir(parents=True, exist_ok=True)


def pair_of_class(ice_states, table, want, labels):
    """One representative same-sector ice pair whose flipped set is `want`."""
    states = [int(s) for s in ice_states]
    for i in range(len(states)):
        for j in range(i + 1, len(states)):
            if labels[i] != labels[j]:
                continue
            diff = states[i] ^ states[j]
            name = table.get(diff)
            if want == "hexagon" and name == "hexagon":
                return i, j
            if want == "L8" and name is None and bin(diff).count("1") == 8:
                return i, j
    return None


def anatomy(blocks, z0, i, j):
    w = np.asarray(blocks.coupling)[:, i] * np.asarray(blocks.coupling)[:, j]
    E = np.asarray(blocks.poles)
    denom = z0 - E
    amp = float(np.sum(w / denom))
    w_tot, w_abs = float(np.sum(w)), float(np.sum(np.abs(w)))
    return {
        "amp": amp,
        "W_tot": w_tot,
        "W_abs": w_abs,
        "cancel": w_abs / abs(w_tot) if w_tot else np.inf,
        "D_eff": w_tot / amp if amp else np.nan,
        "E_bar": float(np.sum(w * E) / w_tot) if w_tot else np.nan,
        "E_bar_abs": float(np.sum(np.abs(w) * E) / w_abs) if w_abs else np.nan,
        "K_incoh": float(np.sum(np.abs(w) / denom)),
        "E_min": float(E.min()), "E_max": float(E.max()),
        "z0": z0,
    }


def main():
    couplings = [float(v) for v in sys.argv[1:]]
    cluster = geometry.build_cluster("cubic", (1, 1, 1))
    states = fixed_magnetization_basis(cluster.n_sites, cluster.n_sites // 2)
    index = {int(s): k for k, s in enumerate(states)}
    ice_states = np.asarray(cluster.ice_states, dtype=np.uint64)
    ice_indices = np.asarray([index[int(s)] for s in ice_states])
    labels = polarization_sector_labels(cluster)
    table = string_classes(cluster)

    hex_pair = pair_of_class(ice_states, table, "hexagon", labels)
    l8_pair = pair_of_class(ice_states, table, "L8", labels)
    print("representative pairs: hexagon %s, L8 %s" % (hex_pair, l8_pair))

    records = []
    for jpm in couplings:
        t0 = time.perf_counter()
        ham = microscopic_character_hamiltonian(cluster, states, jpm,
                                                np.zeros(3))
        z0 = float(eigsh(ham.real.tocsc(), k=1, which="SA", tol=1e-11, ncv=80,
                         return_eigenvectors=False)[0])
        blocks = build_blocks(cluster, states, ice_indices, ham)
        rec = {"jpm": jpm, "z0": z0,
               "hexagon": anatomy(blocks, z0, *hex_pair)}
        if l8_pair:
            rec["L8"] = anatomy(blocks, z0, *l8_pair)
        records.append(rec)
        h = rec["hexagon"]
        print("\njpm=%+0.3f  z0=%+.5f  (%.0f s)"
              % (jpm, z0, time.perf_counter() - t0), flush=True)
        print("  hexagon: amp=%+.4e  W_tot=%+.4e  W_abs=%.4e" %
              (h["amp"], h["W_tot"], h["W_abs"]))
        print("           cancel=%.4f  D_eff=%+.4f  E_bar=%+.4f "
              "E_bar_abs=%+.4f" %
              (h["cancel"], h["D_eff"], h["E_bar"], h["E_bar_abs"]))
        print("           |amp|/K_incoh=%.4f   QHQ span [%.3f, %.3f]" %
              (abs(h["amp"]) / abs(h["K_incoh"]) if h["K_incoh"] else np.nan,
               h["E_min"], h["E_max"]))
        with open(OUT / "mu_anatomy.json", "w") as fh:
            json.dump(records, fh, indent=1, default=float)

    print("\n" + "=" * 78)
    print("VERDICT TABLE  (normalised to the weakest coupling)")
    print("=" * 78)
    ref = records[0]["hexagon"]
    print("  jpm      D_eff   D_eff/D_ref   cancel  cancel/ref   "
          "(E_bar-z0)  coherence")
    for r in records:
        h = r["hexagon"]
        print("  %+0.3f  %7.4f     %7.4f    %7.4f   %7.4f     %7.4f    %7.4f"
              % (r["jpm"], h["D_eff"], h["D_eff"] / ref["D_eff"],
                 h["cancel"], h["cancel"] / ref["cancel"],
                 h["E_bar"] - h["z0"],
                 abs(h["amp"]) / abs(h["K_incoh"]) if h["K_incoh"] else np.nan))
    print("\n  Compare D_eff/D_ref with 1/mu from the collapse:")
    print("    1/mu = 1.17 (x=0.03) 1.37 (0.06) 1.69 (0.10) 2.15 (0.15)")
    print("           2.64 (0.20)  3.14 (0.25)  3.65 (0.30)")
    print("  If D_eff/D_ref tracks those, mu is a renormalised virtual gap.")
    print("  If instead `cancel` tracks them, mu is destructive interference.")


if __name__ == "__main__":
    main()
