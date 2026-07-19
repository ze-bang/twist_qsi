#!/usr/bin/env python3
"""Compare the 16-site QED holonomy protocol against perturbative targets."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent


def center(h):
    h = np.asarray(h)
    return h - np.eye(h.shape[0]) * np.trace(h) / h.shape[0]


def metrics(a, b):
    ac = center(a)
    bc = center(b)
    rel_to_b = np.linalg.norm(ac - bc) / np.linalg.norm(bc)
    scale = (np.vdot(bc, ac) / np.vdot(bc, bc)).real
    rel_fit = np.linalg.norm(ac - scale * bc) / np.linalg.norm(ac)
    ea = np.linalg.eigvalsh(ac)
    eb = np.linalg.eigvalsh(bc)
    spec_raw = np.linalg.norm(ea - eb) / np.linalg.norm(eb)
    spec_fit = np.linalg.norm(ea - scale * eb) / np.linalg.norm(ea)
    return {
        "rel_fro_to_target": float(rel_to_b),
        "best_scale": float(scale),
        "rel_fro_after_best_scale": float(rel_fit),
        "rel_spectrum_raw": float(spec_raw),
        "rel_spectrum_after_best_scale": float(spec_fit),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "npz",
        type=Path,
        nargs="?",
        default=HERE / "twist_resolved_qed_full_band_jm0p05.npz",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=HERE / "qed_protocol_16site_comparison_jm0p05.json",
    )
    args = ap.parse_args()

    z = np.load(args.npz, allow_pickle=True)
    summary = json.loads(str(z["summary"]))
    selected = [
        "jpm",
        "n_sites",
        "sz_dim",
        "ice_dim",
        "n_band",
        "n_grid",
        "n_twists",
        "twist_kind",
        "solver",
        "Tpk_qed_phi0",
        "Tpk_qed_twist_operator_avg",
        "Tpk_pt_all",
        "Tpk_pt_delta0",
        "g4",
        "ghex",
    ]
    pairs = [
        ("qed_phi0", "pt_all", z["H_qed_phi0"], z["H_pt_all"]),
        ("qed_phi0", "pt_delta0", z["H_qed_phi0"], z["H_pt_delta0"]),
        ("qed_projected_avg", "pt_all", z["H_qed_twist_avg"], z["H_pt_all"]),
        ("qed_projected_avg", "pt_delta0", z["H_qed_twist_avg"], z["H_pt_delta0"]),
    ]
    out = {
        "source_npz": str(args.npz),
        "summary_selected": {k: summary[k] for k in selected if k in summary},
        "min_ice_overlap": min(d["ice_overlap_min"] for d in summary["diagnostics"]),
        "max_full_vector_gram_deviation": max(
            max(
                abs(d["full_vector_gram_min"] - 1.0),
                abs(d["full_vector_gram_max"] - 1.0),
            )
            for d in summary["diagnostics"]
        ),
        "operator_metrics": {
            f"{lhs}_vs_{rhs}": metrics(a, b)
            for lhs, rhs, a, b in pairs
        },
    }
    args.out.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
