#!/usr/bin/env python3
"""
Twist-average FTLM static quadrupole thermal-expansion coefficients
``α_Q=(⟨QH⟩-⟨Q⟩⟨H⟩)/T²`` (stored as ``expectation`` under
``correlations/*_alpha`` in ``ham/quad_qh_static/ed_results.h5``).

Writes ``quad_qh_twist_avg_analysis.npz`` with arrays shaped (n_J3, n_T), aligned
to the same log-spaced ``T`` convention as ``analyze_jpm03_twist_j3.py``.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np

CHANNELS = (
    "Eg_Q1_alpha",
    "Eg_Q2_alpha",
    "T2g_Q_xy_alpha",
    "T2g_Q_xz_alpha",
    "T2g_Q_yz_alpha",
)

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _abs_h5_path(p: str | Path) -> Path:
    p = Path(p)
    return p.resolve() if p.is_absolute() else (_REPO_ROOT / p).resolve()


def _case_anchor_h5(rec: dict) -> Path:
    """HDF5 under ``ftlm/`` or ``full_thermo/`` — parent.parent is the twist case dir."""
    p = rec.get("thermo_h5") or rec.get("ftlm_h5")
    if not p:
        raise KeyError("run record has neither thermo_h5 nor ftlm_h5")
    return _abs_h5_path(p)


def read_quad_h5(h5path: Path) -> dict:
    out: dict[str, np.ndarray] = {}
    with h5py.File(h5path, "r") as f:
        g = f["correlations"]
        T = None
        for name in CHANNELS:
            prefix = f"correlations/{name}"
            Tn = f[prefix + "/temperatures"][...]
            if T is None:
                T = Tn
            else:
                assert np.allclose(T, Tn), f"T grid mismatch in {h5path} for {name}"
            out[name] = f[prefix + "/expectation"][...]
            out[name + "_err"] = f[prefix + "/expectation_error"][...]
    out["T"] = T
    return out


def _interp(T_src: np.ndarray, y: np.ndarray, T_dst: np.ndarray) -> np.ndarray:
    return np.interp(T_dst, T_src, y)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True, help="scan root (summary.json)")
    ap.add_argument(
        "--quad-subdir",
        type=str,
        default="quad_qh_static",
        help="hamiltonian subdirectory with static Q-H HDF5 (default: quad_qh_static)",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output npz (default <root>/quad_qh_twist_avg_analysis.npz)",
    )
    args = ap.parse_args()

    root = args.root.resolve()
    summary_path = root / "summary.json"
    if not summary_path.is_file():
        raise SystemExit(f"missing {summary_path}")
    summary = json.loads(summary_path.read_text())

    by_j3: dict[float, list] = defaultdict(list)
    for rec in summary["runs"]:
        by_j3[float(rec["J3"])].append(rec)

    j3_sorted = sorted(by_j3)
    quad_sub = args.quad_subdir

    all_curves = []
    for rec in summary["runs"]:
        h5 = _case_anchor_h5(rec).parent.parent / "ham" / quad_sub / "ed_results.h5"
        if not h5.is_file():
            raise SystemExit(f"missing {h5} — run run_quad_static_ftlm_scan.py")
        all_curves.append(read_quad_h5(h5))

    t_lo = max(float(d["T"][0]) for d in all_curves)
    t_hi = min(float(d["T"][-1]) for d in all_curves)
    n_T = max(len(d["T"]) for d in all_curves)
    T_ref = np.exp(np.linspace(np.log(t_lo), np.log(t_hi), n_T))

    pack = dict(T=T_ref, j3=np.array(j3_sorted), Jpm=float(summary["Jpm"]), Jzz=float(summary["Jzz"]))

    for ch in CHANNELS:
        rows = []
        err_rows = []
        for j3 in j3_sorted:
            stacks_Y = []
            stacks_e = []
            for r in by_j3[j3]:
                h5 = _case_anchor_h5(r).parent.parent / "ham" / quad_sub / "ed_results.h5"
                d = read_quad_h5(h5)
                stacks_Y.append(_interp(d["T"], d[ch], T_ref))
                stacks_e.append(_interp(d["T"], d[ch + "_err"], T_ref))
            n_tw = len(stacks_Y)
            Y_avg = np.mean(np.stack(stacks_Y, axis=0), axis=0)
            e_avg = np.sqrt(np.mean(np.stack([e**2 for e in stacks_e], axis=0), axis=0)) / np.sqrt(n_tw)
            rows.append(Y_avg)
            err_rows.append(e_avg)
        pack[ch.replace("_alpha", "") + "_twist_avg"] = np.stack(rows, axis=0)
        pack[ch.replace("_alpha", "") + "_twist_avg_err"] = np.stack(err_rows, axis=0)

    if args.out is not None:
        out_npz = args.out
    elif quad_sub == "quad_qh_static":
        out_npz = root / "quad_qh_twist_avg_analysis.npz"
    else:
        out_npz = root / f"quad_qh_twist_avg_{quad_sub}.npz"
    np.savez_compressed(out_npz, **pack)
    print(f"wrote {out_npz}")


if __name__ == "__main__":
    main()
