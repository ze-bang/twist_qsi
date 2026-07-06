#!/usr/bin/env python3
"""
Load ``summary.json`` produced by ``run_jpm03_twist_j3_ftlm.py``,
twist-average thermodynamic curves per J3, and write ``twist_avg.npz``.

Reads ``thermo_h5`` when present (full-ED ``/thermodynamics`` from the driver
default), otherwise ``ftlm_h5`` for legacy FTLM outputs.

Optionally scans ``mtpq_spin/`` for ``flct_*.dat`` / ``flct_*_Sx.dat`` style
outputs and records raw paths (full parsing left to the user / follow-up).
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _abs_h5_path(p: str | Path) -> Path:
    p = Path(p)
    return p.resolve() if p.is_absolute() else (_REPO_ROOT / p).resolve()


def read_ftlm(h5path: Path):
    """Read ``ftlm/averaged/*`` from a single HDF5 (standard / non-symmetry path)."""
    with h5py.File(h5path, "r") as f:
        T = f["ftlm/averaged/temperatures"][...]
        E = f["ftlm/averaged/energy"][...]
        C = f["ftlm/averaged/specific_heat"][...]
        S = f["ftlm/averaged/entropy"][...]
        Ce = f["ftlm/averaged/specific_heat_error"][...]
        Ee = f["ftlm/averaged/energy_error"][...]
    return dict(T=T, E=E, C=C, S=S, Ce=Ce, Ee=Ee)


def read_ftlm_symm_sectors(ftlm_dir: Path) -> dict:
    """``--symm`` FTLM writes per-sector HDF5 under ``ftlm/sector_*``. Merge curves
    with Hilbert-space weights ``dim(sector) / sum dim``.

    This is a practical proxy for twist-averaged monitoring; it is not a full
    multi-sector partition-function recombination.
    """
    dim_path = ftlm_dir / "sym_metadata" / "sector_dimensions.txt"
    if not dim_path.is_file():
        raise FileNotFoundError(f"missing sector dimensions: {dim_path}")
    dims = np.array([int(x) for x in dim_path.read_text().split() if x.strip()], dtype=float)
    w = dims / np.sum(dims)

    sector_dirs = sorted(
        (p for p in ftlm_dir.glob("sector_*") if p.is_dir() and (p / "ed_results.h5").is_file()),
        key=lambda p: int(p.name.split("_")[1]),
    )
    if len(sector_dirs) != len(dims):
        raise RuntimeError(
            f"sector count mismatch: {len(sector_dirs)} dirs vs {len(dims)} dims"
        )

    T_ref = None
    E_acc = C_acc = S_acc = None
    Ce_acc = Ee_acc = None

    for sd, wt in zip(sector_dirs, w):
        h5 = sd / "ed_results.h5"
        with h5py.File(h5, "r") as f:
            T = f["ftlm/averaged/temperatures"][...]
            E = f["ftlm/averaged/energy"][...]
            C = f["ftlm/averaged/specific_heat"][...]
            S = f["ftlm/averaged/entropy"][...]
            Ce = f["ftlm/averaged/specific_heat_error"][...]
            Ee = f["ftlm/averaged/energy_error"][...]
        if T_ref is None:
            T_ref = T
            E_acc = wt * E
            C_acc = wt * C
            S_acc = wt * S
            Ce_acc = (wt * Ce) ** 2
            Ee_acc = (wt * Ee) ** 2
        else:
            assert np.allclose(T, T_ref), "sector temperature grid mismatch"
            E_acc += wt * E
            C_acc += wt * C
            S_acc += wt * S
            Ce_acc += (wt * Ce) ** 2
            Ee_acc += (wt * Ee) ** 2

    assert T_ref is not None
    return dict(
        T=T_ref,
        E=E_acc,
        C=C_acc,
        S=S_acc,
        Ce=np.sqrt(Ce_acc),
        Ee=np.sqrt(Ee_acc),
    )


def read_ftlm_auto(h5path: Path) -> dict:
    """Dispatch between standard layout and streaming-symmetry sector layout."""
    with h5py.File(h5path, "r") as f:
        try:
            f["ftlm/averaged/temperatures"]
        except KeyError:
            return read_ftlm_symm_sectors(h5path.parent)
    return read_ftlm(h5path)


def read_full_ed_thermo(h5path: Path) -> dict:
    """Exact partition-function thermodynamics from ``/thermodynamics`` (no error bars)."""
    with h5py.File(h5path, "r") as f:
        T = f["thermodynamics/temperatures"][...]
        E = f["thermodynamics/energy"][...]
        C = f["thermodynamics/specific_heat"][...]
        S = f["thermodynamics/entropy"][...]
    z = np.zeros_like(T)
    return dict(T=T, E=E, C=C, S=S, Ce=z, Ee=z)


def read_thermo_auto(h5path: Path) -> dict:
    """Prefer exact ``/thermodynamics`` if present, else FTLM layouts."""
    with h5py.File(h5path, "r") as f:
        if "thermodynamics/temperatures" in f:
            return read_full_ed_thermo(h5path)
    return read_ftlm_auto(h5path)


def _thermo_h5_for_rec(rec: dict) -> Path:
    p = rec.get("thermo_h5") or rec.get("ftlm_h5")
    if not p:
        raise KeyError("run record has neither thermo_h5 nor ftlm_h5")
    return _abs_h5_path(p)


def _interp_positive_T(T_src: np.ndarray, y: np.ndarray, T_dst: np.ndarray) -> np.ndarray:
    """Linearly interpolate ``y(T)`` onto ``T_dst`` (monotone increasing ``T_src``)."""
    return np.interp(T_dst, T_src, y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True,
                    help="scan root (directory containing summary.json)")
    ap.add_argument("--out", type=str, default="",
                    help="output npz path (default: <root>/twist_avg_analysis.npz)")
    args = ap.parse_args()

    root = Path(args.root)
    summary_path = root / "summary.json"
    if not summary_path.exists():
        raise SystemExit(f"missing {summary_path}")
    summary = json.loads(summary_path.read_text())
    jpm = summary["Jpm"]
    jzz = summary["Jzz"]
    mode = summary.get("thermo_mode", "")
    if mode == "full":
        print("note: thermo_mode=full — exact spectrum curves from /thermodynamics.")
    elif summary.get("use_symm"):
        print(
            "note: use_symm=true — if using FTLM, sector curves were merged with "
            "Hilbert-space weights (see read_ftlm_symm_sectors docstring)."
        )

    by_j3: dict[float, list] = defaultdict(list)
    for rec in summary["runs"]:
        by_j3[float(rec["J3"])].append(rec)

    out_npz = Path(args.out) if args.out else (root / "twist_avg_analysis.npz")

    j3_sorted = sorted(by_j3)

    # Common temperature grid: different sweeps may use different ``--temp-bins``;
    # interpolate every run onto the same grid before twist-averaging.
    # IMPORTANT: QED's FTLM uses ``T_i = exp(log T_min + i * Δ)`` (log-spaced),
    # so we must rebuild the common grid in log space, not linear, otherwise
    # all low-T resolution is thrown away.
    all_curves: list[dict] = []
    for rec in summary["runs"]:
        all_curves.append(read_thermo_auto(_thermo_h5_for_rec(rec)))
    t_lo = max(float(d["T"][0]) for d in all_curves)
    t_hi = min(float(d["T"][-1]) for d in all_curves)
    n_T = max(len(d["T"]) for d in all_curves)
    T_ref = np.exp(np.linspace(np.log(t_lo), np.log(t_hi), n_T))

    C_rows = []
    E_rows = []
    S_rows = []
    Cerr_rows = []

    for j3 in j3_sorted:
        rows = by_j3[j3]
        if len(rows) != 8:
            print(f"warning: J3={j3} has {len(rows)} twists (expected 8)")
        stacks = []
        for r in rows:
            d = read_thermo_auto(_thermo_h5_for_rec(r))
            stacks.append(
                dict(
                    T=T_ref,
                    C=_interp_positive_T(d["T"], d["C"], T_ref),
                    E=_interp_positive_T(d["T"], d["E"], T_ref),
                    S=_interp_positive_T(d["T"], d["S"], T_ref),
                    Ce=_interp_positive_T(d["T"], d["Ce"], T_ref),
                    Ee=_interp_positive_T(d["T"], d["Ee"], T_ref),
                )
            )
        C_avg = np.mean(np.stack([s["C"] for s in stacks]), axis=0)
        E_avg = np.mean(np.stack([s["E"] for s in stacks]), axis=0)
        S_avg = np.mean(np.stack([s["S"] for s in stacks]), axis=0)
        C_err = np.sqrt(np.mean(np.stack([s["Ce"] ** 2 for s in stacks]), axis=0)) / np.sqrt(
            len(stacks)
        )
        C_rows.append(C_avg)
        E_rows.append(E_avg)
        S_rows.append(S_avg)
        Cerr_rows.append(C_err)
        i_peak = int(np.argmax(C_avg))
        print(
            f"J3={j3:6.3f}  C_peak={C_avg[i_peak]:.4f} at T={T_ref[i_peak]:.4g}  "
            f"(g_hex scale ~ {12.0 * abs(jpm) ** 3 / jzz ** 2:.4g} Jzz)"
        )

    np.savez_compressed(
        out_npz,
        T=T_ref,
        j3=np.array(j3_sorted),
        C_twist_avg=np.stack(C_rows, axis=0),
        E_twist_avg=np.stack(E_rows, axis=0),
        S_twist_avg=np.stack(S_rows, axis=0),
        C_twist_avg_err=np.stack(Cerr_rows, axis=0),
        Jpm=jpm,
        Jzz=jzz,
    )
    print(f"wrote {out_npz}")


if __name__ == "__main__":
    main()
