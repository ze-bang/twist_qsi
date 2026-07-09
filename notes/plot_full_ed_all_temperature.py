#!/usr/bin/env python3
"""All-temperature full-ED check for the 16-site twist protocol.

The low-band twist projection is not a full-Hilbert thermal trace.  This
script therefore computes the exact microscopic spectrum and then performs a
minimal spectral replacement: only the ice-band eigenvalues are replaced by the
twist-projected clean band, while every higher-energy state is kept from the
untwisted full ED spectrum.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
QED_PY = ROOT / "QED" / "python"
if QED_PY.exists():
    sys.path.insert(0, str(QED_PY))

import qed  # noqa: E402
import recompute_finite_size_artifact as R  # noqa: E402
from twist_resolved_full_band_symmetry import site_translation_perms  # noqa: E402
from twist_resolved_qed_full_band import build_qed_operator  # noqa: E402

FIGS = HERE / "figs"
FIGS.mkdir(exist_ok=True)


def exact_full_spectrum(cl: R.Cluster, jpm: float, cache: Path) -> np.ndarray:
    if cache.exists():
        data = np.load(cache)
        return np.asarray(data["E_full"], dtype=float)

    op = build_qed_operator(cl, jpm, np.zeros(3), "physical")
    perms = [p.tolist() for p in site_translation_perms(cl)]
    t0 = time.time()
    res = qed.full_spectrum(
        op,
        symmetry=perms,
        point_group="off",
        spin_flip="off",
        time_reversal="off",
        device="cpu",
        verbose=False,
    )
    evals = np.sort(np.asarray(res.eigenvalues, dtype=float))
    np.savez_compressed(
        cache,
        E_full=evals,
        jpm=float(jpm),
        n_sites=int(cl.n_sites),
        n_eigenvalues=int(len(evals)),
        elapsed_s=float(time.time() - t0),
        symmetry="four FCC translations, all Sz sectors",
    )
    return evals


def specific_heat_batched(evals: np.ndarray, temps: np.ndarray, batch: int = 96) -> np.ndarray:
    e = np.asarray(evals, dtype=float)
    e = e - np.min(e)
    out = np.empty_like(temps, dtype=float)
    for start in range(0, len(temps), batch):
        t = temps[start : start + batch]
        beta = 1.0 / t[:, None]
        w = np.exp(-beta * e[None, :])
        z = np.sum(w, axis=1)
        e1 = np.sum(w * e[None, :], axis=1) / z
        e2 = np.sum(w * e[None, :] ** 2, axis=1) / z
        out[start : start + batch] = (e2 - e1**2) / t**2
    return out


def peak_in_window(temps: np.ndarray, curve: np.ndarray, lo: float, hi: float) -> tuple[float, float]:
    mask = (temps >= lo) & (temps <= hi)
    idx = np.flatnonzero(mask)
    if len(idx) == 0:
        raise ValueError("empty peak window")
    k0 = idx[int(np.argmax(curve[mask]))]
    if 0 < k0 < len(temps) - 1:
        x = np.log(temps[k0 - 1 : k0 + 2])
        y = curve[k0 - 1 : k0 + 2]
        den = y[0] - 2.0 * y[1] + y[2]
        if abs(den) > 1e-30:
            dx = 0.5 * (y[0] - y[2]) / den
            if abs(dx) < 1.0:
                xpk = x[1] + dx * (x[1] - x[0])
                ypk = y[1] - 0.25 * (y[0] - y[2]) * dx
                return float(np.exp(xpk)), float(ypk)
    return float(temps[k0]), float(curve[k0])


def shifted_replacement_band(e_full: np.ndarray, e_clean: np.ndarray) -> np.ndarray:
    n = len(e_clean)
    clean = np.sort(np.asarray(e_clean, dtype=float))
    shift = float(np.mean(e_full[:n]) - np.mean(clean))
    out = np.array(e_full, copy=True)
    out[:n] = clean + shift
    return np.sort(out)


def main() -> None:
    jpm = -0.05
    cl = R.build_cluster("cubic", (1, 1, 1))
    spectrum_cache = HERE / "full_ed_16site_spectrum_jm0p05.npz"
    lowband_path = HERE / "twist_resolved_qed_dipole2_M2_jm0p05.npz"
    out_path = HERE / "full_ed_all_temperature_check_jm0p05.npz"

    e_full = exact_full_spectrum(cl, jpm, spectrum_cache)
    low = np.load(lowband_path)
    e_low_phi0 = np.sort(np.asarray(low["E_qed_phi0"], dtype=float))
    e_low_clean = np.sort(np.asarray(low["E_qed_twist_avg"], dtype=float))
    n_band = len(e_low_clean)

    low_match = float(np.max(np.abs(np.sort(e_full[:n_band]) - e_low_phi0)))
    e_replaced = shifted_replacement_band(e_full, e_low_clean)

    temps = np.geomspace(1.0e-4, 5.0, 1600)
    c_full = specific_heat_batched(e_full, temps)
    c_replaced = specific_heat_batched(e_replaced, temps)
    c_low_phi0 = R.specific_heat(e_low_phi0, temps)
    c_low_clean = R.specific_heat(e_low_clean, temps)

    low_full = peak_in_window(temps, c_full, 2.0e-3, 8.0e-2)
    low_replaced = peak_in_window(temps, c_replaced, 2.0e-4, 8.0e-2)
    high_full = peak_in_window(temps, c_full, 8.0e-2, 3.0)
    high_replaced = peak_in_window(temps, c_replaced, 8.0e-2, 3.0)

    summary = {
        "method": "exact full spectrum plus ice-band spectral replacement",
        "jpm": jpm,
        "n_sites": int(cl.n_sites),
        "n_full_eigenvalues": int(len(e_full)),
        "n_ice_band": int(n_band),
        "low_band_phi0_vs_full_max_abs": low_match,
        "Tpk_full_ED_low": low_full[0],
        "Cpk_full_ED_low": low_full[1],
        "Tpk_replaced_low": low_replaced[0],
        "Cpk_replaced_low": low_replaced[1],
        "Tpk_full_ED_high": high_full[0],
        "Cpk_full_ED_high": high_full[1],
        "Tpk_replaced_high": high_replaced[0],
        "Cpk_replaced_high": high_replaced[1],
        "high_peak_relative_shift": (high_replaced[0] - high_full[0]) / high_full[0],
        "high_peak_relative_height_change": (high_replaced[1] - high_full[1]) / high_full[1],
    }

    np.savez_compressed(
        out_path,
        T=temps,
        E_full=e_full,
        E_replaced=e_replaced,
        E_low_phi0=e_low_phi0,
        E_low_clean=e_low_clean,
        C_full=c_full,
        C_replaced=c_replaced,
        C_low_phi0=c_low_phi0,
        C_low_clean=c_low_clean,
        summary=json.dumps(summary),
    )
    out_path.with_suffix(".json").write_text(json.dumps(summary, indent=2))

    fig, (ax, ax_zoom) = plt.subplots(
        2,
        1,
        figsize=(7.1, 6.2),
        sharex=True,
        gridspec_kw={"height_ratios": [1.15, 1.0], "hspace": 0.08},
    )
    ax.plot(temps, c_full / cl.n_sites, color="black", lw=2.0, label="full ED")
    ax.plot(
        temps,
        c_replaced / cl.n_sites,
        color="#1b9e77",
        lw=1.8,
        label="full ED with clean ice-band replacement",
    )
    ax.scatter([high_full[0]], [high_full[1] / cl.n_sites], color="black", s=18, zorder=3)
    ax.scatter([high_replaced[0]], [high_replaced[1] / cl.n_sites], color="#1b9e77", s=18, zorder=3)
    ax.set_ylabel("$C(T)/N$")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)

    mask_low = temps <= 0.12
    ax_zoom.plot(temps, c_full / cl.n_sites, color="black", lw=1.8, label="full ED")
    ax_zoom.plot(temps, c_replaced / cl.n_sites, color="#1b9e77", lw=1.8)
    ax_zoom.plot(
        temps[mask_low],
        c_low_phi0[mask_low] / cl.n_sites,
        color="#d95f02",
        ls="--",
        lw=1.2,
        label="low band, $\\theta=0$",
    )
    ax_zoom.plot(
        temps[mask_low],
        c_low_clean[mask_low] / cl.n_sites,
        color="#1b9e77",
        ls=":",
        lw=1.5,
        label="low band, $2\\delta$ $M=2$",
    )
    ax_zoom.axvline(4.0 * jpm * jpm, color="#d95f02", ls="-.", lw=0.8)
    ax_zoom.axvline(12.0 * abs(jpm) ** 3, color="#7570b3", ls="--", lw=0.8)
    ax_zoom.set_xscale("log")
    ax_zoom.set_xlim(1.0e-4, 5.0)
    ax_zoom.set_ylim(bottom=0.0)
    ax_zoom.set_xlabel("$T/J_{zz}$")
    ax_zoom.set_ylabel("$C(T)/N$")
    ax_zoom.legend(frameon=False, fontsize=8, loc="upper left")
    ax_zoom.spines[["top", "right"]].set_visible(False)

    for ext in ("pdf", "png"):
        fig.savefig(FIGS / f"fig_full_ed_all_temperature_specific_heat.{ext}", bbox_inches="tight", dpi=240)
    plt.close(fig)

    print(json.dumps(summary, indent=2))
    print(f"wrote {out_path}")
    print(f"wrote {FIGS / 'fig_full_ed_all_temperature_specific_heat.pdf'}")


if __name__ == "__main__":
    main()
