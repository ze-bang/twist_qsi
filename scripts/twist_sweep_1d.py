"""
Fine twist sweep: phi_x in [0, 2*pi) at phi_y = phi_z = 0.
Computes the lowest-eigenvalue dispersion vs phi_x to visualise the
"photon-like" finite-size mode lifted by the twist.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ED_BIN = ROOT.parent / "QED" / "build" / "ED"
sys.path.insert(0, str(ROOT / "scripts"))
from twist_helper import write_pyrochlore_xxz_with_twist  # noqa: E402

JXX = 0.2
JYY = 0.2
JZZ = 1.0
NUM_SITES = 16
N_PHI = 13
N_EIGS = 24


def run_at(phi_x, run_root: Path, force=False):
    tag = f"phix_{phi_x/np.pi:.4f}pi"
    case = run_root / tag
    ham = case / "ham"
    spec = case / "spectrum"
    spec.mkdir(parents=True, exist_ok=True)

    write_pyrochlore_xxz_with_twist(str(ham), JXX, JYY, JZZ,
                                    (float(phi_x), 0.0, 0.0))
    h5 = spec / "ed_results.h5"
    if force or not h5.exists():
        cmd = [str(ED_BIN), str(ham),
               "--method=LANCZOS",
               f"--num_sites={NUM_SITES}",
               f"--output={spec}",
               f"--eigenvalues={N_EIGS}",
               "--iterations=2000",
               "--tolerance=1e-9"]
        log = case / "run.log"
        with open(log, "w") as f:
            f.write(f"$ {' '.join(cmd)}\n")
            f.flush()
            rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT).returncode
        if rc != 0:
            raise RuntimeError(f"LANCZOS failed for {tag} (rc={rc})  see {log}")
    with h5py.File(h5, "r") as f:
        eigs = np.sort(f["/eigendata/eigenvalues"][...])
    return eigs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "output" / "sweep1d"))
    ap.add_argument("--figs", default=str(ROOT / "paper" / "figs"))
    args = ap.parse_args()

    out = Path(args.out)
    figs = Path(args.figs)
    out.mkdir(parents=True, exist_ok=True)
    figs.mkdir(parents=True, exist_ok=True)

    phis = np.linspace(0.0, 2.0 * np.pi, N_PHI)
    spectrum = []
    t0 = time.time()
    for k, phi in enumerate(phis):
        print(f"[{k+1}/{len(phis)}] phi_x = {phi/np.pi:.3f} pi", flush=True)
        eigs = run_at(phi, out)
        spectrum.append(eigs[:N_EIGS])
    spectrum = np.asarray(spectrum)
    print(f"Total: {time.time()-t0:.1f} s")
    np.savez(out / "spectrum_vs_phix.npz", phis=phis, spectrum=spectrum)

    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    e0 = spectrum[:, 0]
    for n in range(N_EIGS):
        col = plt.cm.plasma(0.05 + 0.85 * n / N_EIGS)
        ax.plot(phis / np.pi, spectrum[:, n] - e0[0], "-", color=col,
                lw=1.0 if n > 0 else 2.0, alpha=0.85,
                label=(r"$E_0$" if n == 0 else (r"$E_{n>0}$" if n == 1 else None)))
    e_mean_per_phi = spectrum.mean(axis=0)
    ax.set_xlabel(r"$\varphi_x \, / \, \pi$")
    ax.set_ylabel(r"$E_n(\varphi_x) - E_0(0)$")
    ax.set_title(r"Twist-induced photon dispersion: $E_n(\varphi_x)$ at $\varphi_y=\varphi_z=0$"
                 + "\n(16-site pyrochlore, $J_\\pm=-0.1\\,J_{zz}$)")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(figs / "fig_photon_dispersion.pdf")
    fig.savefig(figs / "fig_photon_dispersion.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    n_band_show = 16
    for n in range(n_band_show):
        col = plt.cm.viridis(0.1 + 0.8 * n / n_band_show)
        ax.plot(phis / np.pi, spectrum[:, n], "-", color=col,
                lw=1.5, alpha=0.85)
    e0_avg = spectrum[:, 0].mean()
    ax.axhline(e0_avg, color="black", ls="--", lw=0.8,
               label=fr"$\overline{{E_0}}(\varphi_x) = {e0_avg:.4f}$")
    ax.axhline(spectrum[0, 0], color="tab:red", ls=":", lw=0.8,
               label=fr"bare $E_0(0) = {spectrum[0,0]:.4f}$")
    ax.set_xlabel(r"$\varphi_x \, / \, \pi$")
    ax.set_ylabel(r"$E_n(\varphi_x)$")
    ax.set_title(r"Lowest 16 eigenvalues vs continuous twist along x")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(figs / "fig_lowband_vs_phix.pdf")
    fig.savefig(figs / "fig_lowband_vs_phix.png", dpi=160)
    plt.close(fig)

    print("Wrote", figs / "fig_photon_dispersion.png")
    print("Wrote", figs / "fig_lowband_vs_phix.png")


if __name__ == "__main__":
    main()
