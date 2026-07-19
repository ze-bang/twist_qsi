#!/usr/bin/env python3
"""Test candidate projection kernels on the 16-site perturbative row table."""
from __future__ import annotations

import json
from itertools import product
from pathlib import Path

import numpy as np

import recompute_finite_size_artifact as R

HERE = Path(__file__).resolve().parent


def center(h):
    return h - np.eye(h.shape[0]) * np.trace(h) / h.shape[0]


def rel(a, b):
    return float(np.linalg.norm(center(a) - center(b)) / np.linalg.norm(center(b)))


def assemble_filter(cl, pt, jpm, filt):
    h = np.zeros((cl.n_ice, cl.n_ice), dtype=complex)
    for key, power in (("H2", 2), ("H3", 3)):
        rows = pt[key]
        w = np.asarray(filt(rows), dtype=complex)
        vals = (jpm ** power) * rows["c"] * w
        np.add.at(h, (rows["t"], rows["s"]), vals)
    return 0.5 * (h + h.conj().T)


def peak(h, temps):
    e = np.linalg.eigvalsh(h).real
    return R.refined_peak(temps, R.specific_heat(e, temps))


def exact_delta0(cl):
    return lambda rows: ((R.transport_delta(cl, rows) == 0).all(axis=1)).astype(float)


def exact_n0(rows):
    return ((rows["N"] == 0).all(axis=1)).astype(float)


def n_mod(m):
    return lambda rows: ((rows["N"] % m == 0).all(axis=1)).astype(float)


def delta4_mod(cl, m):
    return lambda rows: ((R.transport_delta(cl, rows) % m == 0).all(axis=1)).astype(float)


def physical_delta_grid(cl, m):
    phis = np.array(list(product(range(m), repeat=3)), dtype=float) * (2.0 * np.pi / m)

    def filt(rows):
        delta = R.transport_delta(cl, rows).astype(float) / 4.0
        return np.exp(-1j * delta @ phis.T).mean(axis=1)

    return filt


def delta4_character_grid(cl, m):
    phis = np.array(list(product(range(m), repeat=3)), dtype=float) * (2.0 * np.pi / m)

    def filt(rows):
        delta4 = R.transport_delta(cl, rows).astype(float)
        return np.exp(-1j * delta4 @ phis.T).mean(axis=1)

    return filt


def main():
    cl = R.build_cluster("cubic", (1, 1, 1))
    pt = R.sw_order23(cl, verbose=False)
    jpm = -0.05
    temps = np.geomspace(1e-4, 0.12, 900)
    h_all = R.assemble(cl, pt, jpm, "all")
    h_delta0 = R.assemble(cl, pt, jpm, "delta0")

    tests = {
        "all": lambda rows: np.ones(len(rows["c"])),
        "N0": exact_n0,
        "Nmod2": n_mod(2),
        "delta4_mod2": delta4_mod(cl, 2),
        "delta4_mod3": delta4_mod(cl, 3),
        "physical_delta_M2": physical_delta_grid(cl, 2),
        "physical_delta_M3": physical_delta_grid(cl, 3),
        "physical_delta_M6": physical_delta_grid(cl, 6),
        "physical_delta_M16": physical_delta_grid(cl, 16),
        "delta4_character_M2": delta4_character_grid(cl, 2),
        "delta4_character_M3": delta4_character_grid(cl, 3),
        "delta4_character_M4": delta4_character_grid(cl, 4),
    }

    out = {
        "cluster": {"basis": "cubic", "shape": [1, 1, 1], "n_sites": cl.n_sites, "n_ice": cl.n_ice},
        "jpm": jpm,
        "reference_peaks": {
            "all": peak(h_all, temps),
            "delta0": peak(h_delta0, temps),
        },
        "tests": {},
    }
    for name, filt in tests.items():
        h = assemble_filter(cl, pt, jpm, filt)
        out["tests"][name] = {
            "Tpk": peak(h, temps),
            "rel_to_all": rel(h, h_all),
            "rel_to_delta0": rel(h, h_delta0),
            "max_abs_difference_to_delta0": float(np.max(np.abs(h - h_delta0))),
        }

    path = HERE / "projection_kernel_tests_16site_jm0p05.json"
    path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
