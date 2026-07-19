# PRL submission package

This directory contains the FCC-32 PRL manuscript, Supplemental Material,
figures, and reproducibility data for the winding-free quantum-spin-ice exact
diagonalization method.

## Build

```bash
latexmk -pdf main.tex
latexmk -pdf supplement.tex
latexmk -pdf cover_letter.tex
```

All LaTeX sources use `references.bib` where applicable.

## Letter figures

- Figure 1: wrapping four-loop and contractible hexagon hosted on cubic-16,
  followed by periodic versus winding-free FCC-32 specific heat.
- Figure 2: exact FCC-32 process selectivity and cubic scaling of the
  contractible-ring spectrum and heat-capacity peak.
- Figure 3: periodic and winding-free four-sublattice-traced `Szz` at exactly
  the eight allowed FCC-32 translation momenta, with a linear frequency axis
  and no neutron-polarization factors.
- Figure 4: digitized Ce2Hf2O7 heat capacity, the published NLC-A curve, and
  the FCC-32 low-temperature scale constraint.

## Reproducibility

- `scripts/fcc32_prl_figures.py` regenerates all four Letter figures and the
  FCC-32 numerical records.
- `scripts/recompute_finite_size_artifact.py` provides the cluster geometry,
  path enumeration, projection, thermodynamics, and structure-factor helpers.
- `scripts/digitize_smith2025_heat_capacity.py` documents the calibration of
  the visual-comparison data extracted from Smith et al. Fig. 2(a).
- `data/fcc32_prl_results.npz` stores spectra, thermodynamic peaks, allowed
  momenta, and DSSF arrays.
- `data/fcc32_prl_results.json` stores dimensions, matrix counts, mask
  residuals, and material-parameter constraints.
- `data/fcc32_mixed_sssf.npz` stores the FCC-32 global unpolarized equal-time
  neutron-projection check shown in the Supplement.

Run the primary calculation from `twist_qsi_demo/`:

```bash
python submission/scripts/fcc32_prl_figures.py
```

The dense 2,970-state diagonalizations take roughly two minutes on the current
workstation.

## Scientific scope

Every many-body spectrum in the Letter is obtained from the FCC-32
second-/third-order ice-manifold operator. The minimal polarization-parity
mask is proven equal to explicit zero-transport row filtering through third
order on this cluster; it is not claimed to be an all-order projector.

For the published NLC-A parameters `(Ja,Jb,Jc)=(0.050,0.021,0.004) meV`, the
clean FCC-32 gauge peak is 7.1 mK. The candidate
`A*=(0.0464354,0.0266142,0.0096142) meV` preserves `sum(J_alpha^2)` and
`Jb-Jc` while placing that peak at 25 mK. A* is a target for a new
seventh-order NLC calculation, not a completed NLC refit.

The FCC-32 DSSF is the four-sublattice trace of the longitudinal `Szz`
correlation. It contains no local-to-global axis projection or neutron
polarization tensor. The fully coherent `Szz` source is identically zero at
the exact cluster momenta by the ice constraint, while the sublattice trace
is finite. A transverse spin flip leaves the ice manifold
(`P S_i^\pm P = 0`), so a dynamical spinon response would require a much
larger microscopic defect-sector calculation. The Supplement keeps the
finite-angle local-to-global projection only as a separate FCC-32 equal-time
check.
