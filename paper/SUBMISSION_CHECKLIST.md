# PRL submission campaign

## Frozen method validation

- [x] Loop enumeration and transported-dipole classification are tested on
  cubic-16 and FCC-32.
- [x] Cubic-16 exact-band character convergence passes at `M=3 -> 4` with
  0.013% centered operator change.
- [x] Full-Hilbert cubic-16 thermodynamics uses
  `H_wf = H(0) + W0 [hbar_M - h(0)] W0^dagger` and retains the microscopic
  high-temperature complement exactly.
- [x] Periodic and winding-free cubic-16 low-frequency `Szz` use one
  microscopic probe and only allowed momenta; the omitted complement begins
  above the plotted window.
- [x] Huang et al. QMC thermodynamic provenance and the residual cubic-size
  discrepancy are reported rather than hidden.

## Remaining deliverable 1: FCC-32 Figure 1

- [x] Extend the microscopic source and cubic-16 polar pullback to the full
  ABC/XYZ Hamiltonian, including `Jpmpm=(Jb-Jc)/4`; finite `Jpmpm` uses the
  complete 65,536-state basis rather than fixed total `Sa`.
- [ ] Produce restartable FCC-32 exact-band points for matched `M=2,3,4`
  grids and pass band-overlap, Ritz-residual, and character-convergence gates.
- [ ] Compute the FCC-32 stochastic complement and pass the thermal trace
  error gate over the plotted temperature range.
- [ ] Compute periodic and winding-free `Szz` at all eight FCC-32 translation
  momenta with one probe convention and verified sum rules.
- [ ] Replace every cubic-16 panel and value in Figure 1 with those converged
  FCC-32 products.  Order-three ice-space curves are not eligible.

## Remaining deliverable 2: Ce2Hf2O7 fit

- [x] Smith et al. v5 Figure 2 data extraction and published A/B/QMC seeds are
  reproducible in `campaign/data/`.
- [x] The fit driver rejects non-FCC, legacy, or unconverged model curves.
- [ ] Obtain tabulated experimental `Cmag` and uncertainties from the authors;
  raster data support exploratory ranking only.
- [ ] Generate an adaptive FCC-32 XYZ grid around A, B, and the QMC seed,
  including both `Jpm` and `Jpmpm`.
- [ ] Rank the converged curves, refine each local minimum, and report profile
  intervals, parameter correlations, and sensitivity to the temperature window.
- [ ] Verify the fitted parameter sets against entropy and neutron-scattering
  observables not used in the heat-capacity objective.

## Submission metadata

- [ ] Affiliation, acknowledgments, funding, author contributions, and data
  statement are completed.
- [ ] Claims and captions are re-audited against the final FCC-32 and material
  fit reports.
