# Submission checklist

## Required before upload

- [ ] Replace affiliation placeholders in `main.tex`, `supplement.tex`, and
  `cover_letter.tex`.
- [ ] Replace the acknowledgments placeholder and add funding and data/code
  availability statements.
- [ ] Confirm author list, order, ORCID records, and corresponding-author
  email.
- [ ] Confirm permission and attribution requirements for the digitized
  Smith et al. experimental points and NLC-A curve.
- [ ] Decide whether Ce2Hf2O7 should remain in the title/abstract scope.
- [ ] Select suggested and opposed referees in the APS submission portal.

## Verified in this package

- [x] All Letter many-body results use FCC-32 and its 2,970-state ice basis.
- [x] FCC-32 geometry census: 24 wrapping four-loops, 32 contractible
  hexagons, and 96 wrapping hexagons.
- [x] Explicit zero-transport filtering and the parity mask agree entrywise
  through third order with zero double-precision residual.
- [x] Contractible hexagon amplitudes are unchanged by projection.
- [x] FCC-32 projected spectra and heat-capacity peaks collapse in units of
  `g_hex`; periodic results do not.
- [x] Figure 3 uses only all eight allowed FCC-32 momenta and a linear
  frequency axis, with no neutron-polarization factors.
- [x] Local `[111]` axes remain inside the global unpolarized neutron
  projector in the FCC-32 equal-time SI calculation.
- [x] The digitization procedure and visual-comparison limitation are
  recorded in the data file and Supplement.
- [x] A* is labeled as a high-temperature-moment-constrained candidate, not
  a seventh-order NLC fit.

## Needed before claiming a joint material fit

- [ ] Run A* through the same seventh-order NLC implementation used for the
  published high-temperature curve.
- [ ] Fit tabulated Ce2Hf2O7 data with uncertainties and a declared objective
  function.
- [ ] Restore the full principal-axis XYZ Hamiltonian beyond the U(1)
  reduction.
- [ ] Compute the FCC-32 microscopic defect-sector transverse DSSF if a
  dynamical spinon/neutron comparison is retained.
- [ ] Add Ce form factor, anisotropic g tensor, thermal populations, and
  instrumental resolution for an absolute neutron prediction.
- [ ] Refine the character grid beyond `M=2` before making all-order claims.
