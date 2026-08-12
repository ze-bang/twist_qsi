# Submission checklist

Updated 2026-08-11 for the three-figure, FCC-32-only paper.

## Physics and numerical checks

- [x] Define every recurring symbol, including loop perimeter `L` and virtual
  depth `d`; do not use the ambiguous historical `ell` notation.
- [x] Use transport-masked Feshbach as the sole method spine.
- [x] Remove counterterm and polar-pullback results from all claims and figures.
- [x] Benchmark paired 2970-root FCC-32 periodic and masked spectra against
  sign-free QMC and over the full root-complete coupling range.
- [x] Explain how `Szz(q,omega)` is evaluated with bordered matrices whose
  Schur complements are masked Feshbach blocks.
- [x] Define simple loops and exclude vertex-joined/disconnected composites.
- [x] Derive the leading analytic `K_L` coefficient and successive ratios.
- [ ] Gate every Fig. 3 physics statement on the matched, depth-specific
  `d=1,2,3,4` convergence ladder at `|Jpm|/Jzz=0.30`; distinguish convergence
  of successive long-loop ratios from convergence of `K8/K6`.
- [x] Separate indirect neutron diagnostics from direct Wilson-loop protocols
  in programmable gauge simulators.

## Presentation checks

- [x] Delete the method cartoon; use only FCC-32 thermodynamics, FCC-32
  all-momentum response, and the FCC-32 loop-ratio figure.
- [x] Remove redundant numerical tables and the causal-ladder figure.
- [x] Record all active data inputs in `DATA_PROVENANCE.md`.
- [ ] Build the revised FCC-32 manuscript with REVTeX and clear undefined references, citations, and
  overfull boxes.
- [ ] Inspect every revised rendered page at publication scale.

## Author-supplied submission metadata

- [ ] Acknowledgments and funding identifiers.
- [ ] Author-contribution statement.
- [ ] Public repository or archive DOI for the data-availability statement.
