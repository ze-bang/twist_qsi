# Paper status

`main.tex` and `supplement.tex` are generated around the frozen cubic
validation campaign.  The two remaining production deliverables are a Figure
1 made entirely from converged FCC-32 XYZ products and a Ce2Hf2O7 parameter
fit over those products.  The current paper intentionally does not claim
either result: the archived FCC-32 curves are order-three ice-space
diagnostics, and the available material data are a raster extraction without
tabulated uncertainties.

Build with:

```bash
latexmk -pdf -cd paper/main.tex
latexmk -pdf -cd paper/supplement.tex
```

The current package is a rigorous working draft, not yet scientifically ready
for PRL submission.  The exact missing calculations are listed in the final
table of the Supplemental Material and in `SIMULATION_PLAN.md`.
