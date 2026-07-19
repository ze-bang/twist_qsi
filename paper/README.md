# Paper status

`main.tex` and `supplement.tex` are generated around the active validation
campaign.  They intentionally do not claim a Ce2Hf2O7 fit or converged
FCC-32 dynamics because the order, character-grid, external QMC, and
full-Hamiltonian FCC-32 gates have not all passed.

Build with:

```bash
latexmk -pdf -cd paper/main.tex
latexmk -pdf -cd paper/supplement.tex
```

The current package is a rigorous working draft, not yet scientifically ready
for PRL submission.  The exact missing calculations are listed in the final
table of the Supplemental Material and in `SIMULATION_PLAN.md`.
