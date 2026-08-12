# Single paper

The sole active manuscript is:

> **Removing Torus Artifacts Reveals Nonperturbative Loop Locality in Quantum
> Spin Ice**

Its narrative has one method and one physics result:

1. transport-masked Feshbach downfolding removes the false second-order torus
   process and restores the physical third-order thermodynamic scale;
2. the corrected operator map shows that longer simple loops remain strongly
   subordinate to the six-site ring, even where bare perturbation theory
   predicts proliferation.

Every numerical result in the active paper is on the 32-site FCC cluster.
Figure 1 contains only the paired FCC-32 thermodynamic benchmarks (there is no
method cartoon), Fig. 2 compares periodic and masked longitudinal response at
all eight FCC-32 momenta, and Fig. 3 gives the FCC-32 loop hierarchy.

Build from this directory with `latexmk -pdf main.tex`. The active files are
`main.tex`, `references.bib`, `figs/fig_fcc32_thermodynamics.pdf`,
`figs/fig_fcc32_dssf.pdf`, and `figs/fig_loop_ratios.pdf`. Figure inputs are
documented in `DATA_PROVENANCE.md`.

No counterterm or polar-pullback construction appears in the manuscript.
Older figure files in `figs/` are archived campaign assets and are not active.
