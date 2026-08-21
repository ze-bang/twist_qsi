# A single renormalized energy denominator controls the loop expansion of QSI

**Replaces** `paper_A_selection_rules`, `paper_B_material_verdict`,
`paper_C_emergent_maxwell` — all moved to `archive_superseded_2026-08-17/`
with a note on why each was dropped.

**The claim, in one line.** The nonperturbative loop expansion has the same
operator content as the perturbative one and differs by a single number:
`K_L = K_L^(0) mu^(L/2-1)`, i.e. every energy denominator is renormalized
`Jzz -> D_eff(Jpm)`. Consequences: the photon scale is 4-11x colder than the
standard estimate, the two specific-heat peaks cannot merge, and the Ce
pyrochlore gauge feature has never been inside a measured temperature window.

## Provenance — every number in the manuscript

| Number | Source | Job |
|---|---|---|
| collapse table (mu_6..mu_12, 7 couplings, depths 2 and 3) | `campaign/analyse_loop_collapse.py` | 19163192 |
| K_L raw amplitudes, FCC-32 depths 2/3 | `campaign/outputs/wp1_tomography/tomo_fcc32_*.json` | pre-existing |
| M_0 = 0, M_1 = 12\|Jpm\|^3 exactly; D_eff both clusters | `campaign/analyse_mu_denominator.py` | 19191293 |
| spectral anatomy (coherence 1.3-4.5%, centroid rise) | `campaign/analyse_mu_anatomy.py` | 19163466 |
| BW vs Hermitian normalization, cubic-16 exact | `campaign/run_map_normalization.py` | 19160491 |
| continuum edge / band bottom vs coupling | `campaign/outputs/fcc32_thermodynamics/thermo_*.json` | pre-existing |
| raw vs winding-free C(T) peaks (the merging table) | `campaign/outputs/specific_heat/heat_*.json` | pre-existing |
| Ce2Hf2O7 T1/T2, 20.6 mK data floor | `outputs/ce2hf2o7_peak_analysis.json`, `campaign/data/` | pre-existing |
| host coefficients 1.070 / 0.521 | `SIMULATION_PLAN.md` | pre-existing |
| Figures | `campaign/make_loop_amplitude_figure.py`, `campaign/make_denominator_figure.py` | 19163541, 19192695 |

## PENDING (marked red in the PDF)

The entropy integral on the published Ce2Hf2O7 trace. Script exists at
`archive_superseded_2026-08-17/paper_B_material_verdict/analysis/entropy_budget.py`;
move it here before submission. Expect an inconclusive result — the digitized
data start at 20.6 mK and the unmeasured tail is worth 38-114% of S_ice — which
is itself the point worth reporting.

## Claims deliberately NOT made, and why

- **"The process takes an expensive route."** Withdrawn. There is no route
  selection; all intermediate states are summed and the cheap ones carry the
  largest weight. What is true is that `sum_k w_k = 0` exactly, so degenerate
  intermediates would give zero amplitude, and `D_eff` is a weighted mean, not
  the energy of any state. Sec. IV says this.
- **"D_eff is not the spinon gap."** Softened. A weighted mean necessarily
  exceeds the minimum of the same distribution, so the bare comparison is
  vacuous; only the growth of the ratio (1.29 -> 2.90) is content. Figure 2(b)
  is labelled accordingly.
- **"Rigorous exclusion of the NN model for Ce2Hf2O7."** Not claimed. The
  argument is that the gauge feature is below every measured temperature, which
  is stronger and does not depend on the unconverged host coefficient.
- **The emergent-permittivity / flux-sector story.** Falsified (29/29 fits
  fail); not in this paper. See the archive.
- **Photon-number parity selection rules.** Largely known; not in this paper.

## Known soft spots a referee will find

1. `mu` is not depth converged (0.274 -> 0.293 from depth 2 to 3 at x=0.30).
   Stated in Sec. VII; direction is favourable (true suppression is weaker) and
   nowhere near enough to change the conclusions.
2. Intensivity rests on one doubling, 16 -> 32 sites.
3. The hexagon departs from the collapse by up to 44% in amplitude. The law is
   about L >= 8.
4. `K_L/K_6` ratios are normalization sensitive (1.16 vs 3.56 at x=0.20 for
   L=6 vs L=8) even though `mu` is not. Any hierarchy statement must quote its
   normalization — this also bears on the companion manuscript's Fig. 3.

## Build

    sbatch build.sbatch      # 12 pages, clean

`refs.bib` was drafted partly from memory — verify every entry before
submission, especially `SimonPatriKim2022`, `Desrochers2023`, `Poree2025`.
