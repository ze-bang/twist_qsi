# Paper B — Is the 25 mK anomaly of Ce2Hf2O7 the emergent photon?

**One-line thesis.** Naive periodic ED puts a *torus artifact* peak at ~29 mK,
right on the measured 24 mK anomaly; remove the artifact and the NN model
predicts 6-8 mK, with a certified peak-ratio ceiling of 0.198 against a measured
0.374. Plus two model-free constraints (Pauling entropy budget, photon T^3 tail)
that decide the question without any Hamiltonian.

This is the materials paper. It is deliberately framed as **"not accounted for
by the NN model"**, NOT as a rigorous exclusion — see "The claim you cannot make".

## Status of every number in the draft

### LIVE

| Number | Source |
|---|---|
| T2 = 24.19 mK, T1 = 64.65 mK, ratio 0.3742 | `outputs/ce2hf2o7_peak_analysis.json` |
| NLC-A periodic 28.7 mK / masked 6.0 mK; (-0.152, 0.085) 37.5 / 8.1 mK | `outputs/ce2hf2o7_peak_scan.json` |
| mK table (0.71/7.52, 6.01/17.40, 7.72, 8.11) and masked/g6 column | `outputs/qed_windfree_*.json` |
| 80 grid points, 66 certified, max ratio 0.198 at (+0.05, +0.085); next 0.121, 0.111 | `outputs/wp2_gate/wp2_ratios.json` |
| best RMSE point (-0.156, +0.115), Jzz = 0.236 K = 0.0203 meV, peaks 3.53 / 67.2 mK | `outputs/ce2hf2o7_scan_ranking.json` |
| log-log peak exponents 1.80 periodic / 2.79 masked | `paper/DATA_PROVENANCE.md` |
| host coefficients c16 = 1.070, c32 = 0.521 | `SIMULATION_PLAN.md` |
| L-truncation error up to 40% low at strong coupling | `SIMULATION_PLAN.md` |
| model_overlap_min 0.887 (-0.05) -> 0.360 (-0.18); band fails for Jpm <= -0.24 | `outputs/ce2hf2o7_peak_scan.json` |
| raw flux sign inverted over the whole (Jx, Jy) plane | `paper/flux_methods.tex` |
| spinon peak untouched by the mask (<1% out to Jpm = -0.09) | `paper/pedagogical.tex` |
| NLCE order-5: s2 2.2% vs A 3.7% vs B 5.1%; high-T tail sees only sum J^2 | external NLCE run (memory `ce2hf2o7-ring-exchange-vs-t2`) |
| digitized trace starts 20.6 mK at C = 1.933, peaks 2.25 at 23.7 mK | `campaign/data/ce2hf2o7_smith2025_digitized.csv` |

### PENDING

1. **The entropy integral, Eq. (test) — the one new piece of analysis.**
   Not evaluated. Script written: `analysis/entropy_budget.py`.

       sbatch analysis/run_entropy.sbatch      # 5 min, 1 core

   It prints the measured part plus BOTH tail models, because the missing
   sub-20 mK data leaves a systematic comparable to S_ice itself
   (T^3 tail ~ 0.64, linear tail ~ 1.93, against S_ice = 1.687 J/mol/K).
   The likely honest outcome is "inconclusive without author data below 20 mK" —
   still worth publishing, because it names the measurement that decides it.

2. **Reconcile the two ratio ceilings.** `wp2_gate/wp2_ratios.json` gives max
   certified ratio 0.198; `ce2hf2o7_peak_analysis.json` gives largest reachable
   0.0501 with shortfall 7.48x. Same physics, different peak-identification
   rules. Pick one rule, rerun both, quote one number. The draft flags this in
   red — do not clear the flag by keeping the more favourable number.

3. **FCC-32 leg.** The whole coupling-plane scan is cubic-16, the only cluster
   where the full Jpmpm basis is affordable. `outputs/fcc32_xyz_grid/` does not
   exist and `outputs/fcc32_smoke/` is empty. The draft handles this by arguing
   direction (c32 < c16 pushes the peak colder, worsening the discrepancy),
   which is sound, but a referee will ask for at least one FCC-32 point at NLC-A.

## The claim you cannot make

**Do not write "rigorous exclusion of the NN model".** The campaign notes
originally framed WP2 that way and the numbers do not support it:

- the ceiling 0.198 against a measured 0.374 is a factor 1.9, not orders of
  magnitude;
- the host coefficient alone is a factor ~2, and it is not a bounded error — it
  is an unconverged extrapolation;
- 14 of 80 grid points are uncertified, and they sit at the strong-coupling end
  where the ratio would be largest.

The defensible claim is the one in the abstract: *not accounted for* within NN
XYZ, with the artifact coincidence explaining why earlier small-cluster
calibrations looked like agreement. The decisive experiment (C_mag below 20 mK,
exponent + entropy) is the constructive contribution.

## Relation to the other two drafts

- Paper A supplies the reason the low-T tail is worth measuring in more than one
  channel: in **octupolar** QSI (the live possibility for Ce2Hf2O7) the photon is
  neutron-dark, so thermodynamics and Raman/ultrasound are the only access.
- Paper C supplies the coefficient in the T^3 law: the emergent light speed c,
  measured nonperturbatively from the electric-flux sector energies.

## Build

    sbatch build.sbatch

`refs.bib` was drafted partly from memory — verify every entry before submission.
