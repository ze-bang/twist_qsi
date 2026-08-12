# Data provenance for the FCC-32 masked-Feshbach paper

Audited 2026-08-11 against `paper/main.tex`. The active manuscript has three
figures and no numerical tables. Every newly computed numerical dataset uses
the same 32-site FCC cluster; no cubic-16, counterterm, or polar-pullback
output enters an active figure or claim.

## Figure 1 — `fig_fcc32_thermodynamics.pdf`

Producer: `campaign/make_fcc32_paper_figures.py`.

The input files are `campaign/outputs/fcc32_thermodynamics/thermo_*.{json,npz}`.
They are produced by `campaign/run_fcc32_thermodynamics.py` and contain paired
periodic and transport-masked spectra. Each pair uses the identical FCC-32
host, depth-`d=1` 140,442-state microscopic space, shared Feshbach anchor
energies, and all 2970 ice-descended self-consistent roots. A coupling is
plotted only if the JSON root-count gate records both spectra as complete.

| Panel | Content | Additional input |
|---|---|---|
| (a) | FCC-32 ice-band heat capacity at `Jpm/Jzz=+0.046` | `campaign/data/huang_2018_qmc_jpm_0p046.csv` |
| (b) | Dominant paired heat-capacity peak over the negative-coupling root-complete range | Analytic guides `4 Jpm^2/Jzz` and `12 |Jpm|^3/Jzz^2` |

When more than one reproducible maximum exists, the producer selects the one
with the greatest heat-capacity height from the saved curve; it does not
silently switch to the lowest-temperature shoulder. A log--log fit to the
first three couplings (`0.01--0.03`) gives powers `1.8023` (periodic) and
`2.7904` (masked).

The digitized comparison is from Huang *et al.*, Phys. Rev. Lett. 120, 167202
(2018), Fig. 1(b). Its low-temperature peak is at
`T/Jzz=0.001100447383`. The paired FCC-32 peak temperatures are
`0.006590442862` (periodic) and `0.0005715559718` (masked). The numerical
spectra include only the low-energy
ice-descended band and are not presented as the full high-temperature
thermodynamics of the `2^32`-state spin system.

## Figure 2 — `fig_fcc32_dssf.pdf`

Producer: `campaign/make_fcc32_paper_figures.py`.

| Method | Input at `Jpm/Jzz=-0.20` | Construction |
|---|---|---|
| Naive periodic | `campaign/outputs/wp3_dssf/dssf_periodic_fcc32_L1_-0p200.json` | Direct sparse diagonalization of the depth-`d=1` 140,442-state periodic Hamiltonian |
| Transport masked | `campaign/outputs/wp3_dssf/dssf_fcc32_L1_-0p200_pp+0p000.json` | Bordered transport-sector Hamiltonian whose Schur complement is the masked Feshbach block |

Both calculations retain 120 eigenstates, all eight momenta allowed by
FCC-32, and four pyrochlore-sublattice longitudinal probes. At Gamma, the
integrated inelastic weights are `0.3125849374` (periodic) and
`1.417981913e-29` (masked). At the seven nonzero momenta they are
`0.1778666--0.2293784` (periodic) and `0.2140561--0.2987769` (masked).

The independent repeat at `Jpm/Jzz=-0.05` uses files with the corresponding
`-0p050` tag. Its Gamma weights are `0.4878642` (periodic) and
`1.361249e-30` (masked); the repeated calculation is reported in the text but
is not added as another figure panel.

## Figure 3 — `fig_loop_ratios.pdf`

Producer: `campaign/make_clean_paper_figures.py`.

| Content | Inputs |
|---|---|
| Complete virtual-depth `d=2` ratios at seven couplings | `campaign/outputs/wp1_tomography/tomo_fcc32_L2_-0p{030,060,100,150,200,250,300}.json` |
| Matched virtual-depth `d=3` checks at `-0.20` and `-0.30` | `campaign/outputs/wp1_tomography/tomo_fcc32_L3_-0p200_n512.json`; `tomo_fcc32_L3_-0p300_n512.json` |
| Leading perturbative curves | Analytic loop ratios in `paper/main.tex` |

Here `L2` and `L3` in historical filenames mean virtual-space depth. The
manuscript denotes that quantity by `d`; it reserves `L` for the number of
flipped spins, equivalently the bond perimeter, of a connected degree-two
simple loop. At `Jpm/Jzz=-0.30`, the matched `d=3` ratios are
`K8/K6=0.262297`, `K10/K6=0.0617438`, and `K12/K6=0.0203689`.

## Scope

The active results are finite-volume and virtual-depth truncated. Larger
hosts, complete `d=3` FCC-32 tomography, a thermodynamic-limit Wilson-loop
law, or a material-specific spectrum would require new campaigns and are not
claimed in the paper.
