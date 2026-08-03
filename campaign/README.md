# Active campaign

The cubic-16 exact-band calculation and the cubic/FCC topology audit are frozen
method validation.  The remaining production campaign has two outputs:

1. a Figure 1 built entirely from converged FCC-32 exact-band XYZ results;
2. a Ce2Hf2O7 heat-capacity fit over converged FCC-32 model curves.

`fcc32_xyz_manifest.json` defines the Hamiltonian, grids, convergence gates,
and Figure 1 products.  The microscopic builder and cubic-16 driver now include
`Jpmpm=(Jb-Jc)/4` on the full spin basis because pair flips break the conserved
spin component.  FCC-32 has a 2,970-dimensional ice reference space, but the full
32-spin microscopic Hilbert space is much larger; legacy order-three matrices
on the ice space are topology diagnostics and are not production inputs.

`material_fit_config.json` records the data source, parameter domain, published
seeds, and model-curve location.  A valid curve in
`outputs/fcc32_xyz_grid/*.npz` must contain:

```text
method = "fcc32_exact_winding_free_xyz"
n_sites = 32
Ja_meV, Jb_meV, Jc_meV
temperature_K, heat_capacity_J_molCe_K
character_M, character_converged, complement_converged
```

Run `make material-fit` to rank available curves.  The driver rejects wrong
methods, clusters, and unconverged character or complement calculations.  The
current Smith et al. curve is a reproducible v5 raster extraction without
tabulated uncertainties, so its result is explicitly exploratory until author
data are supplied.

The earlier `run_validation.py`, `run_nonperturbative.py`, and `run_dssf.py`
remain the reproducible cubic validation path.  They do not produce an FCC-32
material fit.

`twist_averaging.tex` (compiled `twist_averaging.pdf`) documents the
observable-level twist exploration
(`run_twist_average.py`, `twist_average_scan_m*.json`,
`fig_twist_average.pdf`): why solve-per-twist averaging keeps the g4 power law
with at best 1/3 of the artifact (optimal single twist `(pi/2)^3`), and the
regimes where twist averaging is legitimate.

## The paper: the self-consistent counterterm (July 2026)

`paper/main.tex` and `paper/supplement.tex` are the only write-ups; the earlier
`resonance_extension.tex` and `method_failure.tex` were folded into the
supplement and deleted.  Every number in the main text comes from these four
scripts:

| script | what it produces | output |
| --- | --- | --- |
| `run_counterterm_spine.py` | **the protocol**: `C = -(1-Delta)F(z_0)` with the Brillouin-Wigner anchor `z_0 = E_0(H+C)` solved by secant to 1e-13; complete 65536-level spectrum, C(T), S(T), labelled peaks, deviation from the exact masked reference, and the symmetrised variant for comparison | `outputs/counterterm_spine` |
| `run_anchor_convergence.py` | the central check: ground-multiplet splitting against the anchor residual along the secant trajectory, showing the splitting is a convergence artifact (linear over ten decades) and not an intrinsic residue | `outputs/anchor_convergence` |
| `run_dssf_counterterm.py` | frame-free longitudinal DSSF, mask off against mask on, using `P S^z_q P` (exactly diagonal in the ice basis -- no isometry) | `outputs/dssf_counterterm_*.npz` |
| `make_spine_table.py`, `make_anchor_figure.py`, `make_spine_tracking_figure.py`, `make_figures.py` | `paper/tab_spine.tex` and the figures | `paper/`, `paper/figs` |

```bash
python campaign/run_counterterm_spine.py                 # all couplings, ~95 s each
python campaign/run_anchor_convergence.py -0.05 -0.20 -0.30
python campaign/run_dssf_counterterm.py 0.046
python campaign/make_spine_table.py && python campaign/make_anchor_figure.py
python campaign/make_spine_tracking_figure.py && python campaign/make_figures.py
```

**The trap this campaign fell into.** `run_method_comparison.py` iterated the
anchor at most three times, leaving a residual of ~1e-2 at `Jpm = -0.20`.  The
resulting splitting of the mask-protected ground multiplet was reported as an
intrinsic "anchoring residue" and repaired by averaging the six lowest levels
(`run_sector_symmetrisation.py`).  It was an unconverged fixed point: with a
secant step the residual reaches 1e-13 in ten solves, the splitting falls to
1e-12, and the symmetrisation makes the ring anomaly *worse*.  Plain iteration
converges at rate ~(1-Z) and cannot get there.  Those two scripts are kept
because `fig_peak_tracking.pdf` panel (b) is the measurement that demonstrates
this, and because their cached spectra supply the reference and periodic
comparisons.

## Large-|Jpm| frame-free campaign (July 2026)

These drivers back Secs. VII-XII of the supplement, all cubic-16 unless
stated.  Every one of them is frame-free: none uses a Riesz projector, a Löwdin
pullback or a Gram matrix.

| script | what it produces | output directory |
| --- | --- | --- |
| `run_feshbach_anatomy.py` | exact ice-block Feshbach map, its string-class decomposition, dressed ring amplitude `g`, RK potential `V`, residue `Z`, masked band and ring peak | `outputs/feshbach_anatomy` |
| `run_strong_coupling_gs.py` | exact ground-multiplet observables of the raw cluster (ice weight, defects, hexagon flux, winding amplitude, transverse condensate) | `outputs/strong_coupling` |
| `run_corner_coherence.py` | corner-closure spectrum, collective/individual transport measures, greedy-selection experiment | `outputs/corner_coherence` |
| `run_corner_decomposition.py` | corner returns of the same band for `H` and for `H+C` | `outputs/corner_decomposition` |
| `run_winding_residual.py` | transporting fraction of the ice-block map before/after loop-class counterterms | `outputs/winding_residual` |
| `run_counterterm_observables.py` | observables and coherence of the loop-class counterterm Hamiltonian | `outputs/counterterm_observables` |
| `run_exact_counterterm.py` | **the protocol**: `C = -(1-Delta)F(z_0)` in-block counterterm, spectrum of `H+C`, agreement with the masked Feshbach roots | `outputs/exact_counterterm` |
| `run_truncated_feshbach.py` | spinon-truncated map on cubic-16 (validation) and FCC-32 (size check) | `outputs/truncated_feshbach` |
| `run_fcc32_band.py` | full masked ice-block band of a cluster from the truncated map | `outputs/fcc32_band` |
| `summarize_large_jpm.py` | consolidated table of all of the above | stdout |
| `make_large_jpm_figures.py` | the four figures of the note | `paper/figs` |

Typical invocations:

```bash
python campaign/run_feshbach_anatomy.py --no-oracle -0.05 -0.20   # drop --no-oracle to check roots against the exact spectrum
python campaign/run_exact_counterterm.py -0.05 -0.20
python campaign/run_truncated_feshbach.py fcc32 1 60 -0.05 -0.20
python campaign/run_fcc32_band.py fcc32 1 -0.05
python campaign/summarize_large_jpm.py
```

### Specific-heat scan (July 2026)

| script | what it produces | output |
| --- | --- | --- |
| `run_specific_heat_scan.py` | complete 65536-state spectrum (every S^z sector, Klein-blocked), C(T)/N for raw `H` and winding-free `H+C`, all local maxima | `outputs/specific_heat` |
| `make_specific_heat_figure.py` | `fig_specific_heat.pdf` + the peak table and power-law fits | `paper/figs` |

```bash
python campaign/run_specific_heat_scan.py 0.05 -0.05 -0.20 -0.50   # ~50 s per coupling
python campaign/make_specific_heat_figure.py
```

The gauge anomaly is the tallest maximum below the spinon one, and counts as
resolved only if the curve dips below 70% of the smaller peak between them --
otherwise it is a shoulder, and is reported as "merged" rather than as a peak
position.
