# Data index

One line per retained directory: what it is, which script produces it, and
what consumes it.  Directories are the unit of provenance; every scanner
writes one json (+npz) per parameter point, so partial grids resume by
rerunning missing points.

## Current campaign (mask + fold instruments)

| directory | contents | producer | consumers |
|---|---|---|---|
| `jpmpm_fold_plane/` | full-space fold over the `(Jpm, Jpmpm)` 8x7 grid, cubic-16: fold/raw peaks, masked roots, ice weights; the exact calibration grid for the remote FCC-32 phase | `run_jpmpm_fold_plane.py` | `make_jpmpm_curves_figure.py`, `make_jpmpm_heatmap.py` |
| `two_sided_fold/` | counterterm-free thermodynamics on the XXZ line, all four definitions' curves + third-peak taxonomy figures | `run_two_sided_fold.py` | `make_two_sided_peaks_figure.py`, `make_third_peak_figure.py`, `paper/pedagogical.tex` |
| `multi_anchor_band/` | self-consistent masked bands: FCC-32 L=1 coupling scan + cubic-16 L-ladder calibration + two-cluster peaks figure | `run_multi_anchor_band.py` | `make_fcc32_peaks_figure.py` |
| `fcc32_sector_grounds/` | per-sector bordered ground energies (the no-crossing verdict), L=1 grid + L=2 confirmation | `run_fcc32_sector_grounds.py` | pedagogical Sec. "first steps onto FCC-32" |
| `masked_band_complete/` | complete exact masked bands (BIC-recovered) on cubic-16, the oracle for every truncated instrument | `run_masked_band_complete.py` | fold, plane scanner oracles, `make_third_peak_figure.py` |
| `wf_thermodynamics/` | raw and `H+C` full fixed-Sz spectra per coupling (fold inputs) | `run_wf_thermodynamics.py` | `run_two_sided_fold.py` |
| `fcc32_band/` | the superseded fixed-anchor FCC-32 lane; retained as the failed-calibration reference | `run_fcc32_band.py` | pedagogical lesson box |

## Frozen validation era (retained: cited by main.tex figures or docs)

| directory | why retained |
|---|---|
| `specific_heat/` | the full-ensemble C(T) scan behind the measured power laws |
| `method_comparison/`, `counterterm_spine/`, `counterterm/` | spine/comparison data feeding `fig_peak_tracking` / `fig_summary` via `make_figures.py`, `refine_peaks.py` |
| `gamow/` | zero-width verdicts (photon never decays) cited in pedagogical |
| `exact_counterterm/`, `feshbach_anatomy/`, `truncated_feshbach/`, `strong_coupling/`, `polar_stitch/`, `jpm_levels/`, `anchor_convergence/` | numbers quoted in the paper/pedagogical record |
| `twist_average_curves_m*.npz`, `fig_twist_average.*` | the observable-twist-averaging falsification (floor 1/3, wrong power law); curves retained so the figure reproduces without the 35 MB spectra cache |
| `ce2hf2o7_*`, `bulk_pt_*`, `bw_greyzone_summary.json` | material section and grey-zone ladder of record |

## Removed (recoverable from git history, tag `38800f4` and earlier)

Dead ends whose conclusions are recorded in `paper/pedagogical.tex` and the
commit log: `cache/twist_avg/` (35 MB of per-twist spectra for the falsified
averaging scheme), `counterterm2/`, `counterterm3/` (loop-catalogue
counterterms: inadequate, moved the physical map by up to 42%),
`dressed_sweep/`, `clean_sweep/`, `gapframe_sweep/`, `corner_coherence/`,
`corner_decomposition/` (frame-era diagnostics: all frames fail together;
superseded by the frame-free map), `linearized_counterterm/`, `logs/`.
