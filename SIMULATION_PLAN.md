# Validation plan: a convergent winding-free band protocol

## Campaign status and goal (updated 2026-08-09) — physics-first remote phase

**The method, current form.**  The optimized protocol is the direct
deletion of every ice-block matrix element connecting different transport
(polarization) sectors, applied to the exact Feshbach map `F(z)` (the bare
ice block is diagonal, so only the folded map exposes the amplitudes to be
deleted).  The character-grid construction below survives as the derivation
and as a validation oracle: the `M=2` corner average equals the parity mask
exactly.  Full-range thermodynamics is the two-sided fold; large-cluster
band spectra come from multi-anchor interpolated self-consistency and
per-sector bordered matrices, all under graded `L`-truncation with the
cubic-16 exact grids as calibration gates.  Conceptual frame (pedagogical
Sec. "the gauge theory knew"): the mask enforces, at finite size, the
emergent electric one-form symmetry that the thermodynamic limit enforces
dynamically; the winding artifact is a flux-sector tunneling of virtual
charges, exponentially small in circumference.

**State of the results (through 2026-08-08).**
- Instruments, calibration-gated: two-sided fold (`run_two_sided_fold.py`;
  weighted variant falsified), bordered sector solver
  (`run_fcc32_sector_grounds.py`), multi-anchor band
  (`run_multi_anchor_band.py`, ring peak 0.3-1% vs exact), DSSF sweep
  (`run_dssf_sweep.py`: S^zz fold + exact Sz=+-1 Lanczos S^xx).
- cubic-16 `(Jpm, Jpmpm)` plane complete (56 points): certified two-peak
  ratio ceiling ~0.10 at (-0.24, 0.15); re-entrant dissolution boundary;
  near-CHO ratio 0.332 at (-0.125, 0.30) NOT certified (27/90 band).
- FCC-32 XXZ line: no 6->2 crossing anywhere (cubic-16 host pathology);
  L=1 band at 13 couplings; host coefficients c16 = 1.070, c32 = 0.521;
  L-truncation error measured (L=1 ~40% low at strong coupling, L=2
  11-19%, L=3 <1%).
- Material confrontation (cubic-16): at the fitted Ce2Hf2O7 couplings
  (-0.125, 0.085) the winding-free gauge peak sits at ~7 mK (Jzz = 0.05
  meV; ~3 mK anchored on the measured 65 mK spinon peak) while NAIVE
  periodic ED puts its artifact peak at ~29 mK — on top of the observed
  25 mK anomaly.  Identifications of 25 mK as "the photon" calibrated on
  small periodic clusters inherited the artifact.
- Loop anatomy (cubic-16, `run_feshbach_anatomy.py`): within a geometry
  class all amplitudes identical to 5e-15 (K_L is sharply defined);
  |g_eff|/g6 falls 0.81 -> 0.07 while long/hex rises 0.13 -> 1.17 and
  |V/g| stays <= 0.022: the effective gauge theory delocalizes rather
  than developing a potential term or an instability.

**The physics goal of this phase** (reframed 2026-08-09 after external
review): stop treating the mask as the headline and chase the two results
it enables — (A) *virtual spinons do not kill the emergent photon; they
delocalize its Hamiltonian* (loop-amplitude tomography, an emergent gauge
range xi_gauge, and whether the residue Z is the organizing variable);
(B) *does the nearest-neighbor XYZ model produce the 25 mK anomaly of
Ce2Hf2O7 as dressed gauge dynamics, or can it be rigorously excluded?*
(either outcome is publishable; the ORNL paper itself suspects beyond-NN
terms).  Decision tree: WP2 positive -> build the PRL around the material;
WP2 null -> build it around WP1, with WP2 as the material corollary.

### Work packages, in order

**WP0 — local pre-flight (cubic-16, hours; run before or alongside
cluster spin-up).**
- *0a, organizing-variable discriminator.*  Loop-class anatomy
  (K_6, K_8+/hex ratio, |V/g|, Z) on a POSITIVE-Jpm ladder
  (+0.02 .. +0.20, ~8 points; extend `run_feshbach_anatomy.py`) and at
  ~6 certified `(Jpm, pp)` points of the existing plane (full-65536 fold
  path of `run_jpmpm_fold_plane.py`).  Test: do long/hex and K_8/K_6
  collapse against (1-Z) across sign and pp, or against the bare
  couplings?  Current data is ambiguous: at matched |Jpm| ~ 0.05 the two
  signs give long/hex 0.317 / 0.321 despite different Z (0.929 / 0.890)
  — one ladder decides.  Collapse => "Z is the physical scaling
  coordinate" opens the paper; no collapse => organize WP1 by couplings.
- *0b, Gamma selection-rule generality.*  Verify on the 2970 FCC-32 ice
  states that the q=0 sublattice magnetizations are constant on each
  transport sector (pure combinatorics, minutes).  Decides whether the
  exact vanishing of the winding-free longitudinal Gamma response is a
  theorem of the construction or a cubic-16 accident — required before
  any selection-rule claim in WP3.

**WP1 — FCC-32 loop-amplitude tomography -> xi_gauge (headline A).**
Extract the masked ice-block fold entries at Jpm/Jzz = -0.03, -0.06,
-0.10, -0.15, -0.20, -0.25, -0.30; classify every entry by flipped-string
geometry: perimeter (6, 8, 10, 12), single-connected vs disconnected
(product processes), spatial diameter, and the Delta-rho = 0 transport
check.  Confirm the cubic-16 within-class degeneracy persists (if yes,
K_L needs no median).  Fit K_L ~ A exp(-(L-6)/xi_gauge); plot (i)
K_L/K_6 vs L per coupling, (ii) xi_gauge vs coupling, (iii) xi_gauge vs
(1-Z) and vs the inverse continuum-edge gap.  **Truncation gate
(critical): the longest loops converge SLOWEST in L-truncation and are
biased low — exactly the quantity of interest.**  Calibrate K_8/K_6 and
K_10/K_6 vs truncation depth on cubic-16 (L=1..4, L=4 exact) and quote
the transfer as the error bar; kill criterion: if the L=1 -> L=2 drift of
K_10/K_6 exceeds its inter-coupling variation, add L=3 spot checks before
claiming any xi trend.  Machinery: block-CG fold columns in the truncated
space (as in `run_multi_anchor_band.py`); 2970 columns per point.

**WP2 — decisive NN-XYZ test of the 25 mK anomaly (headline B).**
- Blocking step (unchanged from the 08-03 plan): extend
  `bfs_space`/`build_hamiltonian` in `run_truncated_feshbach.py` to pair
  moves (`S+S+`/`S-S-`, transport increments +-c_ij); gate against the
  exact cubic-16 56-point plane grid.
- Then scan the fit-constrained (Ja, Jb, Jc) region around the published
  seeds (NLC_A, NLC_B, QMC_ordered in `material_fit_config.json`) — the
  full XYZ triplet, not only the (Jpm, pp>=0) slice.  Per point record:
  n_found (dissolution), Z, gauge/spinon peak positions, and the entropy
  split (the gauge crossover must carry the Pauling R ln(3/2); the
  25 mK peak's measured entropy content is an independent discriminator).
- Question: does ANY certified parameter set consistent with the 65 mK
  anomaly + entropy put a gauge feature at 25 mK?  Prior from cubic-16
  is against it (ratio ceiling 0.098 vs measured 0.38; FCC-32 host
  coefficient pushes the gauge peak colder) — a clean null is the
  expected and equally publishable outcome: a rigorous NN-XYZ exclusion
  answering the experiment's open question.  The 08-03 items
  "dissolution boundary on FCC-32" and "gauge-peak surface" are absorbed
  into this scan; the typicality option stays optional.

**WP3 — sector-resolved S^zz(q, omega) on FCC-32 at the WP2 best point.**
The fit-independent prediction: which spectral weight is forbidden by
flux superselection, and where the pi-flux background redistributes the
allowed weight in momentum (contact with the GMFT symmetry-
fractionalization signatures).  Builds on the cubic-16 halves already in
`outputs/tier1_probes/` (Gamma-rule sticks, photon omega(X)) via the
per-sector bordered solvers; requires WP0b to have passed.

**WP4 — radial cuts through the XYZ near-degeneracy ring (last; cheap).**
3-4 radial cuts, FCC-32 L=1, measuring Z, band gap, K_6, K_8/K_6,
continuum edge, and sector ordering simultaneously.  Pre-registered kill
criterion (the 6->2 lesson): if only the sector gap sharpens with system
size, it is a host effect — discard without ceremony.

**What NOT to headline** (external review concurs with the campaign's own
findings): the exact +-1/4 flux value (tied to cubic-16 flippability
counting), the 6->2 crossing (host pathology, already demoted), and the
BIC protection story (correct, but easily dismissed as finite-volume
sector structure; keep it as a diagnostic, not a claim).

**Remote execution notes.**  Every scanner is a standalone script taking
`(coupling, ...)` pairs on the command line and writing one json+npz per
point (embarrassingly parallel by grid point; resume = rerun missing
points).  Dependencies: numpy + scipy only; thread count via
`OMP_NUM_THREADS`/`OPENBLAS_NUM_THREADS` (8 per job is calibrated; do NOT
run concurrent unbounded-BLAS jobs — measured 30x slowdown).  Measured
per-point costs on a 32-core/125 GB node: cubic-16 plane point ~15-50
min; cubic-16 anatomy/DSSF point ~1-3 min; FCC-32 L=1 multi-anchor
coupling ~70 min; FCC-32 L=2 bordered sector ~155 s/sector (85 sectors);
L=2 spaces ~2.5 M states (XXZ), several-fold larger with pair moves —
budget WP2 accordingly.  SLURM infra started in `campaign/pilot/`.

The sections below are the frozen protocol definition and the original
validation ladder; they remain the reference for what "winding-free"
means microscopically.

## Scientific question

Periodic QSI clusters contain ice-preserving paths that close only through a
periodic image.  On cubic-16 the shortest such process flips four spins and
appears at order `Jpm^2/Jzz`, before the physical contractible six-spin ring
exchange at order `|Jpm|^3/Jzz^2`.  The task is to remove the former without
asserting that a low-order effective Hamiltonian is the microscopic model.

## Definition

For tetrahedron `t`, define `Q_t = sum_{i in t} S_i^z`.  The fixed reference
operator is the ice-rule projector

```text
P_ice = product_t 1_{0}(Q_t) = 1_{E_ice}(H0).
```

Its rank is cluster dependent, but the protocol is defined by `P_ice`, not by
enumerating or naming that number of states.  `Ran(P_ice)` means the range of
the projector, `{|psi>: P_ice |psi> = |psi>}`: the complete subspace satisfying
the ice rule on every tetrahedron.

For oriented bond `(i,j)`, let
`d_ij = r_j - r_i - n_ij^T L` and
`c_ij = r_i + r_j - n_ij^T L`.  The source is an explicit phase in every
microscopic exchange and pair-flip matrix element:

```text
H(theta) = H0 - Jpm sum_<ij> [
    exp(+2 i theta.d_ij) S_i^+ S_j^-
  + exp(-2 i theta.d_ij) S_i^- S_j^+
] + Jpmpm sum_<ij> [
    exp(-2 i theta.c_ij) S_i^+ S_j^+
  + exp(+2 i theta.c_ij) S_i^- S_j^-
].
```

This is the full finite-cluster microscopic Hamiltonian.  No effective
Hamiltonian, energy denominator, or coupling expansion has been introduced.

For a microscopic move, define its lifted polarization increment as `-d_ij`,
`+d_ij`, `+c_ij`, or `-c_ij` for `S_i^+S_j^-`, `S_i^-S_j^+`,
`S_i^+S_j^+`, or `S_i^-S_j^-`, respectively.  The phases multiply to
`exp(-i theta.q_gamma)`, where
`q_gamma = 2 sum_l Delta p_l`.  A completed contractible path has
`q_gamma = 0`; a path closed only through a periodic image has nonzero
integer transport.

`P_theta` is not the ice projector.  It is the exact spectral projector of
`H(theta)` onto the isolated band continuously connected to `Ran(P_ice)`:

```text
P_theta = (1 / 2 pi i) contour_integral (z - H(theta))^-1 dz,
rank(P_theta) = rank(P_ice).
```

The contour encloses that band and no other eigenvalues.  If the band gap
closes, the protocol fails.  If
`G_theta = P_ice P_theta P_ice` is nonsingular on `Ran(P_ice)`, the polar map

```text
W_theta = P_theta P_ice G_theta^(-1/2)
h(theta) = W_theta^dagger H(theta) W_theta
```

puts every exact band operator in the same ice-rule reference space.  The
winding-free operator is the continuous zero-transport Fourier coefficient

```text
h_0 = (2 pi)^-3 integral_[0,2pi)^3 h(theta) d^3 theta.
```

The implemented character rule

```text
h_M = M^-3 sum_{m in Z_M^3} h(2 pi m / M)
```

keeps `q = 0 mod M`; finite `M` aliases `q = M k`.  The residual is measured
by the `M -> 2M` ladder.

## Why this is nonperturbative

At each character point, the production calculation diagonalizes the
microscopic `H(theta)` at the physical coupling.  It does not expand the
eigenvalues, `P_theta`, `G_theta^(-1/2)`, or `h(theta)` in either transverse
coupling.
Formally, an analytic band could be written as

```text
h(theta; lambda) = sum_n lambda^n sum_q h[n,q] exp(-i theta.q).
```

The continuous character integral would give `sum_n lambda^n h[n,0]`.
Low-order perturbation theory computes a few terms in that series; production
instead evaluates `h(theta; lambda=1)` directly and then takes its zero
Fourier component.  Perturbative rows are retained only to verify that the
source removes the winding four-loop and retains the contractible hexagon.

The nonperturbative claim is finite-cluster and selected-band specific.  It
still requires an isolated band, nonsingular polar overlap, eigensolver
convergence, character-grid convergence, and separate cluster-size tests.

For final observables, embed the winding-free band difference into the full
microscopic Hilbert space:

```text
bar_h_M = trace_matched(h_M, h(0))
H_wf = H(0) + W_0 [bar_h_M - h(0)] W_0^dagger
Z_wf = Tr_full exp(-beta H_wf).
```

Equivalently, partition-function moments of the bare band are subtracted from
the microscopic trace and winding-free band moments are added.  This is an
evaluation identity for one temperature-independent full-Hilbert Hamiltonian,
which equals `H(0)` on the exact band complement.  The embedded correction is
band-supported and generally nonlocal.

## Frozen validation

The mechanism and cubic exact implementation are now fixed rather than open
campaign stages:

| Quantity | Frozen evidence |
|---|---|
| topology | all audited winding four-loops and wrapping hexagons are removed; all contractible hexagons remain on cubic-16 and FCC-32 |
| nonperturbative construction | each cubic source point uses the microscopic Hamiltonian and exact polar pullback |
| pair-flip implementation | cubic-16 finite `Jpmpm` uses the full 65,536-state basis; the M=2 gauge identity holds to `2.5e-16` |
| character resolution | cubic `M=3 -> 4` centered operator change is 0.013% |
| band identity | minimum cubic ice overlap is 0.762 and Ritz residuals are below `4e-12` |
| thermal complement | cubic high-temperature microscopic crossover is retained by the full-Hilbert embedding |
| external check | QMC heat comparison improves strongly, while the remaining heat/entropy mismatch is explicitly identified as a size limitation |

The order-two/order-three rows remain topology audits only.  They are never
eligible as FCC-32 thermodynamics, dynamics, or material-fit curves.

## Remaining deliverable A: FCC-32 Figure 1

Figure 1 must be regenerated entirely on the `2 x 2 x 2` FCC cluster.  This is
not a mechanical cluster-name change.

1. **XYZ source.** The cubic-16 implementation is complete.  Scale the same
   dominant-`a` convention to FCC-32,

   ```text
   Jpm   = -(Jb + Jc) / 4
   Jpmpm =  (Jb - Jc) / 4.
   ```

   with both exchange channels sourced by the microscopic polarization change.
   The FCC-32 path phase and zero-character selector must still be tested
   independently for `S+S-` and `S+S+` sequences.

2. **Scalable band extraction.** FCC-32 has `rank(P_ice)=2970`, but the full
   `Sa=0` XXZ sector has dimension 601,080,390 and `Jpmpm` removes even that
   conservation law.  The dense cubic backend is invalid.  Production needs a
   distributed matrix-free eigensolver or another exact sparse backend with
   restartable source points, residuals, band gaps, and polar-overlap spectra.

3. **Character convergence.** Evaluate matched `M=2,3,4` grids.  Require less
   than 5% centered operator change and less than 2% change in the plotted heat
   capacity between the final two grids.  `M=2` alone only proves removal of
   the shortest audited paths.

4. **Microscopic complement.** Estimate the untouched thermal complement with
   a stochastic trace method and require less than 1% relative trace error in
   the plotted range.  Replacing an ice-space spectrum without this complement
   is not an all-temperature result.

5. **Figure products.** Save FCC-32 loop geometry, periodic and winding-free
   `C` and `S`, and `Szz` at all eight translation momenta.  Periodic and
   winding-free dynamics must use one dressed probe and pass zeroth-moment sum
   rules.  The NPZ metadata contract is recorded in `campaign/README.md` and
   numerical gates in `campaign/fcc32_xyz_manifest.json`.

The existing cubic QMC discrepancy is not deleted.  The FCC-32 result must show
that the drift is reduced before the cluster is presented as a bulk surrogate.

## Remaining deliverable B: Ce2Hf2O7 parameter fit

The fit uses the nearest-neighbor ABC Hamiltonian, not the XXZ slice.  For the
published NLC-A parameters `(0.050, 0.021, 0.004)` meV,
`Jpm=-0.00625` meV and `Jpmpm=0.00425` meV, so omitting `Jpmpm` is not a
controlled material approximation.

1. **Data.** The active v5 raster extraction and provenance are in
   `campaign/data/`.  It supports visual comparison and exploratory ranking.
   Submission-quality confidence intervals require author-provided tabulated
   `Cmag` values and uncertainties.

2. **Seeds and domain.** Start adaptive sampling around published NLC minima A
   and B and the ordered-regime QMC minimum.  Enforce
   `abs(Ja)>=abs(Jb),abs(Jc)` and `Jb>=Jc`; do not silently choose a permutation
   of `(Jx~,Jy~,Jz~)` when comparing neutron observables.

3. **Objective.** With raw uncertainties, reproduce the published weighted
   high-temperature fit over 1.5--3 K and the shape fit over 0.1--1.5 K, then
   add the FCC-32 winding-free low-temperature range down to 0.025 K.  Until
   uncertainties are available, report only the unweighted RMSE ranking from
   `run_ce2hf2o7_fit.py`, never a chi-square or confidence interval.

4. **Refinement.** Rank the converged grid, refine every local minimum, and
   profile all three exchanges.  Report temperature-window sensitivity,
   character/complement numerical errors, and parameter correlations.

5. **Holdout tests.** Entropy and neutron scattering are validation data, not
   additional free normalization parameters.  A heat-capacity minimum becomes
   a material candidate only if these holdout observables are consistent.

The old `A*` moment-matched point is a useful seed, not a fit: it used a
third-order FCC ring scale and no experimental likelihood.

## Definition of done

The campaign is complete when (i) Figure 1 contains only converged FCC-32
products satisfying the manifest, and (ii) the fit report contains at least
one refined ABC minimum with numerical error bars, experimental uncertainty
provenance, and holdout checks.  Neither requirement can be met by relabeling
the archived 2,970-state order-three calculation.

## Reproducibility

`campaign/run_nonperturbative.py --max-grid 4` writes the exact pulled-back
operators, thermodynamics, and gate report to `campaign/outputs/`; individual
source points are restartable under `campaign/cache/nonperturbative_points/`.
Passing `--jpmpm VALUE` selects the complete cubic-16 spin basis and emits
selected-band products; all-temperature products additionally require a
matching microscopic complement.
`campaign/run_validation.py` retains the order-three topology diagnostics.
`campaign/run_dssf.py` computes the frozen cubic-16 longitudinal validation.
`campaign/make_figures.py` reads saved products.  The FCC-32 contract is in
`campaign/fcc32_xyz_manifest.json`; `campaign/run_ce2hf2o7_fit.py` ranks only
converged files satisfying that contract.  `legacy/` contains rejected
approaches and preliminary claims; active code never imports it.
