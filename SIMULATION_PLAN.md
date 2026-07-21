# Validation plan: a convergent winding-free band protocol

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
| character resolution | cubic `M=3 -> 4` centered operator change is 0.498% |
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
