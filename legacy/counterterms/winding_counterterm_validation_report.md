# Microscopic winding-counterterm validation

Updated: 2026-07-18

## Protocol under test

The current candidate is the full cubic-16 microscopic model,

\[
H_{\rm imp}=H_{\rm XXZ}+\kappa_4 \widetilde W_4
 +\kappa_{6w}\widetilde W_{6w},
\]

where `W4` and `W6w` are symmetry-complete sums over all wrapping four- and
six-site loops.  For each loop `C`, the locally projected operator is

\[
\widetilde W_C=\Pi_C W_C\Pi_C,
\qquad
\Pi_C=\prod_{t:\,t\cap C\ne\varnothing}P_t^{\rm ice}.
\]

Thus a counterterm flip acts only when every tetrahedron touching that loop is
two-in/two-out.  This operator is inserted in the complete microscopic Hilbert
space and diagonalized together with `H_XXZ`; it is not a downfolded or
perturbatively constructed effective Hamiltonian.  Within the global ice
manifold, `W_C` and `Wtilde_C` are exactly identical.  The local projector only
prevents the counterterm from directly altering nearby monopole sectors.

The original unprojected one-term protocol is retained below as a historical
baseline.  The locally projected two-term candidate is used for the current
all-temperature and dynamical gates.

## Topological operator audit

- Cubic-16 contains 36 wrapping four-loops and 16 contractible hexagons.
- The 90 ice states split into 25 exact three-component flux sectors.
- All 1,152 nonzero ice-manifold matrix elements of `W4` change flux sector.
- Every contractible-hexagon matrix element preserves flux sector.

This establishes that the counterterm targets only the topologically wrapping
four-loop operator at the microscopic level.

## Coefficient calibration

At `Jpm/Jzz = -0.05`, the refined nonperturbative scan gives

\[
\kappa_4/J_{zz}=0.0075,
\]

with a broad minimum over approximately `0.0070--0.0080`.  Separate transfer
temperatures select `0.0070` at `beta=20` and `0.0075` at `beta=50,100`.
The perturbative value `4 Jpm^2/Jzz = 0.0100` was recorded only as a blind
reference and was not used in the selection.

| Diagnostic | Periodic | Improved |
|---|---:|---:|
| flux transfer, `beta=20` | 0.27151 | 0.00454 |
| flux transfer, `beta=50` | 0.81970 | 0.02954 |
| flux transfer, `beta=100` | 0.90711 | 0.12118 |
| defect gap (`Jzz`) | 0.70340 | 0.76430 |
| low-band heat-capacity peak (`Jzz`) | 0.012948 | 0.002156 |
| relative infinite-T variance change | 0 | 8.27e-5 |

The residual transfer at `beta=100` is not zero.  A single four-body
counterterm removes about 87% of this low-temperature winding diagnostic, not
all higher-order wrapping processes.

## Coupling sweep and independent local reference

The coefficient calibration was repeated for `Jpm/Jzz = -0.03, -0.04, -0.05,
-0.06`.  The transfer temperatures were not set by a perturbative formula.
Instead, each coupling was referenced to an independently diagonalized
embedded hexagon.

The reference consists of the six tetrahedra sharing one central pyrochlore
hexagon.  Each of the six external spin pairs is frozen in an opposite-spin
configuration.  The six central spins then have exactly two ice configurations,
all central sites retain both tetrahedral constraints, and the active graph is
a six-cycle with no four-cycle or periodic identification.  Direct
diagonalization of its complete 64-state microscopic Hamiltonian gives the
local hexagon doublet splitting and heat-capacity scale without constructing an
effective Hamiltonian.

| `Jpm/Jzz` | best `kappa4/Jpm^2` | flux transfer: bare -> improved | `Tpeak/Thex_exact`: bare -> improved | relative infinite-T variance change |
|---:|---:|---:|---:|---:|
| -0.03 | 3.375 | 0.8490 -> 0.0664 | 20.74 -> 1.831 | 1.37e-5 |
| -0.04 | 3.375 | 0.7576 -> 0.0689 | 14.35 -> 1.715 | 4.32e-5 |
| -0.05 | 3.000 | 0.6747 -> 0.0549 | 10.67 -> 1.777 | 8.27e-5 |
| -0.06 | 2.625 | 0.5981 -> 0.0532 | 8.33 -> 1.831 | 1.30e-4 |

The optimum is a coupling-dependent nonperturbative calibration rather than a
fixed perturbative coefficient.  Over this interval the corrected cubic peak
tracks the exact local hexagon peak with a nearly constant factor `1.72--1.83`,
while the periodic peak departs parametrically at weak coupling.  The factor is
not unity, so the embedded hexagon validates recovery of the local scale and
coupling dependence, not exact finite-cluster amplitudes.

The tabulated coefficient ratios are grid-resolved minima.  The local ratio
spacing is `0.375` (with the earlier `Jpm=-0.05` scan providing finer
resolution), so the sweep establishes a calibration band and its coupling
drift rather than a precision interpolation formula.  Full all-temperature SLQ
was performed at `Jpm=-0.05`; at the other couplings the exact infinite-T
second moment supplies the high-temperature distortion gate.

## Zero-dipole benchmark and wrapping hexagons

Near weak coupling, the previous zero-dipole character-projected band provides
an independent target.  It is used only for validation: every counterterm
candidate is still diagonalized in the full 12,870-dimensional microscopic
`Sz=0` block.  The comparison minimizes the centered RMS difference between
the complete 90-state spectra, normalized by the target spectral standard
deviation; the heat-capacity peak and flux transfer are out-of-objective checks.

The one-term counterterm does not reproduce this benchmark.  Its peak is 63%
high at `Jpm=-0.03` and 76% high at `Jpm=-0.05`.  Adding the symmetry-complete
sum over all 48 wrapping hexagons,

\[
H_{\rm imp}^{(2)}=H+\kappa_4W_4+\kappa_{6w}W_{6w},
\]

gives a transferable weak-coupling coefficient
`kappa6w/|Jpm|^3 = -6` when the previously calibrated `kappa4(Jpm)` is retained.

| `Jpm/Jzz` | zero-dipole transfer | `W4` transfer | `W4+W6w` transfer | `Tpeak/Tdelta0`: `W4` -> two-term | spectral error: `W4` -> two-term |
|---:|---:|---:|---:|---:|---:|
| -0.03 | 1.89e-4 | 0.0664 | 0.0343 | 1.632 -> 1.260 | 1.843 -> 1.221 |
| -0.05 | 5.21e-4 | 0.0549 | 0.0266 | 1.763 -> 1.315 | 1.725 -> 1.068 |

The wrapping-hexagon term therefore removes about half of the remaining
transfer and improves the centered spectrum at both couplings.  The defect gaps
remain above `0.77 Jzz`, and the infinite-temperature variance changes remain
below `8.3e-5`.

This is a partial correction, not closure.  The transfer remains roughly
50--180 times larger than in the zero-dipole band, and the corrected peak is
still 21--32% high.  Independent two-parameter fits can move an individual peak
closer, but produce incompatible spectral shapes (including a split low-T
feature at `Jpm=-0.03`).  The next justified operator is an
environment-dressed wrapping four-loop, not another coefficient fitted only to
`Tpeak`.

## Local ice projection

The same scan was repeated with `W4` and `W6w` conditioned on the ice rule for
all tetrahedra touching each loop.  The calibrated `kappa4(Jpm)` was retained,
and a refined shared-coefficient scan selected

\[
\kappa_{6w}/|J_{\pm}|^3=-7.5.
\]

| `Jpm/Jzz` | spectral error: unprojected -> local | transfer: unprojected -> local | `Tpeak/Tdelta0` | local defect gap | relative infinite-T variance change |
|---:|---:|---:|---:|---:|---:|
| -0.03 | 1.221 -> 0.984 | 0.0343 -> 0.0248 | 1.133 | 0.838 | 8.59e-7 |
| -0.05 | 1.068 -> 0.909 | 0.0266 -> 0.0186 | 1.094 | 0.732 | 5.18e-6 |

The mean centered spectral error decreases from `1.145` to `0.947`.  This is a
real improvement because the projected and unprojected operators are exactly
equal on all 90 ice states: it arises solely from treating defect sectors more
carefully.  Nevertheless, the transfer remains 36--132 times above the
zero-dipole target.  Scalar retuning of `kappa4` does not remove this floor; the
spectral and transfer minima are sharp near the independently calibrated
values.

As a diagnostic, replacing the local projector by the global
`Pice W Pice` projector worsens the shared spectral error to `1.038` and the
mean transfer to `0.0265`.  The global projector is therefore rejected both on
locality grounds and on the numerical benchmark.

## Environment-dressed four-loop test

The smallest environment-resolved four-loop operator was also constructed,

\[
D_4=\frac12\sum_C\{N_C,W_C\},
\]

where `N_C` counts flippable wrapping four-loops sharing at least two sites with
`C`.  Regression of `H_phi0-H_delta0` shows that `W4+D4` reproduces the complete
Hamming-distance-four block to relative residual below `5e-13`.  The removed
operator norm is 98--99% in this block; Hamming distances six and higher carry
only 0.7--1.8% of the norm.

With the local ice projectors and shared scaling `lambda4/|Jpm|^4=+6`, `D4`
reduces the mean spectral error only from `0.947` to `0.928` and leaves the mean
transfer unchanged (`0.0217` -> `0.0219`).  The individually resolved errors
are `0.983` at `Jpm=-0.03` and `0.873` at `Jpm=-0.05`.  This modest,
non-transferable gain does not justify adding a third production coefficient.

## All-temperature gate

The lowest 90 states were diagonalized exactly.  Their orthogonal complement
in the full 65,536-state Hilbert space was traced with 20-probe, 180-step
deflated stochastic Lanczos quadrature.  Running the same estimator on the bare
model and comparing with its exact full spectrum gives a worst heat-capacity
error of `1.21e-3` per spin and RMS error `3.33e-4` per spin.

| Diagnostic | Periodic | Improved |
|---|---:|---:|
| low-T peak position (`Jzz`) | 0.012948 | 0.001338 |
| high-T peak position (`Jzz`) | 0.252820 | 0.252311 |
| high-T peak height | 4.52229 | 4.49317 |

The table now uses the locally projected `W4+W6w` candidate at
`Jpm/Jzz=-0.05`, `kappa4=0.0075`, and `kappa6w=-0.0009375`.  The
high-temperature peak moves by `-0.201%`; its height changes by `-0.644%`.
The improved peak's jackknife uncertainty is `2.85e-4` per spin.  The exact
infinite-temperature variance change is `5.18e-6`.

## Dynamical gate

The zero-temperature longitudinal response was evaluated at the allowed
extended-zone pinch-point star `(200)/(020)/(002)`.  The primitive `X`
representatives were explicitly checked and have an exact ice-manifold
extinction, so they must not be used to assess the photon response on this
cluster.

| Low-band spectral diagnostic | Periodic | Improved |
|---|---:|---:|
| integrated low-band `Szz` weight | 0.43472 | 0.43083 |
| total inelastic static `Szz` weight | 0.43792 | 0.43952 |
| `Szz` low-band captured fraction | 0.9927 | 0.9802 |
| first weighted `Szz` excitation (`Jzz`) | 0.04193 | 0.00238 |
| `Szz` centroid (`Jzz`) | 0.04421 | 0.00531 |
| integrated low-band contractible-hexagon weight | 0.23581 | 0.10749 |
| total inelastic static hexagon weight | 0.24937 | 0.12573 |
| contractible-hexagon low-band captured fraction | 0.9456 | 0.8550 |
| contractible-hexagon peak (`Jzz`) | 0.12468 | 0.01461 |

These values use the locally projected two-term candidate.  The total
longitudinal weight is preserved to `0.37%` while its energy is strongly
reduced, and `98.0%` of that weight is captured in the computed low band.  By
contrast, the total static contractible-hexagon weight falls by `49.6%`; this
is not merely spectral weight shifted above the 180-state window.  This is the
principal unresolved control: it may be removal of four-loop contamination
from the hexagon channel, or it may indicate overcorrection.  A larger-cluster
benchmark is required to distinguish these possibilities.

## Rejected diagonal counterterm

Adding `mu4 (F4 - <F4>_infinity)` can lower the residual transfer, but it fails
the high-temperature and spectral-scale controls.  At `mu4=-0.004`, the low
peak collapses to `2.70e-5 Jzz` and the infinite-temperature variance changes
by 1.19%.  At `mu4=-0.002`, the variance still changes by 0.60%.  Therefore the
validated candidate keeps `mu4=0`.

## Current verdict

The locally ice-projected microscopic counterterm program passes the topology,
defect-gap, all-temperature thermodynamic, and longitudinal `Szz` gates on
cubic-16 and improves the complete weak-coupling spectrum relative to the
unprojected protocol.  It does not yet close the method: residual winding
transfer remains one to two orders of magnitude above the zero-dipole
benchmark, and the static contractible-hexagon weight is reduced by half.  No
thermodynamic-limit, FCC-32, or Ce2Hf2O7 fit claim is justified yet.

Required next gates:

1. Identify an additional diagnostic or operator that reduces the transfer
   floor at both weak couplings; `D4` and the global ice projector do not pass
   this gate.
2. Resolve the contractible-hexagon spectral-weight change using an active
   larger-cluster or controlled-boundary calculation; the frozen-boundary
   reference validates the scale but not that spectral sum rule.
3. Establish the FCC-32 loop census, symmetry orbit, and allowed momentum set
   before fitting Ce2Hf2O7.
