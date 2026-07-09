# Simulation plan: clean ice-band ED for 2x2x2 FCC and pi-flux Ce pyrochlores

## Goal

Use the transported-dipole character formalism developed in
`notes/finite_size_loop_projection_notes.pdf` as the working ED protocol for
the quantum spin ice gauge sector.  The production workhorse is the full
microscopic QED character average:

1. build the microscopic Hamiltonian `H_mic^{2delta}(theta)` at the eight
   `M = 2` character points;
2. extract the lowest ice-like band from QED for each character point;
3. pull each band back to the ice basis as an operator;
4. average those operators over the character grid;
5. compute `C(T)`, `S(T)`, and low-energy observables from the averaged clean
   band.

The validated minimal character protocol is

```text
transport charge: Q = 2 delta
character grid:  M = 2, theta_mu in {0, pi}
projection:      full-QED operator character average, equivalent to Q=0 at
                 the perturbative row level
```

The 16-site full-microscopic check showed that this removes the spurious
winding four-loop peak while leaving the higher-temperature microscopic
spin-ice peak intact.  The campaign now moves to the 2x2x2 FCC pyrochlore
cluster, then benchmarks the zero-flux case against sign-free QMC, and finally
uses the same clean ice-band protocol to predict the pi-flux thermodynamics for
Ce2Hf2O7 and Ce2Zr2O7.

## What has been validated locally

- Cubic 16-site full ED:
  - exact full spectrum: 65,536 states;
  - low-band QED extraction agrees with the exact lowest 90 eigenvalues to
    `6.22e-14`;
  - full ED low peak: `T/Jzz = 0.0129475`;
  - clean ice-band replacement low peak: `T/Jzz = 0.0012227`;
  - full ED high peak: `T/Jzz = 0.252820`;
  - clean ice-band replacement high peak: `T/Jzz = 0.252931`.
- The high peak moves by only `4.37e-4` relatively, so the projection is acting
  where it should: on the finite-size ice-band loop artifact, not on the
  spinon/constraint-violation thermal scale.
- The old `4 delta` / `M = 3` language is obsolete.  All production notes and
  scripts should use `2 delta`, `M = 2`.
- The perturbative/SW row projector remains a diagnostic and cheap preflight.
  It is not the final workhorse once the corresponding full-QED low-band
  extraction is feasible.

## 2x2x2 FCC target

There is no conceptual change on the 2x2x2 FCC cluster:

- the ice manifold has dimension 2,970 rather than 90;
- the primitive vectors are FCC primitive vectors rather than cubic Cartesian
  unit vectors;
- `delta` must be represented in the FCC primitive coordinate frame used by the
  cluster builder;
- the same `Q = 2 delta`, `M = 2` character grid resolves the unwanted
  finite-size transport sectors;
- the artifact is still a winding-loop artifact, not a local contractible
  four-spin ring exchange.

There is one important practical change.  Full-Hilbert exact ED is not the
right tool at 32 sites: `2^32` is too large.  The 32-site campaign therefore
does not repeat the 16-site exact full-spectrum replacement literally.  Instead
it uses the same full-QED character-average band workhorse at the low-energy
level:

1. QED low-band extraction for the full microscopic Hamiltonian at the eight
   `2delta`, `M = 2` character points;
2. operator-level character averaging in the 2,970-dimensional FCC-32 ice
   basis;
3. perturbative/SW row tables as the row-level audit and fallback benchmark;
4. QMC benchmarks in the zero-flux sign-free regime;
5. QED/FTLM for high-energy or non-ice-sector diagnostics when needed.

The 16-site all-temperature spectral-replacement test is a validation of the
method, not the production algorithm for 32 sites.

## Immediate tasks

| ID | Task | Output | Gate |
|----|------|--------|------|
| T1 | Rebuild FCC-32 row tables with `Q = 2 delta` labels in FCC primitive coordinates. | `output/fcc32/rows_order23_or_234.npz` plus row census JSON. | Row census separates `Q=0` from winding rows exactly. |
| T2 | Port the full-QED `2delta`, `M = 2` low-band character-average workhorse from cubic-16 to FCC-32. | `H_qed_twist_avg` and diagnostics for all eight character points. | Low-band pullback is well conditioned; QED average agrees with row-level `Q=0` structure in the perturbative window. |
| T3 | Compute FCC-32 full-QED clean and untwisted-band `C(T)` for `Jpm/Jzz = +/-{0.03,0.04,0.05,0.06,0.08,0.10}` and the material window `Jpm/Jzz = -{0.14,0.16,0.18,0.20}`. | Peak table and `C(T)` curves. | Clean peak scales with `ghex = 12 |Jpm|^3/Jzz^2`; untwisted band shows four-loop drift. |
| T4 | Benchmark zero-flux clean FCC-32 against sign-free QMC. | QMC comparison table. | Agreement in `C(T)`, entropy release, and photon peak location within agreed finite-size/QMC error bars. |
| T5 | Add material parameter ingestion for Ce2Hf2O7 and Ce2Zr2O7. | `materials/*.json` with raw and rotated couplings. | `Jxz' = 0` after rotation; U(1) projection has `Jpmpm = 0`. |
| T6 | Run pi-flux clean FCC-32 full-QED workhorse at the U(1)-projected material points. | Material prediction table and plots. | Stable against nearby `Jpm/Jzz` sweep and, if available, order-4 SW row diagnostics. |

## QMC benchmark protocol

QMC can benchmark the zero-flux side, not the pi-flux side.

1. Use `Jpm > 0` in the same local-frame convention as the ED notes.
2. Run large-cluster sign-free QMC for the U(1) Hamiltonian
   `Jpmpm = Jzpm = 0`.
3. Compare against the clean FCC-32 ice-band ED, not the bare periodic ED.
   The preferred ED object is the full-QED character-averaged band; the SW
   row projector is the diagnostic comparison.
4. Compare dimensionless quantities first:
   - `T_peak / ghex`;
   - `C(T)/N`;
   - entropy released through the gauge peak;
   - low-T integrated weight.
5. Treat a mismatch as a method failure until checked against:
   - temperature normalization;
   - sign convention for `Jpm`;
   - whether QMC includes non-ice sectors at the compared temperature;
   - whether order-4 SW corrections are needed at the benchmark coupling.

Passing this benchmark is the main credibility gate before presenting pi-flux
predictions.

## Material parameter pipeline

For each material start from the fitted local exchange matrix in the
dipolar-octupolar frame.  In the common simplified notation this is the
`x-z` mixed form

```text
H_ij = Jx S_i^x S_j^x + Jy S_i^y S_j^y + Jz S_i^z S_j^z
     + Jxz (S_i^x S_j^z + S_i^z S_j^x).
```

First rotate the local `x-z` axes to remove `Jxz`.  Let

```text
M_xz = [[Jx,  Jxz],
        [Jxz, Jz ]].
```

Choose an orthogonal rotation `R(theta)` that diagonalizes `M_xz`:

```text
R(theta)^T M_xz R(theta) = diag(Jx_rot, Jz_rot),
tan(2 theta) = 2 Jxz / (Jx - Jz),
```

with the rotated labels chosen consistently with the final dominant-axis
assignment.  Equivalently use the eigenvalues

```text
Jx_rot, Jz_rot = (Jx + Jz)/2 +/- sqrt(((Jx - Jz)/2)^2 + Jxz^2),
```

Then map the rotated transverse couplings to the QSI convention used by the ED
code.  The Ising axis is the dominant exchange axis.  For the Ce2Zr2O7 and
default Ce2Hf2O7 parameter sets below, the dominant axis is `y`, so

```text
Jzz   = Jy,
Jpm   = -(Jx_rot + Jz_rot) / 4,
Jpmpm =  (Jx_rot - Jz_rot) / 4,
```

For the first production campaign we enforce the U(1) model by killing the
non-U(1) term after the rotation:

```text
Jpmpm -> 0,
Jxx_U1 = Jyy_U1 = -2 Jpm,
Jzz_U1 = Jzz.
```

For a dominant rotated `x` or `z` axis, use that dominant eigenvalue as `Jzz`
and use the remaining two eigenvalues as the transverse pair in the same
formula.  This preserves the fitted `Jzz` and the fitted transverse hopping scale `Jpm`,
while removing the pair-creation term that breaks total `S^z` conservation.
The pi-flux calculation is then the clean FCC-32 ice-band ED at the material's
`Jpm/Jzz` ratio.  If `Jpm < 0`, the material point lies in the pi-flux sector
of this convention.

## Material parameter table

The current working material inputs are below.  All raw exchange constants are
in meV.  `lambda_-` and `lambda_+` are the two eigenvalues of the `x-z` block
after rotating away `Jxz`, with `lambda_+ >= lambda_-`.  For dominant-`y`
sets, `Jzz = Jy`, `Jpm = -(lambda_+ + lambda_-)/4`, and
`Jpmpm = (lambda_+ - lambda_-)/4` before the U(1) projection.

| material / set | source | raw `(Jx,Jy,Jz,Jxz)` meV | `theta` deg | `(lambda_-,lambda_+)` meV | QSI `(Jzz,Jpm,Jpmpm)` meV | U(1) run `(Jzz,Jpm,Jpmpm)` meV | `Jpm/Jzz` |
|---|---|---:|---:|---:|---:|---:|---:|
| Ce2Zr2O7, set 2 | Gaudet/Changlani et al. optimized set used for later calculations | `(0.0385, 0.0880, 0.0200, 0.0000)` | `0.000` | `(0.020000, 0.038500)` | `(0.088000, -0.014625, +0.004625)` | `(0.088000, -0.014625, 0)` | `-0.166193` |
| Ce2Hf2O7, set a | Poree et al. representative dominant-`Jy` fit | `(0.0110, 0.0440, 0.0160, -0.0020)` | `-70.670` | `(0.010298, 0.016702)` | `(0.044000, -0.006750, +0.001601)` | `(0.044000, -0.006750, 0)` | `-0.153409` |
| Ce2Hf2O7, set b, default | Poree et al. field-thermo parameter set | `(0.0200, 0.0470, 0.0130, -0.0080)` | `-33.185` | `(0.007768, 0.025232)` | `(0.047000, -0.008250, +0.004366)` | `(0.047000, -0.008250, 0)` | `-0.175532` |
| Ce2Hf2O7, set c | Poree et al. representative dominant-`Jx` fit | `(0.0460, 0.0220, 0.0110, -0.0010)` | `-1.635` | `(0.010971, 0.046029)` | `(0.046029, -0.008243, +0.002757)` | `(0.046029, -0.008243, 0)` | `-0.179082` |

Default production choices:

- Ce2Zr2O7: use set 2.
- Ce2Hf2O7: use set b as the default, and run sets a/c as sensitivity checks.
- All material points are pi-flux in this convention because `Jpm < 0`.

Parameter sources:

- Ce2Zr2O7 set 2 is the optimized set quoted as
  `Jx = 0.0385, Jy = 0.088, Jz = 0.020, Jxz = 0` meV in the later-calculation
  paragraph of Gaudet/Changlani et al., "Sleuthing out exotic quantum spin
  liquidity in the pyrochlore magnet Ce2Zr2O7".
- Ce2Hf2O7 sets a/b/c are the representative fits quoted in Poree et al.,
  "Dipolar-octupolar correlations and hierarchy of exchange interactions in
  Ce2Hf2O7"; the field-thermodynamics paper uses set b,
  `Jx = 0.020, Jy = 0.047, Jz = 0.013, Jxz = -0.008` meV.

The corresponding normalized U(1) ED runs are therefore:

```text
Ce2Zr2O7 set 2:       Jzz = 1, Jpm = -0.1661931818, Jpmpm = 0
Ce2Hf2O7 set a:       Jzz = 1, Jpm = -0.1534090909, Jpmpm = 0
Ce2Hf2O7 set b:       Jzz = 1, Jpm = -0.1755319149, Jpmpm = 0
Ce2Hf2O7 set c:       Jzz = 1, Jpm = -0.1790815330, Jpmpm = 0
```

Use these exact normalized points in addition to the coarse sweep
`Jpm/Jzz = -{0.14,0.16,0.18,0.20}`.

## Material JSON records

Each material should also have a JSON record with the raw fitted exchange
constants, their source, the rotation angle, the rotated couplings, and the
U(1)-projected couplings.

```json
{
  "material": "Ce2Hf2O7",
  "source": "fill with paper / table / fit identifier",
  "raw": {"Jx": null, "Jy": null, "Jz": null, "Jxz": null, "units": "K or meV"},
  "rotation": {"theta_rad": null, "Jx_rot": null, "Jz_rot": null},
  "qsi": {"Jzz": null, "Jpm": null, "Jpmpm_before_projection": null},
  "u1_projected": {"Jzz": null, "Jpm": null, "Jpmpm": 0.0}
}
```

Create one record for Ce2Zr2O7 set 2 and three records for Ce2Hf2O7 sets a/b/c.
The production table must report both:

- the raw fitted model, so the connection to experiment is transparent;
- the U(1)-projected model actually simulated in the first campaign.

## Production observables

For each benchmark/material point compute:

- clean low-temperature `C(T)/N`;
- entropy `S(T)/N` through the gauge scale;
- `T_peak`, `C_peak`, and `T_peak/ghex`;
- bare-vs-clean contrast for the same cluster;
- row-sector weights by `Q = 2 delta`;
- optional low-energy `Szz(q, omega)` in the clean ice band.

For material-facing plots, convert the final temperature axis back to Kelvin or
meV using the fitted `Jzz` units after the U(1) projection.

## Expected figures

1. FCC-32 method figure: bare vs clean `C(T)` for representative `Jpm`.
2. QMC benchmark figure: zero-flux QMC vs clean FCC-32 ED.
3. Pi-flux prediction figure: clean FCC-32 `C(T)` for Ce2Hf2O7 and Ce2Zr2O7
   U(1)-projected material points.
4. Parameter-flow figure/table: raw `(Jx,Jy,Jz,Jxz)` -> rotated
   `(Jx_rot,Jy,Jz_rot)` -> QSI `(Jzz,Jpm,Jpmpm)` -> U(1)
   `(Jzz,Jpm,Jpmpm=0)`.

## Current caution flags

- The FCC-32 calculation should not use the 16-site exact full-spectrum
  replacement as if it were available at 32 sites.
- The QMC comparison must be performed on the zero-flux side only.
- The pi-flux material predictions are meaningful only after the zero-flux QMC
  gate is passed.
- `Jpmpm -> 0` is a controlled first campaign, not a claim that the fitted
  material has exact U(1) symmetry.  Later runs should restore `Jpmpm` once the
  clean projection method is established for the U(1) sector.
