# A non-perturbative, all-J± protocol for winding-loop-free QSI exact diagonalization

**Status (2026-07-17).** This note upgrades the QED low-band operator average
(`SIMULATION_PLAN.md`, `finite_size_loop_projection_notes.tex`) in three ways:

1. Two exact **structure theorems**, verified to machine precision against the
   saved cubic-16 data (`verify_mask_identity.py`): the entire eight-corner
   M=2 character average collapses to **one diagonalization plus a
   polarization-parity mask**, and the character group factorizes as
   (boundary twists) x (Resta polarization phases).
2. A **homotopy-theoretic optimality statement**: on T^3 the transported-dipole
   character projection is the *maximal* topologically defined cleaning; what
   it keeps (matched winding pairs included) is exactly what has a bulk
   counterpart and *cannot* be removed by any label.
3. A **non-perturbative production protocol valid at any J±** — graded
   Brillouin–Wigner (Feshbach) downfolding — that removes the ice-band
   pullback bottleneck (`eta_min -> 0` at J±/Jzz ≈ −0.18) entirely.  It never
   selects eigenstates, never forms a Gram matrix, and is algebraically exact
   at every coupling; its breakdown modes are physical (spinon collapse of the
   gauge sector), not numerical.

Throughout: cluster torus `T^3 = R^3 / Z^3 L`, transported dipole
`delta = rho + N L`, integer row charge `Delta = 2 delta`, character grid
`theta in {0, pi}^3` (M=2).

---

## 1. Structure theorems (verified)

### Theorem A — M=2 isospectrality

Let `X = sum_i r_i S_i^z` be the many-body polarization (home-cell
coordinates) and `V(theta) = exp(2i theta . X)`, a **diagonal** unitary in the
S^z product basis.  Conjugating the bare Hamiltonian by `V(theta)` multiplies
the hop `S_i^+ S_j^-` by `exp(2i theta . (r_i - r_j))`, while the dipole2
deformation carries `exp(2i theta . d_ij)` with
`d_ij = r_j - r_i - n_ij L`.  The ratio is `exp(-2i theta . n_ij L)`.
Since `L` is an integer matrix, at `theta in {0, pi}^3` this factor is
`exp(-2 pi i * integer) = 1`.  Hence

```
H_mic(theta) = V(theta) H_mic(0) V(theta)^dagger      exactly, for all 8 corners.
```

**Consequences.**
- All eight corner spectra are identical (verified: spread 3.6e-15).
- Any observable diagonal in the S^z basis — including P_ice, all Ising
  correlators, and every thermal trace `Tr f(H(theta))` — is exactly
  theta-independent.  The empirical facts that observable averaging reproduced
  the bare peak *to all digits* (0.0129476) and that `eta_min` was identical
  at every corner are now theorems, not numerical coincidences.
- The cleaning power of the protocol lives entirely in the **frame anchoring**:
  averaging is nontrivial only for objects expressed in a *fixed* reference
  frame (the ice basis), because V(theta) acts nontrivially there.

### Theorem B — the mask identity

P_ice is diagonal in the S^z basis, so it commutes with V(theta).  Therefore
the low-band ice-frame data transform as

```
X(theta) = D(theta) X(0),   S(theta) = S(0),   Q(theta) = D(theta) Q(0),
H_B(theta) = D(theta) H_B(0) D(theta)^dagger,
```

where `D(theta)_{alpha alpha} = exp(2i theta . x_alpha)` and
`x_alpha = sum_i r_i S_i^z(alpha)` is the polarization of ice state alpha.
The eight-corner average is then an exact entrywise projection:

```
(H_B^clean)_{alpha beta} = (H_B(0))_{alpha beta} * [ 2(x_alpha - x_beta) even componentwise ].
```

Verified on cubic-16 at J± = −0.03 and −0.05: zero mask mismatches on all
8100 entries, residual < 8e-15; 1284/8100 entries kept.

**Consequences.**
- **8x cost reduction** for every M=2 production run: FCC-32 needs *one*
  QED low-band extraction at theta = 0, then a parity mask with the ice-state
  polarizations expressed in the FCC primitive frame.  No character sweep.
- The projection becomes *exact* by construction (no grid, no aliasing
  question at the M=2 level).
- The M=2 protocol is revealed to be a **Z_2^3 grading of the band operator by
  many-body polarization parity** — the finite-cluster avatar of Resta's
  polarization operator `exp(2 pi i X / L)`.

### Theorem C — factorization of the character group

For general theta the same algebra gives

```
H_mic(theta) = V(theta) H_twist(phi = 2 L^T theta) V(theta)^dagger,
```

i.e. **every dipole2 character deformation is gauge-equivalent to an ordinary
boundary twist composed with a polarization rotation.**  The character average
over the full transported-dipole dual therefore factorizes:

```
(character average over Gamma^)  =  (ordinary twist average)  o  (polarization-parity mask).
```

- At M=2 the twist part is trivial (`2 L^T theta in 2 pi Z^3`) — pure mask.
- At M>=3 (e.g. the dipole4 M=3 grid) the twist part is genuine: spectra split
  (verified: spread 5e-2) and no mask reproduces the average.
- This *explains* the old failure table: ordinary boundary-twist averaging
  keeps exactly half of each winding channel because it implements only the
  twist factor and is blind to the polarization-parity factor.  The
  transported dipole `delta = rho + N L` is precisely the gauge-invariant
  gluing of the two.

---

## 2. The homotopy mathematics

### 2.1 Why `delta` is the complete topological label (optimality)

The torus is aspherical: `T^3 = K(Z^3, 1)`, so `pi_n(T^3) = 0` for n >= 2 and
free homotopy classes of loops are in bijection with conjugacy classes of
`pi_1(T^3) = Z^3 = H_1(T^3; Z)` (abelian, so classes = elements).  A completed
low-energy process is a closed loop in configuration space; lifting the spin
transport to the universal cover `R^3` assigns it the deck transformation
`w in Z^3` (its winding), and `delta` refines `w` by the home-cell dipole.
Two structural facts follow:

1. **Completeness.** There is *no finer homotopy invariant* on T^3 — no
   torsion, no higher obstruction.  Any cleaning scheme based on topological
   labels can remove at most the `w != 0` sectors.  The character projection
   removes exactly these.  It is therefore the *maximal* topological cleaning.
2. **Irremovability of matched pairs.** A process that winds by `+w` and later
   by `-w` is homotopically **trivial**: its lift closes in `R^3`, and it has a
   genuine infinite-lattice counterpart (a virtual spinon excursion of length
   ~ L).  No superselection label can distinguish it from bulk physics, and it
   *should not* be removed: it is an ordinary, honestly convergent finite-size
   correction (extra periodic images), controlled by comparing cluster sizes —
   not a parametric pathology like the odd-winding four-loop.

This resolves the "matched windings survive any operator average" observation
of the twist-averaging study: they survive because they must — they are in the
trivial class.  The protocol's target is precisely and only the nontrivial
classes, and Theorem-A/B implement that projection exactly.

### 2.2 The transport group and Pontryagin duality

Let `Gamma` be the abelian group generated by the transported dipoles of
completed processes (`Gamma subset (1/2) Z^3` for the rows enumerated so far).
It sits in an exact sequence

```
0  ->  Z^3 L  (windings, = pi_1(T^3))  ->  Gamma  ->  Gamma / Z^3 L  (polarization classes)  ->  0
```

The dual group `Gamma^ = Hom(Gamma, U(1))` is the **character torus**.
Ordinary boundary twists realize only the characters of the winding subgroup
(flat U(1) connections, `Hom(pi_1, U(1))`); the Resta polarization phases
realize the characters of the quotient.  Theorem C is the statement that the
dipole2 deformation realizes the full dual `Gamma^` and that the exact
sequence splits at the level of the deformations — twist x mask.  The old
"physical twists suppress but do not project" table is the statement that
averaging over a *subgroup* of characters projects onto the annihilator of
that subgroup, which is larger than the trivial class.

### 2.3 The process algebra and descent

Grade the low-energy effective algebra by transport:
`A = ⊕_{q in Gamma} A_q`, with `A_q A_q' ⊂ A_{q+q'}`.  The character torus
acts by `a_q -> chi(q) a_q`; the character average is the isotypic projection
onto the invariant subalgebra `A_0` (it is an algebra — products of neutral
processes are neutral, which is why cleaning must happen at operator level and
commutes with subsequent diagonalization).  **Descent statement:** every
process on the infinite lattice whose support fits in a fundamental domain
descends to a unique element of `A_0`, and conversely every element of `A_0`
lifts to the cover.  The cleaned cluster is the largest quotient of the bulk
low-energy dynamics that fits on the torus.  (At path lengths long enough to
see multiple images, the lift is no longer unique — that is the matched-pair
image correction of Sec. 2.1, an ordinary finite-size effect.)

### 2.4 Two non-perturbative topological diagnostics

- **Spectral flow / monodromy.** Thread a *continuous* physical twist
  `phi: 0 -> 2 pi` along direction mu and follow the low levels of
  `H_twist(phi)`.  States carrying winding content flow into different levels
  (nontrivial monodromy of the band bundle over the twist circle); neutral
  states return.  Since `pi_1(U(1)-flat connections) = Z^3` pairs with the
  winding lattice, the permutation + accumulated Berry phase is a
  *non-perturbative* measure of residual winding admixture in any candidate
  gauge sector, valid at any J±.  Use it as a validity gate where perturbative
  row tables no longer exist.
- **Polarization winding (Resta).** `z_mu = <exp(2 pi i X_mu / ell_mu)>` on
  candidate gauge states measures the polarization class of Sec. 2.2 directly
  and is computable from any Krylov vector for free.

---

## 3. Production protocol: graded Brillouin–Wigner downfolding (any J±)

### 3.1 The object

The eigenband + Loewdin pullback needs the microscopic low band to *be* an
ice band; this fails by J±/Jzz ≈ −0.18 (eta_min -> 0).  Replace it with the
exact Feshbach/Schur downfolding onto the ice block, which requires **no band,
no eigenvectors, no Gram matrix**:

```
F(z; theta) = P H(theta) P  +  P H(theta) Q [ z - Q H(theta) Q ]^{-1} Q H(theta) P,
```

with `P = P_ice`, `Q = 1 - P`.  This is defined for every J± and every z off
`spec(Q H Q)`.  Its geometric-series expansion is a sum over *completed*
virtual excursions `P -> Q -> ... -> Q -> P`, each carrying a definite
transported dipole — exactly the objects the character grading classifies, now
resummed to infinite order.  Eigenvalues with ice weight solve the exact
nonlinear problem `det[ F(E) - E ] = 0`; this is Brillouin–Wigner theory made
exact, not an expansion.

### 3.2 Equivariance and the clean operator

P and V(theta) are both diagonal in the S^z basis, so Theorems A–C transfer
verbatim to F:

```
F(z; theta) = D(theta) F_twist(z; 2 L^T theta) D(theta)^dagger.
```

**Stage A (M=2, the workhorse).**  One operator, no character sweep:

```
F_clean(z)_{alpha beta} = F(z; 0)_{alpha beta} * [ 2(x_alpha - x_beta) even componentwise ].
```

This projects every virtual process, to infinite order in J±, onto
`Delta = 2 delta = 0 (mod 2)` — it kills the four-loop and wrapping-hexagon
sectors identically, at any coupling.

**Stage B (M-refinement).**  Sectors with even nonzero transport (double
winding etc.) survive M=2.  Refine by adding genuine boundary twists
`phi = 2 L^T theta` for `theta` on the M=4 grid and averaging the
mask-conjugated results (Theorem C).  Convergence criterion: clean spectrum
and target observables stationary from M=2 -> 4 within error budget.  In the
perturbative window M=2 is provably sufficient (all resolved rows have
`Delta_mu in {0, ±1}`); at material couplings M-convergence is a reported
diagnostic.

**Stage C (solve).**  Options, in order of preference:

1. *Self-consistent scalar loop:* pick `z_0` at the target band center, solve
   `E_n^{(k+1)} = eig_n F_clean(z = E_n^{(k)})`; converges linearly with rate
   `||dF/dz|| < 1` (equal to the non-ice weight of the state — small whenever
   the state is gauge-sector-like).
2. *des Cloizeaux canonical form:* `H_dC = S^{-1/2} F' S^{-1/2}`-type
   energy-independent Hermitian effective Hamiltonian built from the
   `F(z)`-family; reduces exactly to the existing Loewdin construction when
   the band exists (use as the cross-check, not the definition).
3. *Linearization:* `H_lin = F_clean(z_0) + (E - z_0) dF_clean/dz|_{z_0}`
   gives a generalized eigenproblem; error `O((E - z_0)^2 ||d2F/dz2||)`.

Thermodynamics of the gauge sector then proceed exactly as now (C(T), S(T),
spectral replacement for the all-temperature check).

### 3.3 Numerics

- `F(z; 0)` costs `N_ice` shifted solves `[z - Q H Q]^{-1} (Q H P e_alpha)`.
  On FCC-32: 2970 right-hand sides in the fixed-Sz sector (dim 601,080,390 —
  use the existing QED MPI/GPU stack).  Shifted-CG/MINRES solves *all* z
  points from one Krylov space per RHS; block-Krylov across the 2970 RHS
  (they share the low-lying spectral content).  Stage A needs this **once**
  (theta = 0 only) — this is 8x cheaper than the already-planned eight-corner
  eigenvector campaign and needs no eigenvectors at all.
- Chebyshev/rational filtering variant: expand
  `[z - QHQ]^{-1} = sum_k c_k(z) T_k(QHQ)` to get the full z-dependence from
  one moment sequence per RHS.
- Hermiticity check `F(z*)^dagger = F(z)` and mask idempotence are free
  online diagnostics.

### 3.4 Validity diagnostics (replace eta_min)

| diagnostic | meaning | gate |
|---|---|---|
| `dist(z-window, spec(Q H Q))` (Lanczos on QHQ) | gauge-sector resonances spectrally clear of spinon continuum | > 0: controlled; -> 0: gauge sector dissolving (physics, report as such) |
| `\|\|dF_clean/dz\|\|` on the solved window | non-ice weight of gauge states; convergence rate of Stage C | << 1 controlled; O(1) gray zone |
| M-convergence (Stage B) | residual even-transport contamination | stationary M=2 -> 4 |
| spectral flow under continuous twist threading (Sec. 2.4) | non-perturbative winding content of the cleaned states | trivial monodromy on the gauge window |
| cluster comparison cubic-16 vs FCC-32 of `A_0` observables | matched-pair image corrections (irremovable, Sec. 2.1) | consistent trend |

Crucially, every failure mode is now *interpretable*: BW downfolding never
becomes ill-defined; when `z` collides with `spec(QHQ)` the statement is that
the cluster's gauge sector has merged with the spinon continuum at that
coupling — a physical crossover/phase boundary, which is itself a result.

### 3.5 Optional second leg: quasi-adiabatic band tracking

Where one *does* want an energy-independent band picture beyond the
perturbative window: define the dressed gauge band by parallel transport of
the spectral projector `P_band(J±)` along a path from small |J±| (Kato /
quasi-adiabatic continuation), instead of "lowest N_ice states".  The band is
then well-defined until a genuine gap closing on the path — again converting
method failure into physics (location of the QSI phase boundary on the
cluster).  Mask as in Theorem B.  Use as a cross-check of Stage A/C in the
gray zone.

### 3.6 Validation ladder

1. **Reduction check (controlled window).**  cubic-16, J± = −0.03, −0.05:
   masked-BW spectrum must agree with the existing eight-corner QED operator
   average (already exact vs full ED to 6e-14) within the Stage-C tolerance.
2. **Gray zone.**  J± = −0.08, −0.10: masked-BW vs Loewdin pullback; quantify
   drift as `||dF/dz||` grows.
3. **Beyond the pullback wall.**  J± = −0.18 (CHO-like), −0.30 (CZO-like) on
   cubic-16: masked-BW is defined; report diagnostics honestly — if
   `dist(z, spec QHQ) -> 0`, the *finding* is that cubic-16 has no isolated
   gauge sector there, and FCC-32 carries the burden.
4. **FCC-32 production.**  Stage A at theta=0 (single run), Stage B spot
   checks, zero-flux QMC gate unchanged, then pi-flux material points.

---

## 4. Immediate campaign actions

1. Adopt the **mask implementation** for all M=2 work (Theorem B): FCC-32
   eight-corner plan collapses to one run + mask.  Implement
   `x_alpha` in the FCC primitive frame; unit-test against
   `verify_mask_identity.py` logic on cubic-16.
2. Add the mask + isospectrality facts to the notes/paper: they turn two
   empirical negative controls (observable averaging fails; eta_min
   corner-independent) into theorems, and they are the cleanest way to present
   the method ("the character average *is* a polarization-parity
   superselection of the downfolded operator").
3. Prototype masked-BW on cubic-16 (dense: `QHQ` is 12780-dim in fixed Sz) —
   a day of work, gives the entire validation ladder rows 1–3 exactly.
4. Wire the spectral-flow diagnostic (continuous twist threading) into the
   existing `twist_resolved_*` tooling — it reuses the `physical` twist_kind
   path that already exists.
