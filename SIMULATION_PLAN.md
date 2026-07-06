# Simulation plan — unbiased 32-site ED of quantum spin ice: thermodynamics and dynamical structure factor for the π-flux cerium pyrochlores

## 0. Goal (one paragraph)

Compute, on the 32-site (2×2×2 FCC) pyrochlore cluster, the two quantities
that confront quantum spin ice (QSI) with experiment — the specific heat
`C(T)`/entropy `S(T)` and the dynamical spin structure factor `S(q,ω)` —
**free of the finite-size ring-exchange artifact**, benchmark them against
sign-problem-free quantum Monte Carlo (QMC) in the 0-flux sector, and then
deliver them in the **π-flux sector QMC cannot reach** at the exchange
parameters of the dipolar–octupolar cerium pyrochlores
(Ce₂Zr₂O₇, Ce₂Sn₂O₇, Ce₂Hf₂O₇). The emergent photon shows up as the
**lower peak** of `C(T)` and as low-ω weight in `S^{zz}(q,ω)`; both are the
targets.

---

## 1. The method question, resolved

**Motivation for twist averaging (the original idea):** the periodic
cluster carries spurious ice-preserving four-site ring exchanges,
`g₄ = 4J±²/Jzz`, which are *second order* in `J±` and therefore
parametrically larger than the genuine hexagon photon coupling
`g_hex = 12|J±|³/Jzz²`. They live entirely in the ice manifold, so they
contaminate exactly the low-energy (gauge/photon) sector: the lower `C(T)`
peak and the low-ω `S^{zz}` weight. The hope was that averaging over
U(1) boundary twists would delete them.

**Does twist averaging remove the 4-ring? Verified: NO — for both `C(T)`
and `S(q,ω)`.** The theorem (`paper/twist_theory_proof.tex`) is that a ring
exchange is a sum over *virtual paths* (perfect matchings), one of which
uses no wrapping bond and so survives any twist; and that averaging a
*nonlinear observable* keeps all even powers of the spurious coupling.
Both failure modes confirmed numerically (16-site, exact):

| low-energy `S^{zz}(ω)` weight centroid, J±=−0.05 | value | in units of g₄ |
|---|---|---|
| bare PBC | 0.0416 | **4.2 g₄** (spurious) |
| twist-averaged (8 corners) | 0.0261 | **2.6 g₄** (still spurious) |
| zero-transport projected | 0.0047 | **0.47 g₄ ≈ 3 g_hex** (photon) |

Twist averaging barely moves the spurious weight; only the projector brings
it to the photon scale. Same conclusion for the `C(T)` low-T peak
(0.037 → 0.031 → 0.0075 Jzz). *(32-site FCC confirms: S^{zz} centroid
1.8 g₄ → 0.28 g₄ at −0.05; 4.7 g₄ → 0.65 g₄ at +0.046.)*

**The robust method separates two things twist averaging conflates.**
The DSSF has two channels with different physics and different cures:

1. **Photon / gauge sector — `S^{zz}(q,ω)` at low ω, and the lower `C(T)`
   peak.** Lives in the ice manifold; *this* is what the 4-ring
   contaminates. **Cure: the zero-transport (δ=0) projector** on the exact
   ice-manifold effective Hamiltonian. Because the ice manifold of the
   32-site FCC cluster is only 2970-dimensional, its **full spectrum** is
   obtained by dense diagonalization → numerically exact, contamination-free
   gauge thermodynamics *and* the photon spectral function
   `S^{zz}(q,ω) = Σ_n |⟨n|S^z_q|0⟩|² δ(ω−ω_n)` from the projected
   eigenstates. No stochastic low-T floor.

2. **Spinon continuum — `S^{±∓}(q,ω)` at ω ≳ 2Δ_s ~ Jzz, and the high-T
   `C(T)` Schottky.** Lives *outside* the ice manifold (pair creation), is
   local physics, and is **not** 4-ring-contaminated. **Method: standard
   full-Hilbert-space FTLM/Lanczos dynamics** on the 2³² cluster (existing
   QED pipeline: `run_dssf.py`, distributed MPI), with a **twist grid used
   for its legitimate purpose — Brillouin-zone interpolation of the genuine
   continuum** (Lin–Zong–Ceperley), *not* for contamination removal.

So twist averaging is retained where it works (BZ sampling of the genuine
spinon bands) and replaced by the projector where it fails (the ice-manifold
photon sector). This is the sharp logical spine of the paper.

---

## 2. What is computed, and by which engine

| Observable | Experimental counterpart | Engine | 4-ring? |
|---|---|---|---|
| `C(T)`, `S(T)` low-T (gauge) peak | photon Schottky, entropy release | projected Heff, full spectrum (2970) | removed by projector |
| `C(T)` high-T (charge) peak | spinon Schottky at T~Jzz | full FTLM (2³²) or bare high-E | uncontaminated |
| `S^{zz}(q,ω)` low ω | emergent photon / restored pinch-point dynamics | projected Heff dynamics (2970) | removed by projector |
| `S^{±∓}(q,ω)` | broad spinon continuum | full FTLM + twist grid (2³²) | uncontaminated |

Feasibility (measured): the projected-Heff engine builds **once** at J±=1
(row tables ∝ J±^k exactly; 218 s, 1.6 MB) and every coupling/both signs is
a ~seconds dense diagonalization. The full-FTLM engine is the existing
distributed pipeline.

---

## 3. Phase A — 0-flux benchmark (credibility firewall)

QMC is sign-problem-free only for J± > 0. Benchmark there.

- **A1** Build 32-site row tables (order 4; + order 3 for convergence). One
  job, ~15 min.
- **A2** `gauge_thermo.py sweep` at J± = +0.03…+0.10, modes `all`/`delta0`:
  projected `C(T)`, `S(T)`, gauge-peak position and entropy plateau.
- **A3** Projected `S^{zz}(q,ω)` at the same couplings (new driver
  `dssf_gauge.py`, Task T1).
- **A4 (human):** digitize the published sign-problem-free QSI QMC at the
  benchmark coupling (**working target J±/Jzz ≈ 0.046**; lock the exact
  reference and observable — candidate: Kato & Onoda, PRL 115, 077202
  (2015), and successors).
- **Gate G1:** projected 32-site `C(T)`/entropy (and, if available, the
  photon feature) agree with QMC within combined error at ≥2 couplings.
  Fail → the perturbative gauge Hamiltonian is not trusted → reassess.

## 4. Phase B — π-flux payoff: the lower peaks

The physically distinct, QMC-inaccessible sector (J± < 0).

- **B1** `C(T)`/`S(T)` at J± = −0.03…−0.10, modes `all`/`delta0`: show the
  bare lower peak sits at g₄ (drifting as g₄/g_hex = Jzz/3|J±| toward small
  |J±|) and the projected lower peak lands at the photon scale, with the
  `J±³` collapse (PBC drifts toward `J±²`). Both flux signs on one panel.
- **B2** Projected `S^{zz}(q,ω)`: the emergent-photon spectral function —
  low-ω weight, its dispersion across the cluster momenta, and the flux
  contrast (π- vs 0-flux photon). This is the DSSF counterpart of the lower
  `C(T)` peak.
- **B3** Full-FTLM `S^{±∓}(q,ω)` spinon continuum at one π-flux coupling for
  the complete `S(q,ω)` picture (2³² pipeline).

## 5. Phase C — experimental parameter sets

- **C1** Take the published exchange parameters (and their uncertainties)
  for Ce₂Zr₂O₇, Ce₂Sn₂O₇, Ce₂Hf₂O₇ from the recent neutron/thermodynamic
  determinations; map onto the effective `(Jzz, J±)` (octupolar-ice
  convention; note DO-doublet sign conventions).
- **C2** Predict `C(T)`, `S(T)`, and `S(q,ω)` (photon + continuum) at those
  parameters; overlay the measured `C(T)` and INS `S(q,ω)`.
- **C3** Deliverable: a controlled, contamination-free ED prediction for the
  photon feature in each material, and a statement of which flux sector each
  material's thermodynamics + DSSF is consistent with.

---

## 6. New code

- **T1 `dssf_gauge.py` (core):** projected-Heff `S^{zz}(q,ω)`. From the
  assembled projected Heff (2970²), full `eigh`, and
  `S^z_q = Σ_i e^{iq·r_i} S^z_i` (diagonal in the ice-config basis), form
  `Σ_n |⟨n|S^z_q|0⟩|² δ(ω−ω_n)` (and finite-T via Boltzmann weights over
  eigenstates). Cheap (seconds/coupling). Verified prototype exists (this
  session).
- **T2 sparse assemble + Lanczos** for the 48-site scaling check (stretch).
- **T3 plotting** (`plot_benchmark/scaling/dssf.py`) → figures.
- **T4** the full-FTLM spinon `S^{±∓}(q,ω)` is the *existing* QED pipeline;
  wire twist grid for BZ sampling (no contamination logic there).

## 7. Resources (single-node unless noted)

| Job | cores | mem | wall |
|---|---|---|---|
| A1 build 32-site o4 | 8 | 16G | 15 min |
| A2–B2 sweeps + DSSF (projected) | 8 | 8G | <10 min each |
| B3 full-FTLM spinon DSSF (2³²) | MPI | large | existing pipeline |
| E 48-site build+Lanczos (stretch) | 16 | 64G | 6–12 h |

Honest note: the projected-sector campaign (A, B1–B2, C) is small and fully
reproducible; the only heavy compute is the *uncontaminated* full-FTLM
spinon continuum (B3), which the existing distributed code already handles.

## 8. Gates & risks

- **G1 (Phase A):** projected 0-flux vs QMC agree → proceed; else methods-PRB.
- **G2 (Phase B):** projected `S^{zz}` photon feature is cluster-stable
  (16↔32↔48 trend) → the DSSF claim is robust; else report `C(T)` only.
- Referee "just twist averaging": we *prove and show* twist averaging fails
  for both `C(T)` and `S(q,ω)` (Sec. 1 table); the projector is the new
  content.
- Referee "perturbative gauge Hamiltonian": G1 QMC benchmark + order 3↔4
  convergence; cerium materials sit at small |J±| where SW converges.
- Referee "32 sites can't resolve the photon dispersion": we report the
  photon *scale and spectral weight* (cluster-converged after projection)
  and the BZ-limited dispersion honestly; twist grid interpolates.

## 9. Critical path

`A1 → A2/A3 → [G1 vs QMC] → B1/B2 → [G2] → C (experimental sets) → merge PRL.`
Everything on it except the full-FTLM continuum (existing pipeline) and the
QMC digitization runs in minutes with the current tooling.
