# Simulation plan — π-flux quantum spin ice: a thermodynamic flux discriminant beyond QMC

**Target:** one merged PRL from `twist_qsi_demo` (method) + `gauge_probe_prl`
(physics). Status of this document: submittable campaign plan with resource
specs, job DAG, go/no-go gates, and the one pivotal piece of new code fully
specified. Most core jobs are minutes-to-hours single-node; the cluster buys
the 48-site scaling point, systematic grids, and reproducible provenance.

---

## 1. The one-sentence thesis

> The emergent photon of quantum spin ice responds to a uniform conjugate
> field with a **sign locked to the gauge flux** — softening the π-flux
> vacuum, hardening the 0-flux one — a background-free thermodynamic
> discriminant of the flux sector. We compute it on a 32-site pyrochlore
> with finite-size winding contamination removed by an exact zero-transport
> projection, **validate the method against sign-problem-free 0-flux QMC**,
> and deliver the discriminant in the **π-flux regime QMC cannot reach**
> (relevant to the cerium pyrochlores Ce₂Zr₂O₇/Ce₂Sn₂O₇/Ce₂Hf₂O₇).

Why PRL (honest): the *method* alone is PRB. The PRL is the physics it
enables — a controlled, benchmarked result in the sign-problem sector that
is currently a gap in the literature, plus a measurable materials-facing
prediction (flux-locked sign of the thermal-expansion / magnetocaloric
response).

**The pivot that makes it robust:** QMC is sign-problem-free for J± > 0
(0-flux) and sign-problematic for J± < 0 (π-flux). Every controlled QSI
thermodynamics result (Kato–Onoda and successors) lives at 0-flux *because
that is the side QMC can reach*. We benchmark there, then cross into π-flux.

---

## 2. Physics deliverables → figures

| Fig | Content | Cluster | Status |
|-----|---------|---------|--------|
| 1 | Concept + level scheme + the transport projector in one panel; PBC vs projected C(T) at one coupling both fluxes | 32 FCC | data exist |
| 2 | **Benchmark:** projected C(T), S(T), photon scale vs published 0-flux QMC at J±≈0.046 | 32 FCC | needs QMC digitization |
| 3 | **Payoff:** π-flux gauge thermodynamics — C(T), entropy release, J±³ collapse of Tpk (PBC drifts to J±², projected stays J±³) | 16+32+48 FCC | 16/32 done; 48 + fig TBD |
| 4 | **Hero:** flux-locked response — B(λ) sign flip (soften π / harden 0), projected, size-converged 16↔32 | 16+32 FCC | **pivotal, §5** |
| S1–S* | matched-pair theory, δ=0 projector, mask validation, convergence | — | drafted in current papers |

---

## 3. Method recap (already validated; supplement material)

- Effective ring-exchange (gauge) Hamiltonian in the ice manifold; couplings
  scale as Jpm^k **exactly** → build **once** at Jpm=1, reuse for every
  coupling and both signs.
- Zero-transport projector: keep only ice-manifold matrix elements whose
  flip-set dipole ρ ≡ 0 (mod L). Two implementations, cross-validated on
  16 sites to <5%:
  - perturbative row-filter `select_rows(..., "delta0")` (all clusters);
  - nonperturbative `SzBasis.transport_mask` on the exactly downfolded H
    (16-site only; validates the truncation).
- Confirmed cheap: 32-site SW order-4 build **218 s**, row table **1.6 MB**,
  full sweep (many couplings × modes, dense diag of 2970) **~13 s**.

---

## 4. Computational phases and jobs

Ice-manifold sizes (fixed): 16 cubic = 90; 32 FCC = 2970 (dense full
spectrum trivial); 48 FCC = 87546 (Lanczos low-spectrum only).

### Phase A — production build (one job, the only "expensive" core job)
Build the shared row tables once.

- **A1** `gauge_thermo.py build --basis fcc --shape 2 2 2 --order 4
  --out rows/rows_fcc222_o4.npz` → 1 node, 8 cores, 16 GB, **15 min** wall
  (measured 218 s; margin for cluster I/O). Also build `--order 3` (cheap)
  for the convergence cross-check.
- **A2** (validation) 16-cubic order-4 build, seconds.

### Phase B — 0-flux benchmark (the credibility firewall)
- **B1** `gauge_thermo.py sweep --rows rows_fcc222_o4.npz
  --jpm 0.03 0.046 0.05 0.07 0.10 --modes all delta0 --order 4
  --out data/thermo_0flux.json` → seconds. Outputs C(T), S(T), Tpk, plateau.
- **B2 (human, off-cluster):** locate + digitize the published 0-flux QMC
  benchmark. Working target J±/Jzz ≈ **0.046**; lock the exact reference and
  observable (C(T) curve and/or entropy plateau and/or photon-scale peak).
  Candidate sources: Kato & Onoda PRL 115, 077202 (2015); subsequent SSE
  QSI studies. **Gate G1:** projected 32-site C(T)/entropy must agree with
  QMC within combined error at ≥2 couplings, or the perturbative-gauge
  approach is not trusted → stop and reassess.
- **B3** convergence: repeat B1 at order 3 vs 4; report Tpk drift (expect
  ≲ few % at |J|≤0.05).

### Phase C — π-flux payoff
- **C1** `gauge_thermo.py sweep --jpm -0.03 -0.046 -0.05 -0.07 -0.10
  --modes all delta0 --order 4 --out data/thermo_piflux.json` → seconds.
- **C2** the **J±³ scaling figure**: Tpk/ghex vs |J±| for PBC (drifts up as
  1/|J±| ∝ g4/ghex — measured 5.4→3.1→2.7 at −0.03/−0.05/−0.08) vs projected
  (flat, O(1)). This is the necessity argument as production data, both
  signs on one panel. Post-processing script `plot_scaling.py` (to write).

### Phase D — response (the hero; needs the new code of §5)
- **D1** build the O(λ²) source-induced ice-manifold operator δH_λ² for
  drives {uniform, [111] = n·z_i} at 16 and 32 sites (§5).
- **D2** B_proj(λ) both fluxes, both clusters; extract the induced-exchange
  coefficient c. **Gate G2:** c(16) ≈ c(32) within ~10–20% (bulk
  polarizability), and the projected B keeps its **sign flip** (soften π /
  harden 0) at both sizes. If c drifts strongly, the response claim
  weakens to "suggestive" → demote Fig 4 to supplement, keep A–C as the PRL.

### Phase E — 48-site scaling point (stretch; the one true cluster job)
- **E1** `gauge_thermo.py build --basis fcc --shape 3 2 2 --order 3
  --out rows/rows_fcc322_o3.npz` → 1 node, 16 cores, 64 GB, **6–12 h**
  (ice 87546; order 3 sufficient for the 2nd-order winding diagnostic).
  Requires **sparse assemble + Lanczos** (§6 task T2).
- **E2** `gauge_thermo.py sweep --rows rows_fcc322_o3.npz --lanczos
  --lanczos-k 1500 --jpm -0.05 0.046 ...` → tens of min/coupling.
  Adds the third point to the J±³ scaling and the entropy-plateau size trend.

---

## 5. PIVOTAL new code: O(λ²) source response in the ice manifold (Task T1)

This is the make-or-break for Fig 4 and the only substantial new code. It
computes the projected flux-asymmetric response at 32 sites **without** the
infeasible Sz=±1 resolvent (that sector is 3.3×10⁹ at 32 sites).

**Physics.** The source X = Σ_i f_i (S⁺_i + S⁻_i) is charged under the
emergent U(1): one flip creates a spinon pair, so its O(λ²) action in the
ice manifold is δH = −P X G X P, G = Q/(0−H₀) with H₀ the Ising energy
(denominators independent of Jpm — same structure as the SW engine).

**Algorithm (extends `ice_pt_lib`):**
1. `apply_X(cl, states, cols, N, amps, f)` — single-site vertex: for each
   site i, flip i (S±), multiply amp by f_i, update the transport bookkeeping
   (raise adds +r_i to the dipole ρ, lower subtracts). Mirror of `apply_V`
   but one site, no bond.
2. Chain per ice column c: `X` (→ one-pair sector, record N, ρ) → divide by
   Ising resolvent (0−E_pair) → `X` (→ back to ice) → keep ice-landing rows.
   Assemble the δH row table exactly like `sw_effective` (dedupe on
   (s,t,N,ρ)); reuse `transport_delta`/`select_rows("delta0")` verbatim.
3. Response: for λ grid, diagonalize `assemble(gauge H) + λ² δH` (and the
   masked versions), read Tpk(λ), fit B [Eq. B=slope of 1−Tpk(λ)/Tpk(0) vs
   λ²/(|J|Jzz)]. Extract c from the diagonal/rescale component (reuse the
   decomposition in `kappa_lambda2_exact`).

**Validation gate:** at 16 sites, δH from T1 must reproduce the exact
resolvent result of `kappa_lambda2_exact.py` (B_exact +3.3/−0.05 etc.) to
<2%. Only then trust the 32-site output.

**Cost:** a 2-hop chain per ice column with Ising denominators → comparable
to a low-order SW build; **minutes** at 32 sites. Drives: uniform f_i=1 and
[111] f_i = n̂·ẑ_i = (1,−⅓,−⅓,−⅓) per sublattice (traceless).

---

## 6. Supporting code tasks

- **T2 (for Phase E):** sparse `assemble` — return `scipy.sparse` from the
  row tables (current `rows_to_matrix` is dense; 87546² dense = 122 GB,
  infeasible). The effective H is very sparse (~50 nnz/row); Lanczos path
  already wired in `gauge_thermo.sweep --lanczos`.
- **T3:** `plot_scaling.py`, `plot_benchmark.py`, `plot_response.py` →
  Figs 2–4 from the JSON outputs. Style-matched to existing paper figs.
- **T4:** merge manuscripts — new `paper/` combining the physics lead
  (gauge_probe κ + flux discriminant) with the method as compact
  methods+supplement. (Writing task, after G1/G2.)

---

## 7. Resource summary

| Job | Node/cores | Memory | Walltime | Notes |
|-----|-----------|--------|----------|-------|
| A1 build 32-site o4 | 1 / 8 | 16 GB | 15 min | once; reused everywhere |
| B/C/D sweeps 32-site | 1 / 8 | 8 GB | < 10 min each | dense diag 2970 |
| D1 response build 32 | 1 / 8 | 16 GB | ~30 min | Task T1 |
| E1 build 48-site o3 | 1 / 16 | 64 GB | 6–12 h | stretch; Task T2 |
| E2 sweep 48-site | 1 / 16 | 64 GB | ~1 h | Lanczos k=1500 |

Total core-hours are modest (dominated by E1). Honesty note: this is **not**
an HPC-scale campaign — it is a small, fully reproducible one; the cluster's
value is the 48-site point, parallel grids with margin, and provenance.

---

## 8. Go/no-go gates (decision points)

- **G1 (after Phase B):** projected 32-site vs 0-flux QMC agree ≥2 couplings
  → proceed. Fail → the perturbative gauge Hamiltonian is not
  quantitatively trustworthy; fall back to a methods-only PRB.
- **G2 (after Phase D):** c(16)≈c(32) and sign flip robust → Fig 4 is the
  hero. Fail → response to supplement; PRL rests on A–C (benchmarked π-flux
  gauge thermodynamics beyond QMC) which is still a defensible PRL.
- **G3 (Phase E optional):** 48-site confirms J±³ collapse → strengthens
  Fig 3; not required for submission.

---

## 9. Risk register (referee-facing)

| Risk | Mitigation |
|------|-----------|
| "Just twist averaging (Ceperley)" | We prove observable-avg fails (kills only odd powers); operator-level δ=0 is new; benchmark shows it matters. |
| "Perturbative gauge H untrustworthy" | G1 QMC benchmark at the same coupling; order 3↔4 convergence (B3); cerium materials sit at small J± where SW converges. |
| "32 sites can't resolve the photon" | We claim gauge **thermodynamics + response**, not dispersion; BZ bound stated honestly; Tpk→0.42 ghex is BZ-limited, not contamination. |
| "Flux-asymmetric response = finite-size" | It is a near-cancellation of a bulk (spinon-polarizability) term and a winding term; we report the **projected** response and its 16↔32 convergence (G2). |
| "Benchmark coupling arbitrary" | Use the exact coupling/observable of the cited QMC; report several couplings, not one. |

---

## 10. Submission order (job DAG)

```
A2(16 validate) ─┐
A1(32 build o4) ─┼─► B1 ─► [G1 vs QMC:B2] ─► C1 ─► C2(scaling fig)
                 │                                        │
                 └──────────────► T1(response code)──► D1 ─► D2 ─► [G2] ─► Fig4
                                                                          │
E1(48 build, T2)  ─────────────────────────► E2(48 sweep) ─► [G3] ───────┘
                                                                          ▼
                                                            T4 merge → PRL draft
```

Critical path to a submittable PRL: **A1 → B1 → G1 → C1/C2 → T1 → D1/D2 →
G2 → T4.** Everything on it except T1 and the QMC digitization already runs
in minutes with the current tooling.
