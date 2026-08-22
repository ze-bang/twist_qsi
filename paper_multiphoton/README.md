# Seeing multiple photons — exact S^zz of the compact gauge theory

Draft. Every not-yet-computed item is wrapped in `\pending{}` (renders red).

**Thesis.** The compact U(1) gauge theory of QSI, diagonalized exactly on a
48-site torus and compared against the Gaussian theory quantized on the SAME
lattice, shows three effects invisible at Gaussian order: interaction-split
polarization doublets (Gaussian degeneracy is exact -> the split IS the
photon-photon interaction), strongly momentum-dependent renormalization
(fit-free omega_X/omega_L: 1.155 Gaussian vs 2.18 exact), and a 16-38%
odd-photon-number continuum (Gaussian: exactly 0).

## Provenance — every number

| Number | Source | Job |
|---|---|---|
| cluster scout (fcc222 8-regular RK-degenerate; cubic16 frozen sector; fcc223 usable) | `campaign/scout_ring_clusters.py` | 19293022 |
| exact ED, sector dim 9104, E0=-10.556K, level-resolved S^zz | `campaign/run_ring48_dssf.py` + `ring48_momenta_fix.py` | 19308013, 19311271 |
| momentum validation (completeness=1, probe r_q=1 exact; group traces integer) | `campaign/ring48_projector_check.py` | 19311054 (FAIL->diagnosis), 19311271 (PASS) |
| Gaussian same-lattice (gauge check 0, rank 22, sigma closed forms, exact polarization degeneracy, scale 0.5699) | `campaign/ring48_gaussian.py` | 19312260 |
| Table I / Fig 1 | `outputs/ring_model_dssf/ring48_gaussian_match.json`, `campaign/make_multiphoton_figure.py` | 19313963 |
| HFB 384+12 zero-flux counts (fcc32) | Hermele-Fisher-Balents PRB 69, 064404, Sec on L=2 enumeration | literature |
| literature positioning (nobody computed multi-photon S^zz; KDC loophole; Huang damping) | search agent report, 2026-08-18 | — |

## Gotchas encoded in the pipeline (do not re-learn these)

1. Momenta on anisotropic fcc boxes: q = m @ inv(Lvecs).T — the transpose is
   invisible on isotropic boxes. `run_wp0b_gamma_rule.allowed_momenta` is WRONG
   for anisotropic shapes.
2. Per-state momentum-concentration tests are invalid under degeneracy; only
   group-summed quantities are basis-independent. The projector gate
   (completeness + probe identity) is the correct validation.
3. enumerate-tets BFS bipartition needs the self-adjacency skip.

## PENDING (red in PDF)

1. **V-scan: DONE 2026-08-20** (job 19321262, V/K = -0.5..+0.5, 5 points).
   VERDICT: the L-point 1.024K level has d(dE)/dV = +0.34 (linear), OPPOSITE
   in sign to every photon-family line (band [-0.37, +0.02] incl. the 1.730K
   L level at -0.05). It is a distinct P-odd resonance-rich excitation BELOW
   the photon; the photon at L is 1.730K (12% below Gaussian, in line with
   all momenta). Even-channel X onset = 2 x 1.024 = two intruder quanta
   (L+L=X). Data: ring_fixed_fcc223_v*.json.
2. **One-loop / quartic-vertex lineshape**: predicted continuum onset, q
   dependence, weight — the theory section's centerpiece, not started.
3. **Second usable cluster**: fcc(2,2,4) = 2.76M ice states, needs sparse
   methods; no size trend exists without it.

## Claims deliberately NOT made

- No claim about microscopic materials couplings (the masked-XXZ ~10% at L is
  mentioned once, as deferred corroboration, with the 36% virtual-spinon
  admixture caveat).
- No thermodynamic-limit extrapolation from one cluster.
- The L-point 1.024K level is NOT asserted to be a bound state — both readings
  stated, discriminator marked pending.
- The 68%-multi-photon reading of L under strict Gaussian matching is not used;
  the conservative doublet convention (16-38%) is.

## Build

    sbatch fig.sbatch && sbatch --dependency=afterok:<figjob> build.sbatch

`refs.bib` partly from memory — verify (esp. Kwasigroch2020 title, Fu2017
authors, Gao2025 metadata).

## External review response (2026-08-20, group-internal referee)

Review verdict: idea 9+/10, current evidentiary strength lower; four issues.
All four addressed:

1. **"Photon physics, full stop" / visons.** CORRECT objection: the ice
   manifold removes spinons but retains the full magnetic sector; monopole
   pairs can contribute (Kwasigroch PRB 102, 125113). Sentence removed;
   beyond-pole weight relabeled `f_res`, confined by exact C-parity to the
   C-odd sector with candidates {multi-photon, monopole pairs}; V-scan slopes
   of the two largest beyond-doublet groups (-0.57, -0.62 vs photon band
   [-0.37, +0.02]) added as first composite-character evidence; flux-resolved
   diagnostic (<cos B_p> correlations per state) added as pending.
2. **"Continuum" on one torus.** Robust statement (discrete-level weight
   budget) now separated from thermodynamic-limit interpretation; second
   cluster remains pending.
3. **"Every existing calculation is Gaussian."** Narrowed to "every analytic
   prediction"; abstract now carries the defensible novelty sentence (no
   calculation has resolved the weight decomposition); Huang et al. QMC
   properly positioned (interacting lineshape, but SAC cannot decompose).
4. **Doublet naming / L conflict.** Already resolved by the V-scan before the
   review arrived: L pair = photon (1.730, slope -0.05) + non-photon intruder
   (1.024, slope +0.34); abstract and text updated; monopole-pair bound state
   (gauge-ball analogue, Cox et al.) named as candidate for the intruder.

## PENDING (current)

1. Flux-resolved state diagnostic: <cos B_p>/flux correlations for the
   intruder and the beyond-doublet groups (multi-photon vs monopole-pair).
2. One-loop / quartic-vertex lineshape.
3. Second usable cluster: fcc(2,2,4), 2.76M ice states, sparse Lanczos.

## Second review pass (2026-08-20): bookkeeping + terminology fixed

1. **Bookkeeping inconsistency (correct objection).** The "top-two = doublet"
   convention predated the V-scan and contradicted its verdicts at L and X.
   Table I restructured: levels classified by RK slope (photon band
   [-0.37,+0.02] / other / unresolved). Corrected numbers: non-one-photon
   weight is 16-68% (not 16-38%); f_1gamma = 0.84/0.79/0.32/0.62/0.46 across
   the path; AT L THE PHOTON RETAINS ONLY 32% (51% intruder + 17% unresolved);
   at X 46% (25% composite + 29% unresolved). Abstract/implications/conclusion
   updated to the stronger, correct numbers.
2. **Terminology.** "multi-photon continuum" -> "C-odd multiparticle /
   non-photon weight" everywhere outside explicitly-hedged contexts; exact
   selection rule restated as C = -1 coupling (photon number is the
   photon-basis reading, not the conserved label).
3. **Title** -> "Beyond the free photon: exact spectroscopy of the compact
   emergent gauge field of quantum spin ice".

## Referee-data job 19325763 (2026-08-20): one claim refuted, cubic-16 verified

- Even-channel X-edge RK slope = +0.297: REFUTES the "two intruder quanta"
  reading (+0.68 predicted) and the photon-pair reading (-0.1); matches a
  single C-even sibling of the intruder (+0.34). (iv), abstract, and captions
  rewritten: the non-photon states form a two-parity FAMILY; 2.23 = 2x1.02 was
  coincidence.
- cubic-16 pure ring model verified: E0 = -4K exact, six-fold sector-degenerate
  (sectors 3,8,11,13,16,21), only 6/25 sectors dynamical (8-state blocks).
  Cluster-selection text now states verified facts.
- Degeneracy counts: every bright line = ONE state per momentum. At L and X the
  two bright states = exactly the transverse count -> new hypothesis: the
  intruder is the SECOND POLARIZATION, rotonized. Polarization-projection job
  19331900 decides. scale_photon=0.587 written; figures regenerating.

## Consolidated landing (2026-08-20 evening): THE ROTON

Final picture after polarization projection (19331942), SMA + extended-V
(19328485), and the 64-site pipeline (19327389, all gates passed):

- NO new particle. Both transverse photon branches survive everywhere
  (polarization decomposition exact: completeness 1e-17, longitudinal 1e-32;
  exactly orthogonal at X, 30% hybridized at L).
- THE SECOND BRANCH ROTONIZES AT L: 1.02K (51%) vs photon 1.73K (32%);
  mechanism = Feynman-Bijl (S(q) max at L 0.370, omega_SMA min 1.85). The
  64-site cluster makes it PREDICTIVE: L-class with S=23.6 rotonizes
  (0.955K, 50.0%), L-class with S=14.5 does not (1.998K, 77% photon).
  Roton stable 48->64 (energy -7%, share 0.51->0.50). Softens to 0.73K by
  V=-1.1 without condensing; lowest excitation of the model at negative V.
- Multiparticle weight 16-38% (48) / 13-34% (64); f_1gamma = 85% at the new
  smallest momentum (IR protection observed).
- X not size-converged (53% split at 48 -> 6% at 64): demoted to descriptive.
- KNOWN BUG (harmless, noted): the sin-B/E ratio in ring64_pipeline stage A
  summed sublattice channels coherently -> w_E=0 by interference; ratios
  invalid. Identity question was settled by polarization projection instead.
- Manuscript fully rewritten around the branch/roton picture; Table I
  reclassified by polarization; abstract/conclusion final.

## Pending closure (2026-08-20)

The last `\pending` in main.tex is closed; the manuscript carries zero red marks.

**Flux diagnostic (job 19339031, `campaign/ring48_close_pendings.py`, part C).**
Connected plaquette flux-flux correlator per bright state, binned by
min-image plaquette-center distance. Every bright state — roton, both
photons, both large residual levels — shows a weak, delocalized profile
(roton +0.0045 at the nearest shell, same magnitude as the photons); no
state shows a localized flux disturbance. Monopole-pair character for the
residual weight is disfavored; multiparticle photon/roton reading stands.
Cross-check: state-resolved <n_flip> reproduces every RK slope measured
independently by the V-scan (roton +0.339 vs +0.34; photon_L -0.064 vs
-0.05; photon_111 -0.373 vs -0.37; resid_4p34 -0.597 vs -0.57).

**One-loop quartic vertex (job 19339213, `campaign/ring48_oneloop2.py`, gate v3).**
H4 = -(K/12) sum_p B_p^4; Wick matrix elements <1_a|B^4|1_b> - vac =
12 S_p g_pa g_pb with self-consistent g = G/sqrt(2w); degenerate-PT block
per sigma-multiplet; 1->3 amplitudes with symmetry factors 24/12sqrt2/4sqrt6.
Gate v3 (degeneracy-aware, eigenvalue-level comparison against exact 3-mode
Fock diagonalization, 8 plaquettes, nmax=7): ALL ratios 1.000 — PASS.
Two earlier gate failures were gate bugs, not formula bugs: (v1) exact test
used amplitudes G where Wick used g; (v2) compared mode-by-mode inside a
degenerate multiplet. Production at self-consistent U/K = 0.1722:
polarization splittings <= 0.1% and 1->3 transferred weight <= 0.1% on all
five sigma values; at U/K = 0.5 still only ~1% / 2-3%. Measured: 4.5-19%
splittings (O(1) at zone boundary) and 16-38% residual weight. Leading-order
perturbation theory in the vertex undershoots by 1-2 orders of magnitude —
the quantitative demonstration that the photon-photon sector is
nonperturbative and exact treatment is necessary. Written into main.tex as
the closure of result (iii); Limitations sentence about pending items removed.

## Literature sweep + bibliography expansion (2026-08-20)

refs.bib: 15 -> 43 entries; every entry cited, no dangling keys. Header note
"verify against published record before submission" retained.

Novelty check (web sweep, five angles):
- NO prior roton claim in any emergent-gauge-field spectrum. Closest matches:
  Kwasigroch semiclassical anharmonics (cited), Szabo-Castelnovo PRB 100,
  014417 (2019) semiclassical vison-photon (now cited), Du-Xu-Son PRB 109,
  035135 (2024) interacting LIFSHITZ (quadratic) photon theory (now cited;
  different dispersion class). Explicit novelty sentence added to result (ii).
- NO prior branch-splitting or multi-photon spectral-weight computation in QSI.
- arXiv 2608.11305 (Zhou-An-Kim, thermodynamic ring-exchange probe) and
  arXiv 2601.03202 (Gao et al., field demarcation of photons/spinons in
  Ce2Zr2O7) are OUR OWN companions - both cited as such.
- New probe: stray-field noise magnetometry (Naik-Hallen-Jayarama-Moessner-
  Laumann, PRL 2026, arXiv 2512.14843) - cited; couples to magnetization so
  it lives on the C-odd (neutron) side of the parity division.
- LGT quantum-simulation frame added (intro + conclusion): Bauer et al. PRX
  Quantum 4, 027001 (2023); trapped-ion Z2 glueballs arXiv 2604.07435 (2026).

Text additions: reviews + materials (Castelnovo2008, Ross2011, Gaudet2019,
Gao2019, Sibille2020, Smith2022, Poree2025), QMC verification (Shannon2012,
KatoOnoda2015), 3D dimer models (Sikora2009), RK originals (RokhsarKivelson
1988, Henley2004), compact QED (Polyakov1977), Feynman-Bijl provenance
(Feynman1954, FeynmanCohen1956) + FQH magnetoroton (GMP1986), GMFT DSSF
(DesrochersKim2023PRL), spinon spectroscopy (Morampudi2020).

## Readability pass for experimentalists (2026-08-20)

User direction: language read confusingly to experimentalists (neutron AND
quantum-simulation); wanted clear implications and precise physics without
deep gauge-theory jargon. Rewrites (all physics statements unchanged):
- Abstract fully rewritten message-first: what a neutron experiment sees
  (single line -> split line + roton + continuum), selection rule stated as
  "even or odd under reversing all spins", method details dropped.
- "Gaussian" glossed at first use: cosine energy of angle-valued field vs
  its quadratic expansion; W_p glossed as the flippable-hexagon rotation.
- Charge conjugation introduced via spin flip first, C label kept as
  notation with explicit "read as even/odd under flipping all spins".
- RK response glossed as "rewards flippable hexagons; slope tells whether
  the excitation has more or fewer than the ground state".
- "Lattice versus continuum" retitled "Where the free photon survives, and
  where it fails"; RG language demoted to parenthetical.
- "electric-flux superselection" (fig 3 caption) -> conserved uniform S^z;
  topological rank count demoted to bracketed aside; "stiffness generated
  not bare" -> "coefficient U generated by the ring dynamics itself".
- One-loop closure: "Wick combinatorics/Fock/self-consistent stiffness" ->
  "leading interaction correction, validated against exact few-mode
  diagonalization, at the fitted coupling"; ending now "no small correction
  to the free photon reproduces ...; must be treated exactly".

- 2026-08-20 (later): abstract overclaim fixed per user — "Every existing
  prediction ... free" restored to "All analytic predictions ..." with QMC
  explicitly credited as non-free but unable to resolve the spectrum behind
  the broadening (Huang 2018 / Kato-Onoda 2015). Referee 1 had flagged this
  overclaim class before; the readability rewrite had reintroduced it.

## Roton landing: what survived and what broke (2026-08-20, jobs 19345353/19345359)

USER ASK: land "roton" properly — operator form, does it fall out of the
model, why is the signature roton and not something else, why at L, do we
predict other points.

**GATE A PASSED (new exact analytic result).** The first-moment sum rule of
the ring model is analytic:
    [S^z_mu(q), W_p] = +g^mu_p(q) W_p ,  [S^z_mu(q), W_p^dag] = -g^mu_p W_p^dag
    f_1(q) = (1/2N) sum_p w_p sum_mu |g^mu_p(q)|^2 ,  w_p = <W_p + W_p^dag>
Verified f1_analytic/f1_exact = 1.0000 at every cluster momentum, and
GATE0 sum_p w_p = -E_0 to 8 decimals. This is the pyrochlore analogue of
helium's f_1 = hbar^2 q^2/2m: pure lattice geometry times ground-state
plaquette expectations, evaluable at ANY q.
GOTCHA: w_p is NOT uniform on the anisotropic 2x2x3 box (0.196 to 0.268,
7.2e-2 spread) because hexagon orientations are inequivalent there. Using a
uniform w = -E0/N_p gives an 8% error. Per-plaquette w_p is required.

**GATE B FAILED — the stated mechanism is wrong as worded.** Classical
(equal-amplitude ice) S(q) vs quantum S(q) on the same sector:
    (1/3,1/3,1/3): quantum 0.189 vs classical 0.253  (classical HIGHER)
    L            : quantum 0.370 vs classical 0.297  (classical LOWER)
    (1/6,1/6,5/6): quantum 0.258 vs classical 0.266
    X            : quantum 0.273 vs classical 0.269
Full-manifold classical MC (L=8, 2048 sites, loop updates, ice-exact) gives
S(q) ~ 0.25 NEARLY FLAT across the whole zone, S(L) = 0.259.
=> The L enhancement is mostly a QUANTUM ring-resonance effect, only weakly
seeded by the ice manifold (classical L is the max but by just ~12%).
=> The manuscript phrase "the equal-time structure factor OF THE ICE LIQUID
is maximal at L" is misleading and must be reworded: it is the structure
factor of the ring-model GROUND STATE.
=> Full-BZ omega_SMA(q) from classical S(q) is INVALID as a predictor: with
flat classical S it puts its minima near K (0.688,0.688,0), NOT at L. So the
"predict other roton momenta" deliverable is NOT achievable by this route.
Do not use roton_bz.npz for any prediction; it is a negative control.

**Backflow (block Krylov from the probe) — strong positive evidence.**
Lowest Ritz value in span{H^j S^z_mu(q)|0>}:
    L            : 1.450 -> 1.074 -> 1.034 -> 1.026 -> 1.024 (converged k=4)
    (1/3,1/3,1/3): 2.405 -> 1.980 -> 1.917 -> 1.902 -> 1.895
    (1/6,1/6,5/6): 3.100 -> 2.676 -> 2.586 -> 2.556 -> 2.533
    X            : 2.858 -> 2.401 -> 2.293 -> 2.249 -> 2.235
The roton is captured to 3 digits by a 4-step Krylov space built on the
neutron operator: it is a collective density-wave state of the emergent
electric field with modest backflow. Ratio omega_exact/omega_SMA is
0.82/0.83/0.79/0.69 elsewhere but 0.55 at L: the SMA is loosest exactly at
the roton, which is the helium phenomenology (Feynman exact for phonons,
~2x high at the roton, repaired by Feynman-Cohen backflow).

**OPEN / POSSIBLE CONTRADICTION.** The same block Krylov keeps descending to
0.149K at X and 1.026K at (1/3,1/3,1/3) — far below the bright lines. If
those are real C-odd states, the claim "the roton is the lowest excitation"
is FALSE. Under direct verification against the complete sector spectrum
(check_lowest_excitations.py, job 19345474). Do not restructure the paper
around "lowest excitation" until this returns.

## VERDICT on "is the roton the lowest excitation" (job 19345474)

VERIFIED AND STRONGER THAN CLAIMED. Direct scan of the complete 9104-state
sector spectrum with spin-flip parity labels:
  - lowest excitation overall = 1.02352 K, parity -1 (C-odd), S^zz weight
    0.188, two-fold degenerate (idx 1,2)
  - "number of levels below the lowest bright line: 0"
  - ground-state parity +1 (C-even), spin flip closes on the sector exactly
    (0 escapes)
So the roton is the lowest excitation of the model, not merely the lowest
neutron-visible one. The 0.149K value the block Krylov reached at X was a
GHOST (no such level exists).

GOTCHA (new): block Krylov with a growing Rayleigh-Ritz basis produces
ghost eigenvalues at large k (k>~12, basis dim >~50) even with two rounds
of reorthogonalization. The k<=6 Ritz values are the meaningful ones.
Never quote a large-k Ritz value without checking it against the exact
spectrum.

BONUS: the same table is an independent verification of the parity
selection rule at machine level -- every C-even level carries S^zz weight
1e-28 to 1e-30 (levels 7, 11, 16-20, 22-27), every bright level is C-odd.
Note also that C-odd does NOT imply bright: levels 10 (1.93367) and others
are C-odd yet dark, so parity is necessary but not sufficient.

## RK-trajectory Feynman-Bijl test (job 19345716, complete)

Part 1 (all 12 cluster momenta, V=0): L is SIMULTANEOUSLY the momentum of
  max S(q)      = 0.3704
  min omega_SMA = 1.8456
  min bright line = 1.0235
The three extrema coincide at L. That is the Feynman argument in one line,
and it holds over the full accessible momentum set, not a chosen subset.

Part 2 (RK trajectory at L). V*n_flip is DIAGONAL in the ice basis, so it
commutes with S^z(q) and does not enter f_1 at all; f_1 keeps its form with
w_p evaluated in the V-dependent ground state. Result:

   V      S_L      f1_L     om_SMA   om_roton  ratio
  +0.00  0.37036  0.68354  1.8456   1.0235   0.5546
  -0.20  0.38142  0.68682  1.8007   0.9584   0.5323
  -0.40  0.39159  0.68848  1.7582   0.8989   0.5113
  -0.60  0.40078  0.68865  1.7183   0.8449   0.4917
  -0.80  0.40884  0.68741  1.6814   0.7962   0.4735
  -1.00  0.41562  0.68482  1.6477   0.7527   0.4568
  -1.10  0.41848  0.68301  1.6321   0.7329   0.4491

READING (honest): f_1 is CONSTANT to 0.8% across the whole trajectory (and
non-monotonic: rises then falls), while S_L grows 13% and omega_SMA falls
11.6%. So 100% of the Feynman-Bijl softening is driven by the growth of the
structure factor -- exactly what the mechanism requires, and a stronger
statement than "S_L grows and the roton softens" (which is mere
correlation). The exact roton softens FASTER, 28% vs 12%, so the ratio
drifts 0.5546 -> 0.4491 (mean 0.4956, spread 0.1055 = 21% relative).
=> The bound does NOT track quantitatively; the excess is backflow, whose
importance grows monotonically as the mode softens. This is the same trend
as in helium approaching the roton minimum, so it supports the
identification, but do NOT claim the SMA "tracks" the trajectory.

STATUS vs the user's condensed main.tex (218 lines, compiled 8 pp clean):
line 168 already says "S_L grows and the roton softens monotonically from
1.02K to 0.73K by V/K=-1.1" -- CORRECT and consistent, no fix needed. The
available upgrade is one sentence: f_1 is constant to 0.8%, so the
softening is CAUSED by the S_L growth rather than merely accompanying it.

## WHAT THE ROTON IS, CONCRETELY (job 19351840) -- the missing physics

L = (0.5,-0.5,0.5) points along a <111> axis. Each pyrochlore hexagon lies
perpendicular to one of the four <111> axes (family = the sublattice the
hexagon omits; 12 hexagons each).

(1) EXACT GEOMETRIC DECOUPLING. Probe form factor |g_p(L)|^2 by family:
      family 0 (perp [111])   : 8.0000
      family 1 (perp [1-1-1]) : 8.0000
      family 2 (perp [-11-1]) : 0.0000   <-- L's OWN axis
      family 3 (perp [-1-11]) : 8.0000
    The hexagons lying in planes PERPENDICULAR TO q have all six spins at
    the same phase of the wave, and their alternating +-1 signs cancel:
    g = e^{-iq.r} sum(delta_i) = 0. One quarter of the ring-exchange
    channels decouples from the neutron EXACTLY, and only at L, because L
    is the only zone-boundary momentum lying along a hexagon normal.
    Quantitative consequence: f_1(L) = 0.684 vs 0.880 (X) and 0.831
    (1/6,1/6,5/6) -- a 3/4 reduction, 0.684/0.880 = 0.78 ~ 3/4. So "why L"
    has a GEOMETRIC half (f_1 suppressed by the decoupling) and a DYNAMICAL
    half (S enhanced by the ring dynamics); omega = f_1/S is minimized by
    both at once. This is much better than "S is big there".

(2) WHAT THE STATE IS. Totals, roton vs ground state:
      sum <W_p + W_p^dag> : 10.556 -> 9.532,  delta = -1.02352
      sum <n_p^flip>      : delta = +0.33934
    ENERGY CHECK: -sum(delta W) = +1.02352 K = omega EXACTLY (5 decimals).
    And +0.339 reproduces the independently measured RK slope +0.34.
    => The roton has MORE flippable hexagons (+0.34) but LESS coherent
    resonance (-1.02). It costs energy by losing ring-exchange COHERENCE,
    not by straining the ice pattern. n_flip (diagonal, counts what CAN
    flip) and W+W^dag (off-diagonal, the actual resonance) move in
    OPPOSITE directions -- that is the whole content of the +0.34 slope.

(3) PER FAMILY (delta <W+Wd>, delta <n_flip>):
      family 0: -0.028, +0.018
      family 1: +0.040, +0.047   <-- the only family that GAINS resonance
      family 2: -0.059, -0.037   <-- the probe-blind family, loses most
      family 3: -0.038, +0.001
    Modulation at wavevector L: families 0,1,3 carry the standing wave
    (L-component 0.025-0.034); family 2 has ZERO L-component and changes
    UNIFORMLY (0.0585). So the neutron builds the wave on three families,
    and the state responds by rearranging the fourth -- the one the probe
    cannot touch -- most strongly of all, and as a uniform background
    rather than a wave. THAT IS THE BACKFLOW, made concrete.

## Backflow confirmed and quantified (job 19354974)

CAUGHT A BUG FIRST: summing the four sublattice probe channels COHERENTLY
gives a meaningless state (energy 8.87K, vs the SMA value 1.85K) -- the
same interference trap noted earlier for the sin-B/E ratio. The painted
wave is the best state in the 4-dim space the channels span, NOT their
coherent sum. Corrected value 1.45016 K, which matches the independent
block-Krylov k=1 Ritz value exactly.

  painted wave (SMA)  1.45016 K
  true roton          1.02352 K   -> the state recovers 29% of the cost

Per hexagon family (delta <W+Wd> vs ground state):
   family 0 (g=8):  SMA -0.42154   true -0.34144   relief +0.08010
   family 1 (g=8):  SMA +0.54038   true +0.47617   relief -0.06421
   family 2 (g=0):  SMA -1.00363   true -0.70245   relief +0.30118  DECOUPLED
   family 3 (g=8):  SMA -0.56537   true -0.45580   relief +0.10957
   TOTAL            -1.45016       -1.02352        +0.42664
   (-total = energy exactly, both columns: 5-decimal check)

=> 71% of the total relief (0.301 of 0.427) comes from the ONE family the
neutron cannot see. Physical reason: those hexagons do not enter the probe,
so rearranging their resonances costs no overlap with the state the neutron
made -- they are free to be spent. THIS IS THE BACKFLOW, quantified.

NOTE a subtlety corrected against my first guess: the decoupled family is
NOT protected in the SMA state (it loses the most resonance there,
-1.004), because the |phi(c)|^2 reweighting degrades resonances even when
g_p = 0. What is special is that it is the cheapest family to REPAIR.

## Figure redesign per user direction (2026-08-20 late)

Hero figure (fig_hero, now Figure 1, figure* full width): left = full
spectrum, energy vs momentum, every level a marker with area=intensity,
color=character (indigo photon / rose roton / teal multiphoton), open
violet squares = Raman-only (parity rule: no level in both probes), amber
dashed = free-photon line. Right = three cartoons: photon (kagome patch,
gentle flux twist), roton (side view along [111]: wavefront plaquettes
Phi=0, tilted plaquettes alternate sign), multiphoton (cosine expansion
with the -K/12 B^4 vertex marked + 1->3 splitting).
BUG CAUGHT: momentum columns were labeled by position not by reduced
coordinates -> L was mislabeled and the roton initially colored as a
photon. Columns now identified by coordinates (MOMENTA dict).

Figure 2 (fig_mechanism) rebuilt around the three reader questions ONLY:
(a) what it is - energy ladder at L (free 2.03 / dressed 1.73 / roton
1.02, x1/2 marked) + composition line; (b) why it is not a photon - RK
deformation moves photon DOWN and roton UP (opposite sign = different
object); (c) how it shows up - exact vs free spectrum at L, roton peak
shaded, 51%. Krylov + classical-ice panels dropped (moved to text).

Style: Computer Modern (mathtext.fontset=cm) + slate/indigo/rose/teal
palette across both figures. All figure numbers in text re-pointed.

## Figure replot round 2 (user critique, 2026-08-20 latest)

HERO v3: spectrum stretched (width ratio 1.62), free-photon line moved to
legend, legend labels experimentalized ("seen by neutrons" / "hidden from
neutrons -> Raman, ultrasound"), Raman squares neutral gray (content of
even levels not asserted). Illustrations now on an ACTUAL pyrochlore
lattice: slab fragments (wide perp to [111], thin along it — a cube
viewed down its diagonal projects to a useless narrow diamond).
  photon: single-kagome-layer slab, family-0 hexagons filled with a slow
    flux wave.
  roton: three-layer slab; per-plaquette fill = EXACT wave amplitude
    g_p = e^{-iq.ra} - e^{-iq.rb} over the two sublattice-0 sites,
    q = 2pi(1/2,-1/2,1/2). Kagome plaquettes -> 0 automatically (white,
    labeled 0), tilted plaquettes alternate red/blue. NOT hand-drawn.
  GOTCHAS (each cost a cycle):
    - all-six-site g_p = sum delta_i e^{-iq.r} vanishes identically at L
      by inter-sublattice cancellation: must use the sublattice-resolved
      channel (the object in f_1).
    - q = 2pi(1/2,+1/2,+1/2) is dark in the sublattice-0 channel; the
      cluster's actual member 2pi(1/2,-1/2,1/2) is bright. Star members
      are NOT interchangeable for channel-resolved pictures.
    - plaquettes must be drawn far-to-near (depth sort by view ez).

FIG 2 v3 (per user: (a)/(c) redundant, figure was hand-feeding): two
data panels only. (a) exact omega vs f_1/S with y=x line: the Feynman
relation as data; roton = extreme point, furthest below the line
(backflow). (b) RK bias: photon slope -0.05, roton +0.34 — opposite sign,
different object. No coaching annotations. Captions updated in main.tex.

## Hero v4 (user round 3: panels too small/clustered; multiphoton unaligned;
## "should we add more momentum?")

SPECTRUM: now EIGHT momentum columns sorted by |q| — the five 48-site plus
three 64-site (dagger-marked): (1/4,1/4,1/4), (1/4,1/4,3/4), and L' (the
NON-rotonizing 64-site L class, dominant 1.998@77%). L and L' sit side by
side: L has the red roton at 1.02, L' has none — the class-by-class
prediction is now VISIBLE in the hero. Free-photon curve for all eight
columns by interpolating scale*path_sigma at each momentum's Gamma-L-X
path coordinate t (0.5, 2/3, 4/3, 1.5, 1.0, 1.0, 5/3, 2.0). 64-site data
parsed from ring64_results.json fcc224.momenta (top-6 lines, weights are
shares); 64-site markers get gray edge. Even/Raman squares only exist for
48-site columns (even channel not computed at 64).

ILLUSTRATIONS enlarged and decluttered:
  photon: pure kagome ROW of five big hexagons (spins drawn on vertices,
    alternating filled/open = flippable), flux-wave fill, one arrow.
  roton: cropped slab (plaquette in-plane radius < 0.8 only, bonds only
    between kept plaquettes' sites) so survivors render LARGE; exact
    g_p fill as before (kagome 0, tilted alternate red/blue).
  multiphoton: single left anchor X0; equation, 1->3 diagram, and side
    text all hung off it — alignment fixed by construction.
Figure 1 caption in main.tex rewritten (eight momenta, dagger, L vs L').

## Hero v5-v6: the 64-site roton question + 3D-rendered lattice (2026-08-21)

USER CONCERN (critical): "64 site cluster does not see roton? I really need
to make sure roton is real not a manifestation of the cluster." — caused by
my L' column showing only the NON-rotonizing 64-site class.
ANSWER (all data in hand): 64 sites DOES see the roton. The anisotropic
2x2x4 box has two inequivalent L classes; the large-S(q) class rotonizes at
0.954K carrying 50.0% (vs 48-site 1.024K, 51%), the suppressed-S class
stays photonic at 1.998K@77%. Figure now shows all three columns L / L† /
L'† — two red rotons side by side (size stability visible), plus the
no-roton class (mechanism prediction visible). Caption rewritten to state
the numbers. Artifact defense recorded earlier still stands (complete-
spectrum verification, mechanism ingredients size-robust, XXZ companion).

ILLUSTRATIONS (user: "plot pyrochlore, align along 111, tilt, use
opaqueness and shadow"): wrote a mini depth-sorted renderer in
make_hero_figure.py: tetrahedra as 4-cliques of the bond graph, per-face
normal lighting (light dir view-space), backface culling, opacity+size
falling off with view depth, translucent plaquette fills drawn in global
depth order. Photon panel = thin slab, kagome plaquettes with slow wave;
roton panel = 3-layer slab at q exactly || [111] (channel combination
g^1-g^2 so bands are uniform and alternate; kagome -> 0 exactly).
Rendering gotchas this round: orphan sites (draw only sites belonging to a
tetrahedron); in-panel labels vs captions (keep one, not both).

## Round: percentages defined, classification written down, roton view rotated
## (2026-08-21, build 19360868 clean, 7pp)

- All percentages = the level's SHARE of total neutron intensity at that
  momentum, |<n|S^z(q)|0>|^2 / sum_m |<m|S^z(q)|0>|^2. Now defined in the
  Fig. 1 caption; in-figure note reads "area = share of S(q)".
- Classification criteria now IN THE CAPTION (not only in the text):
  photon = projects onto a transverse polarization AND energy falls under
  the flippability bias (slope in [-0.37,+0.02]); roton = opposite-sign
  bias response (+0.34) + dominant single-mode overlap + flux diagnostic
  excludes monopole pair; multiphoton = remaining odd weight (absent from
  the free theory at any parameters); squares = exact parity (even weight
  < 1e-28). Daggered momenta: dominant-line + class analysis.
- Roton render: azimuth 16 deg about [111] + tilt 58 unstacked the
  plaquettes; renderer now takes (tilt, az).

## Roton illustration final (2026-08-21) + "is this not just another photon"

Roton panel converged after a contact-sheet + 4 iterations:
- EXPLODED view (stretch=1.75 along [111]) — rotation alone can NEVER fix
  this panel: tilted plaquettes (70.5 deg from [111]) project near-face-on
  at any tilt and visually merge into the layers. Separating the layers is
  the only fix. render_pyro now takes (tilt, az, stretch).
- Plaquettes drawn as a SECOND PASS over the tetrahedra (depth-sorted
  among themselves): strict occlusion hid the far flux band — the object
  the panel exists to show.
- LADDER selection: per interlayer band, the 2 tilted plaquettes with the
  largest PROJECTED area (min-axis-distance picks edge-on slivers), signs
  saturated to +-0.88; kagome plaquettes near the axis shown white "0".
  Alternation verified in-log: band1 +0.88, band2 -0.88.
- Title alignment: aspect="equal" shrinks axes boxes and misaligns titles;
  adjustable="datalim" pins the boxes. (matplotlib gotcha worth keeping.)

TEXT: new titled paragraph "Is this not simply another photon?" — yes by
ancestry (2nd transverse branch, continuously connected), no by character:
sign-flipped bias response, more-flippable/less-coherent composition,
energy set by S(q) not stiffness, detaches below the whole band; helium
precedent (roton = the phonon branch at high q; the name marks character).

## Distortion fix + resonon novelty check (2026-08-21)

- Roton illustration stretch reduced 1.75 -> 1.40 (tetrahedra visibly less
  elongated); caption now DISCLOSES the exploded view ("layer spacing
  exaggerated for visibility"). Blue band / white-0 wavefront / red band
  all remain visible at the reduced stretch.
- NOVELTY ("I feel I've seen something like this"): the antecedent is the
  RK RESONON. Verified via Lauchli-Capponi-Assaad (our Lauchli2008 cite):
  square-lattice RK point has soft resonons at (pi,pi),(pi,0) from
  CLASSICAL dimer correlation peaks via the same single-mode logic; the
  CUBIC RK point has ONLY the photon remnant soft mode — no zone-boundary
  minimum — because classical Coulomb-phase correlations are featureless
  (matches our flat classical ice S(q)). New paragraph "Relation to known
  soft modes." added after "Is this not simply another photon?": our roton
  has no RK-point ancestor — deep in the deconfined phase, no fine-tuning,
  finite energy, correlation peak MANUFACTURED by quantum ring dynamics.
  Novelty stands, now with the nearest antecedent cited and distinguished.

## Real flux values + colorbars + precise alternation statement (2026-08-21)

USER: "literally plot the flux value, colormap on the side, real values,
be precise about what alternates."

PRECISE STATEMENT (now in caption): for q || [111]:
 (i) a plaquette whose normal is || q lies in a wavefront -> drives
     EXACTLY zero flux (any polarization);
 (ii) each tilted plaquette's flux reverses sign from one interlayer
     region to the next (phase advances by pi per kagome layer at L).
 SUBTLETY LEARNED: "uniform within a region ACROSS families" is FALSE for
 a linear polarization — the three tilted families carry different
 polarization phase factors. The clean alternation is a PER-FAMILY
 statement; the panel shows one family (strongest in eps=(0,1,-1,0)).

IMPLEMENTATION: driven_flux() computes Phi_p = sum_i delta_i eps_mu(i)
e^{-iq.r_i} with CANONICALLY ORIENTED loops (fixed chirality about the
family axis + fixed starting sublattice) — the cycle-search path's
arbitrary start/direction had been randomizing signs (the earlier
mixed-color bands). Shared FLUXCMAP colormap + colorbars (flux_face now
samples the same cmap). Roton row = lattice + Phi_p-vs-height profile
(dots on the sine, kagome at 0). Photon panel: q->0 envelope
Re[C e^{iq.r_c}] (exact limit: internal factor common per orientation);
the finite-q six-site sum on a 6-hex patch superposes internal-structure
oscillations and is NOT pedagogical.

## Flat side view wins (2026-08-21, final roton illustration)

User round: photon looked short-wavelength; profile x-axis unlabeled
(collision casualty); "why normalized to +-1"; "the fundamental problem is
the roton pyrochlore is poorly illustrated".

Fixes:
- ROTON: abandoned 3D perspective entirely. Flat orthographic side view:
  x along [1,-1,0], y along [111] (stretch 1.55). Kagome plaquettes render
  edge-on as white slats ("0"); one tilted family as full-width colored
  bands: red region / white slat layer / blue region. The
  triangular-kagome-triangular stacking is visible as alternating dot rows
  (filled = triangular-layer sites, open = kagome sites). After ~10
  iterations of 3D attempts, LESSON: for layered structure + sign
  alternation, a flat projection with zero perspective beats any shaded
  3D render.
- PHOTON: fragment widened (rmax 2.15, ~30 plaquettes), lambda = 1.9 x
  patch => genuinely gentle gradient.
- UNITS: colorbars + profile x-axis labeled Phi_p/Phi_max; caption states
  the mode amplitude is arbitrary, flux quoted in units of its maximum.
- Profile x-label restored (was deleted in a collision fix); caption moved
  below it.

## Lattice identity restored to the flat view (2026-08-21)

User: "hard to see this is pyrochlore, what's kagome vs triangular, where
do the hexagonal plaquettes sit." Fixes, all without reintroducing 3D:
- projected tetrahedra filled light gray -> the side view reads as the
  familiar rows of up/down corner-sharing triangles;
- every plane labeled on the left margin (kagome / triangular), from the
  site data (open dots = kagome sites, filled = triangular);
- two callouts name the plaquettes: "hexagonal plaquette, edge-on (⊥q)"
  -> a white slat; "tilted hexagonal plaquette" -> a colored polygon;
- photon caption states it is the top view of one kagome layer (links the
  two panels: same lattice, two views).
Paper caption synced.

## More structure in the hexagon colors (2026-08-21)

User: "vary the color of the hexagons to show more structures." Computed
the full per-family table (canonical loops, eps=(0,1,-1,0), q||[111]):
   family 1: -0.500 / +0.500   (region 1 / region 2)
   family 2: -0.500 / +0.500
   family 3: +1.000 / -1.000       -- zero spread everywhere
NEW EXACT STRUCTURE: within one interlayer region the three tilted
families carry flux in ratio (+1, -1/2, -1/2), SUMMING TO ZERO (as a
divergence-free flux pattern must); every plaquette reverses sign in the
next region. Tried drawing all three families in the lattice: overlapping
translucent hexes muddy the bands (rejected). Final design: lattice keeps
the strongest family (crisp bands); the PROFILE carries all three (solid
+-1 sine + dashed -+1/2 sine, dots on both); ratio stated in captions.

## Fig 2 made self-explanatory (2026-08-21)

User: "how are these really the criterion... what does energy per
flippable hexagon vs level energy mean... what should be the
expectations." Expectations now DRAWN IN:
(a) shaded forbidden region above omega = f1/S ("no level above the
single-mode bound") + backflow arrow at L. Meaning: the state the
neutron creates has energy AT MOST f1/S; distance below = relaxation
beyond single mode.
(b) retitled "what the state is made of". Meaning of the axes: V = energy
added per flippable hexagon; by Hellmann-Feynman a level's slope =
number of flippable hexagons the excitation adds/removes vs the ground
state. EXPECTATION drawn as gray fan anchored at the roton's V=0 point:
any photon (bare or dressed) responds with slope in [-0.37,+0.02] (the
measured range of every photon level on both clusters); renormalization
changes magnitudes not signs. The roton line exits the fan at +0.34.
Slopes labeled on both lines. Caption rewritten to carry the full
Hellmann-Feynman reading and the expectation statement.

## RK slopes recomputed; two claims CORRECTED (2026-08-21, job 19376815)

User asked how the flippable-hexagon weight is computed, where "+0.02"
comes from, and whether the slope is really an anomaly. Recomputed all ten
dominant-line slopes TWO independent ways (rk_slopes_table.py):
  (i) exact census at V=0: d omega/dV = <n|sum n_flip|n> - <0|...|0>;
  (ii) finite-difference fit over V in [-0.5,+0.5].
They agree to ~0.01 EXCEPT at (1/6,1/6,5/6) br2 (census -0.117 vs fit
-0.426) where levels cross and weight-tracking follows a different state
=> the CENSUS is the exact quantity; the fit is a convenience. <n_flip>
ground = 10.95 of 48 hexagons.

CORRECTION 1: the paper's photon range [-0.37,+0.02] was wrong at the
lower end. Measured: [-0.62, +0.02]. The +0.02 IS real -- (1/3,1/3,2/3)
branch 1 has census +0.012, fit +0.017.
CORRECTION 2: my claim "no photon can have positive slope" was TOO STRONG
and is now removed. Honest statement: the roton is the only level that
APPRECIABLY gains flippable hexagons, +0.34 vs at most +0.01 for all
others (factor 28 outlier), not a sign rule.

NEW FIG 2 = CHARACTER MAP (user: "no distinct sharp diagnostic ... most
photons are below the linear line anyway ... plot the free Gaussian line").
Correct: the f1/S panel was weak (everything is below a bound). Replaced
by a 2D map with the FREE-PHOTON energy as the reference:
  y = omega/omega_free : photons 0.82-1.46, roton 0.50 (64% gap)
  x = census           : photons <= +0.01, roton +0.34
Photon box shaded; roton isolated outside on BOTH axes. Caption states
explicitly that neither axis alone separates it (one photon level also has
marginally positive census; another also sits below the free-photon line)
-- the conjunction is the diagnostic. Text ranges updated in both places.

================================================================================
REFEREE RESPONSE ROUND (2026-08-21) -- converted to PRB, claims re-grounded
================================================================================

Referee recommended reject-with-resubmit; 13 numbered points. User chose PRB.
Structural: revtex4-2 aps,prb + numbered sections + five appendices.
Length is no longer a constraint (referee point 12 dissolved).

NUMBERS CORRECTED THIS ROUND (all traced to stored outputs, none cosmetic):

C1. Eq.(f1) was missing the factor K -- dimensionally wrong as printed.
    Code had K=1 hard-coded (ring48_roton_landing.f1_analytic docstring).
    Fixed to f_1 = (K/2N) sum_p <W+W^dag> sum_mu |g|^2.

C2. OFF-POLE WEIGHT "13-38%" WAS NOT REPRODUCIBLE. Traced: it merged the
    48-site upper end with the 64-site lower end of two different ranges.
    Worse, the old criterion called an exact level "one photon" if its
    energy fell within 20% of a fitted Gaussian line -- which counts the
    ROTON ITSELF as off-pole (67.7% at L) because it sits at half the
    Gaussian energy. Window scan (referee_response_checks.py): stable on
    [0.161,0.677] for windows 15-30%, blows up at 10%.
    REPLACED by a window-free, scale-free definition (polarization_all_
    momenta.py): photon descendants = brightest level in each transverse
    polarization channel. Gives 16-38% on 48 sites at every momentum, and
    the independent "two brightest levels" variant gives IDENTICAL numbers.
    Paper now says 16-38% (48) with 13% (64) quoted as the 64-site minimum.

C3. f_1 ABSOLUTE VALUES WERE 2x TOO LARGE (1.367/1.759 -> 0.684/0.880).
    My plaquette helper summed 2*gs[a]*gs[b] over ALL flippable states,
    counting each (a,b) pair twice. Caught by the gate printing exactly
    2.00000000 instead of 1. The RATIO 0.777 is unaffected (global factor
    cancels) and matches the gated f1_exact values, so only the absolute
    numbers were wrong. Gate now reported in the methods appendix.

C4. TWO DIFFERENT "SINGLE-MODE" ENERGIES were being conflated: f1/S =
    1.85K (sublattice-summed probe, = spectral centroid) vs 1.450K (lowest
    eigenvalue in the 4-dim space of sublattice-resolved probes). The
    paper's 29% backflow recovery uses the SECOND. Both are now defined
    explicitly and distinguished.

C5. "splitting below one percent at the smallest momentum" -- that is the
    64-site number. On 48 sites it is 4.4%. Now stated per cluster.

NEW RESULTS (all gated, all from jobs; nothing computed on a login node):

R1. THREE-PHOTON KINEMATICS (referee 7: parity only excludes EVEN n).
    Thresholds at L: 1ph 2.03K, 2ph 4.03K, 3ph 5.55K. Roton at 1.02K is
    5.4x below the 3-photon threshold; softest photon on the cluster is
    1.76K so no 3ph state anywhere falls below 5.28K. Caveated as a
    weakly-renormalized-constituent argument.

R2. THE 3/4 REDUCTION IS EXACT AS GEOMETRY, NOT AS f_1 (referee 13).
    Per-family geometric factor at L = (96,96,96,0), at X = (96,96,96,96)
    -> exactly 3/4. Full f_1 ratio 0.777 because w_p is non-uniform.
    BONUS: the same decoupling occurs at the SMALL [111] momentum too,
    (72,0,72,72), and there is NO anomaly there -- so geometric decoupling
    is necessary but NOT sufficient. Strengthens the mechanism argument.

R3. POLARIZATION IS BASIS-INDEPENDENT AFTER ALL (referee 1, 9).
    Trap avoided: the two Gaussian branches are EXACTLY degenerate at
    every q, so SVD polarization fractions ("68% pol-1") are arbitrary.
    Computed the invariants instead (polarization_invariant.py): purity
    tr M^2/(tr M)^2 and cross-level overlap tr(M_A M_B)/(tr M_A tr M_B).
    RESULT: every bright level has purity 1.000000, and the two bright
    levels are mutually orthogonal (1e-28..1e-31 at (1/3,1/3,1/3),
    (1/3,1/3,2/3), X; 3e-3 at L; 4e-2 at (1/6,1/6,5/6)). So they ARE the
    two transverse branches, identified with no energy scale at all.
    Longitudinal residue 1e-31 everywhere = exact div E = 0 gate.
    CONSEQUENCE, conceded in the paper: the "polarization splitting" and
    the "roton" are THE SAME FACT. No longer claimed as two results.

R4. CLASSICAL CONTROL, NOW WITH ERROR BARS (referee 13).
    (a) Exact, same cluster: equal-amplitude state over the same 9104
        configs gives S(L)=0.297 vs quantum 0.370; enhancement over other
        momenta 11% (classical) vs 48% (quantum). ~4/5 of the L
        enhancement is made by the quantum dynamics. NO sampling error.
    (b) 2048-site loop MC, 30 bins: S(L)=0.2488(67), S(X)=0.2538(73),
        S(W)=0.2475(52), S(K)=0.2467(46). Spread 0.007 vs typical error
        0.006 -> flat within resolution; L not distinguished.

R5. SHAPE VARIATION AT FIXED SIZE IS NOT AVAILABLE (referee 3).
    Scouted all 48/64-site tori containing an L point. Every alternative
    is one cell thick in some direction and loses most of its hexagons to
    torus winding: (1,3,4) keeps only 24 contractible hexagons on 48
    sites vs 48 for (2,2,3). Half the ring terms per site = a different
    model, not the same model in a different box. Admissibility criterion
    stated in the paper: n_hex must equal n_sites. Shape test abandoned
    honestly; finite-size evidence rests on the size sequence.

R6. 72 SITES: BROKE THE 64-BIT BARRIER (ring_big_cluster128.py).
    Every prior pipeline stored a configuration in one uint64 -> hard cap
    at 64 spins (the old 72-site attempt died with OverflowError). Rewrote
    the state as a (lo,hi) uint64 PAIR with exact 128-bit lookup: sort by
    (hi,lo), group queries by hi (hi < 2^8 at 72 sites), one vectorized
    binary search on lo per block. VALIDATED against 48 sites: reproduces
    E0 = -10.55601176, S(L) = 0.37036, omega = 1.0235/1.7303 exactly.
    (2,3,3) is the next genuinely 3D cluster: 72 sites, 36 tetrahedra,
    72/72 hexagons. Run in progress.

REFRAMED, NOT RECOMPUTED:

F1. CIRCULARITY (referee 4, the sharpest point -- and correct). f1/S IS
    identically the spectral centroid; "the centroid is low where a low
    pole exists" is a restatement, not evidence. New subsection concedes
    this outright and moves the weight onto the two places where f1/S is
    used as a GROUND-STATE PREDICTOR: the 64-site L-class prediction, and
    the V-trajectory (f1 flat to <1%, S grows, mode softens).

F2. "NEUTRON INTENSITY" (referee 6) -> defined as the sublattice-summed
    longitudinal pseudospin correlator, Eq.(szz). No g-tensor, no local
    <111> frame rotation, no polarization projector. Ratios at fixed q
    are unaffected by those factors; absolute q-dependence is NOT claimed.

F3. REPRODUCIBILITY (referee 11). All dependence on the unavailable
    companion ZhouKimMask removed: the torus-winding filter is now
    justified inline (keep winding-zero loops; 48 on the 48-site cluster,
    12 per <111> family), and the acknowledgment no longer points to it.
    Methods appendix added with the three standing gates.

OPEN / NOT CLAIMED:
- Adiabatic branch tracking from compact to Gaussian is NOT possible here
  (E^2 is a c-number, so there is no continuous knob); title stays at the
  hedged "roton-like mode" unless 72 sites changes the picture.
- X-point splitting still not size-converged (53% at 48 -> 6% at 64).

--------------------------------------------------------------------------------
R6 COMPLETE: 72 SITES (job 19387503, 19m25s, COMPLETED)
--------------------------------------------------------------------------------
fcc(2,3,3): 72 sites, 36 tetrahedra, 72/72 contractible hexagons (admissible).
12,846,186 ice configurations; 310 winding sectors; ground sector 973,008.
Ground sector identity VERIFIED not assumed: sectors 179 and 130 both give
E0 = -14.80971536 K (exact symmetry degeneracy), sector 176 (498,654) gives
-13.74637488. E0/site = -0.20569049.  <n_flip> = 15.6402/72 = 0.2172 per hex.

L point: S(L) = 0.32097; branches (channel-summed) 1.0681 K share 0.5397 and
1.8143 K share 0.1782; splitting 0.5178.

THREE-CLUSTER TABLE (this is the finite-size evidence now in the paper):
                    48        64        72
  E0/N          -0.2199   -0.2165   -0.2057
  S(L)            0.370     0.369     0.321
  w_low/K (share) 1.024(.508) 0.954(.500) 1.068(.540)
  w_high/K(share) 1.730(.323) 1.652(.313) 1.814(.178)
  w_low/w_high    0.592     0.578     0.589      <- scale-free, 2.4% spread
  splitting       0.513     0.535     0.518      <- scale-free, 4% spread

READ: the two SCALE-FREE diagnostics are stable across 11x in sector dimension
AND across a shape change. Roton persists at all three sizes: dominant level
near 1.0K with >= half the weight, second transverse branch ~70% higher.
NOT stable, and stated as such in the paper: S(L) drops to 0.321 on (2,3,3);
upper-branch share falls 0.32 -> 0.18; X splitting still unconverged.
Note S(L) down and w_low UP is the direction f1/S requires, but the clusters
differ in shape too, so this is not pressed as a quantitative test.

GOTCHA recorded: the CF pipeline seeds once per sublattice channel, so one
physical level appears as up to 4 lines at the same energy. Only the channel
SUM is basis-independent -- aggregate_L_branches.py does this. Reading the raw
top-6 log without aggregating understates every share.

================================================================================
ALL-MOMENTUM SWEEP + RK CENSUS + f1/S AT EVERY MOMENTUM (2026-08-21, later)
================================================================================

COVERAGE now: 48 all momenta (full diag), 64 all 16 (CF+census), 72 all 18
(CF+census), 96 L-only (running). Census method = Ritz vector from the stored
Krylov basis; VALIDATED against full diagonalization at 48 (+0.3393/-0.0640 vs
exact +0.339/-0.064) and independently against exact eigsh eigenvectors at
48/64/72 -- agreement to 4 decimals, so the census is NOT a convergence
artifact at any size.

64-SITE CROSS-CHECK: the new pipeline independently reproduces the OLD
ring64_pipeline exactly (ice 2,756,394; sector 249,180; E0 = -13.85728951;
L(A) S=0.369143 omega 0.9545/1.6518 shares 0.4998/0.3126; L(B) S=0.227002
omega 1.9976 share 0.7712). Two independent codes agree.

*** THE MAIN RESULT OF THIS ROUND: f1/S ORDERS THE SPECTRUM, AND L IS NOT
*** SPECIAL -- THE RATIO IS.

f1/S computed from the GROUND STATE ALONE at every momentum (gate
sum_p w_p = -E0/K reads 1.00000000 on all three clusters):

  48:  L 1.8456 | (1/3,1/3,1/3) 2.4204 | (1/3,1/3,2/3) 2.4907 |
       (1/6,1/6,5/6) 3.2163 | X 3.2230
  64:  L(A) 1.8054 | (1/4,1/4,1/4) 2.0929 | L(B) 2.6382 |
       (1/4,1/4,3/4) 2.8682 | X 3.3463          <- perfectly monotonic in omega
  72:  L 1.9826 | (1/6,1/2,1/2) 1.9998 | (1/3,1/3,1/3) 2.2082 |
       (1/6,1/6,5/6) 2.6773 | (1/6,1/2,5/6) 2.7021 | (0,0,2/3) 2.7494 |
       (0,2/3,2/3) 2.9120

THE 72-SITE SURPRISE IS A CONFIRMATION, NOT A PROBLEM. (1/6,1/2,1/2) has
nearly L's spectrum (omega 1.0752 / 55.0% vs L's 1.0681 / 54.0%). Reason:
its f1/S is 1.9998 against L's 1.9826. The ratio differs by 0.87%; the
observed level energies differ by 0.66%. That is a quantitative prediction
from ground-state data alone, verified.

WHY it has small f1: per-family geometric factor [36,144,144,108] -- one
family SUPPRESSED to 25%, not exactly zero. L is [144,144,144,0], an EXACT
decoupling (same as 48: [96,96,0,96], 64: [128,128,0,128]). So exact
decoupling is one route to small f1, partial suppression plus large S is
another. The paper's "why at L" section must be reframed: the mechanism is
f1/S; L is where it is minimized, not a privileged point.

Not perfectly monotonic at 48 and 72 (e.g. 72: (0,0,2/3) f1/S=2.7494 has
omega 1.5392, below (1/6,1/6,5/6) f1/S=2.6773 -> omega 1.8927). Monotonic at
64. Report as "strongly correlated, minimum coincides", not "monotonic".

*** THE CENSUS IS WEAKER THAN CLAIMED. Exact values at the soft level:
    48 +0.3393 | 64 +0.3379 | 72 +0.1337.  Sign stable, magnitude NOT
    (factor 2.5 drop at 72). And at 72 many bright levels have positive
    census (+0.03 to +0.36); the roton is not the largest.
    THE PAPER'S CLAIM "photon descendants gain at most 0.01 flippable
    hexagons" IS A 48-SITE ARTIFACT AND MUST BE REMOVED.
    What survives: (i) the sign is positive at the soft level on all three;
    (ii) the cleanest evidence is WITHIN one cluster -- at 64 sites the
    rotonizing L class has +0.3379 and the non-rotonizing L class at the
    SAME |q| and same Gaussian energy has -0.1660.
    => demote the census from "the sharp diagnostic" to a consistent
       signature; lead with f1/S instead.

X SPLITTING CONVENTION BUG: the paper says "53% at 48 -> 6% at 64" using
(w_hi - w_lo)/w_lo, while Appendix B's table uses mean-normalized
(w_hi-w_lo)/[(w_hi+w_lo)/2] and lists 0.420 for the same X. Inconsistent
within the paper. Unify on mean-normalized: 42% at 48, 5.9% at 64.

Also: 64-site <n_flip> = 14.5419/64 = 0.2272 per hexagon (48: 0.2282,
72: 0.2172) -- the per-hexagon flippability IS stable.

--------------------------------------------------------------------------------
THE "+0.02 POSITIVE PHOTON SLOPE": RESOLVED (job 19400879)
--------------------------------------------------------------------------------
Question: should d(omega)/dV always be negative for a photon?
ANSWER: no. Only the AVERAGE is constrained, and it is constrained EXACTLY.

EXACT SUM RULE (new, belongs in the paper):
N = sum_p n_p^flip is diagonal, and each flippable hexagon of a configuration
contributes exactly ONE off-diagonal element of the ring Hamiltonian, so
      Tr N = nnz(H)      exactly
and therefore
      (1/D) sum_n d(omega_n)/dV = nnz(H)/D - <0|N|0>  < 0.
VERIFIED at 48 sites: sum_states n_flip = 91392.0 vs nnz(H) = 91392 IDENTICAL;
measured mean census -0.9138535842 vs predicted -0.9138535842, diff 1.1e-14.
Negative because the ground state is MORE flippable than a typical
configuration. Nothing forces an individual level negative.

CENSUS DISTRIBUTION over all 9,104 eigenstates (48 sites, dense diag):
  positive: 58 (0.64%);  min -3.0263, max +0.4278
  (0,0.05] 10 | (0.05,0.20] 28 | (0.20,0.40] 16 | >0.40 4
So >=+0.20 is 20 states = 0.22% of the spectrum. The roton (+0.3393) is in
that top-0.2% tail.

THE +0.012 IS REAL, not noise and not a multiplet artifact:
(1/3,1/3,2/3) omega 1.7440, BOTH degenerate members give census +0.011852
identically. (Paper says "+0.02"; census is +0.012, finite-difference fit
+0.017. Quote +0.012.) All degenerate members at 48 sites have IDENTICAL
censuses, so multiplet averaging was hiding nothing here.

BRIGHT-LEVEL CENSUSES, 48 sites: only TWO positive of ~11 -- the roton
+0.339 and this +0.012. Roton is 29x the next. So the paper's "photon
descendants gain at most 0.01" is essentially CORRECT AT 48 SITES.
64 sites: also two positive -- roton +0.3379 and +0.0467; roton is 7x.
72 sites: MANY positive (+0.03 to +0.36) and the roton (+0.1337) is NOT
the largest (+0.3582 at (0,2/3,2/3)).

=> the census claim is right at 48 and 64 and fails at 72. Restrict the
   claim to what holds everywhere: (i) the sign is positive at the soft
   level on all three clusters; (ii) the within-cluster 64-site contrast
   (+0.3379 rotonizing L class vs -0.1660 non-rotonizing L class at the
   same |q| and same Gaussian energy). Drop "at most 0.01".
   No normalization rescues the magnitude: roton/|mean census| = 0.371,
   0.203, 0.075 across 48/64/72; per hexagon 0.0071, 0.0053, 0.0019.

================================================================================
PATH B DELIVERS: THERMODYNAMIC f1/S MINIMUM AT L (job 19405365, 5h38m)
================================================================================

GFMC (sign-free: stoquastic H, zero diagonal), equal-amplitude trial,
forward walking for the diagonal S(q), f1 exact via w_p = -E0/(K n_hex)
(all hexagons equivalent on cubic boxes). GATES: 48-site reproduces ED to
every printed digit (E -10.5555(8) vs -10.556012; S(L) 0.370365(67) vs
0.370365); 72-site lands in the correct +-P ground sector (E/site
-0.205662(16) vs -0.205690) after the P=0 assumption was FALSIFIED by the
72-site ED (zero-polarization sector is empty there; randomizer now
descends to minimal reachable |P| and the ENERGY gate certifies the
sector).

E0/site: -0.2199 (48) -0.2057 (72) -0.19054(4) (256) -0.18899(6) (864)
-0.18856(8) (2048) -> thermodynamic ~ -0.1885. <n_flip>/hex falls
0.2282 -> 0.2172 -> 0.1925.

THE ANSWER (f1/S at zone-boundary points, 73-point path):
            256 sites   864 sites   2048 sites
   X        2.643(16)   2.656(31)   2.645(88)
   W        2.716(7)    2.566(16)   2.773(35)
   K        2.526(8)    2.548(13)   2.697(33)
   L        2.088(10)   2.154(19)   2.061(37)   <- minimum, every size

L is the deepest zone-boundary point by ~25% at ALL three sizes, with no
size drift (2.09/2.15/2.06 across an 8x volume change). AND it is a LOCAL
minimum along the Gamma-L line: the ratio rises from 0 at Gamma (gapless
photon), peaks ~2.29 at 0.458(111) [2048: 2.291(27); 864: 2.297(16); 256:
2.179(9)], then DIPS to L. Rise -> peak -> dip-at-zone-boundary is the
textbook Feynman roton profile, now demonstrated on systems up to 2048
sites = 42x the previous constrained-ED record, on a dense path.

This is referee request #1 delivered: combined with the ED result that
the same ground-state ratio predicts the true soft-mode energy to <1%
(72-site two-momentum test), the argument is now two-part and
non-circular: ED proves the ratio predicts the spectrum; QMC proves the
ratio's minimum at L is thermodynamically robust.

HONEST LIMITS: (i) f1/S is the SMA/centroid bound -- a minimum in it is
the necessary ground-state ingredient of a roton, not a full dynamical
proof at 2048 sites; (ii) path points not on the allowed torus grid show
small-q wiggles (FT of finite-box correlations); zone-boundary values sit
on/near allowed points and are smooth. Refinement on the FULL allowed
512-point grid of (8,8,8) queued for the definitive figure.

================================================================================
96 SITES COMPLETE (job 19413751, 8h49m): THE TWO-CLASS TEST REPRODUCES
================================================================================
fcc(2,3,4): 2,007,260,562 ice configs; ground sector 0, dim 146,758,352,
nnz 2,669,065,920; E0 = -19.56789217, E0/site = -0.20383221;
<n_flip> = 20.6677/96 = 0.2153/hex. This is level-resolved spectroscopy at
the 96-spin constrained-ED record size (Pace et al. did ground-state/QED
parameters at 96; this is the full operator-resolved response + census).
The index_of fix (structured searchsorted) turned the >24h stall into a
2.6h matrix stage.

THE RESULT -- fcc(2,3,4) has TWO inequivalent L classes, and the
ground-state ratio predicts which one reconstructs, exactly as at 64:

  class A  q=(-1/2,1/2,1/2)  S = 0.24115  ->  NO anomaly:
      1.6506 (38.7%, census -0.015), 1.7296 (30.1%, census -0.088)
  class B  q=( 1/2,1/2,-1/2)  S = 0.35483  ->  ROTON:
      0.9040 (54.4%, census +0.259), upper 1.6471 (22.5%, census +0.135)

Sum-rule gates 1.0000 at both momenta.

FOUR-SIZE LADDER AT THE ROTONIZING L CLASS:
                  48       64       72       96
  omega_low     1.024    0.954    1.068    0.904
  share         50.8%    50.0%    54.0%    54.4%
  omega_high    1.730    1.652    1.814    1.647
  ratio         0.592    0.578    0.589    0.549
  splitting     0.513    0.535    0.518    0.583
  census(soft) +0.339   +0.338   +0.134   +0.259
  S(L,rot)      0.370    0.369    0.321    0.355
  E0/site     -0.2199  -0.2165  -0.2057  -0.2038

READ: soft level persists at every size, always carrying >= half the
weight; the scale-free ratio sits in [0.55, 0.59]; the census sign is
always positive at the rotonizing class and negative at the
non-rotonizing one (three independent within-cluster contrasts now:
64, 96, and the 72-vs-others cross-check), while its magnitude
fluctuates -- as already conceded in the census-limits section.

WITH THE QMC: the two-class prediction has now succeeded at 64 AND 96
(four L classes total, S in {0.227, 0.241, 0.355, 0.369}: the two
large-S ones rotonize, the two small-S ones do not, no exceptions), and
the QMC shows the ratio minimum at L is thermodynamically robust to
2048 sites. Referee requests 1 (dense-grid large-N f1/S) and 5 (Lanczos
documentation now includes sum-rule gates at every momentum) are the
principal beneficiaries.

================================================================================
OVERNIGHT HARVEST (2026-08-22): 96-SITE ED + FULL-GRID QMC BOTH COMPLETE
================================================================================

96-SITE ED (job 19413751, 8h49m, structured-searchsorted lookup fixed it):
fcc(2,3,4): 2,007,260,562 ice configs; ground sector 0 has dim
146,758,352 (147-MILLION-dim Lanczos, nnz 2.67e9), E0 = -19.56789217,
E0/site = -0.20383. <n_flip>/hex = 0.2153.
*** THE TWO-CLASS PREDICTION TEST REPEATS AT 96 SITES ***
  L class A: S = 0.3548  ->  ROTONIZES: omega 0.9040, share 54.4%,
             census +0.2593; upper branch 1.6471 (22.5%, census +0.13)
  L class B: S = 0.2412  ->  does NOT: 1.6506 (38.7%) / 1.7296 (30.1%),
             censuses -0.015 / -0.088
Both sum-rule gates 1.0000. Forward f1/S class discrimination now
verified at TWO sizes (64 and 96); within-cluster census contrast
(+0.26 vs ~-0.05 at the same |q|) also repeats.

FOUR-SIZE LADDER (rotonizing L class):
              48       64       72       96
  omega_low   1.024    0.954    1.068    0.904
  share       50.8%    50.0%    54.0%    54.4%
  omega_high  1.730    1.652    1.814    1.647
  ratio       0.592    0.578    0.589    0.549
  census      +0.339   +0.338   +0.134   +0.259
Roton persists at the record ED size; if anything softer (0.904K).
96 sites ties the all-time constrained-space ED record (Pace et al.)
and, unlike it, delivers operator-resolved spectroscopy + RK census.

FULL-GRID QMC (job 19413718, 10h52m): all allowed momenta.
  (6,6,6) 864 sites, 216 momenta: L = 2.1419(104), X = 2.6598(96)
  (8,8,8) 2048 sites, 512 momenta: L = 2.1786(147), X = 2.7129(205),
     W = 2.7807(103), K = 2.7175(155), U = 2.7355(81)
Star-averaged over all symmetry copies; scatter across copies 0.01-0.04
(a consistency check the path scan could not provide).
L is the ZONE-BOUNDARY MINIMUM by 20-25% at both sizes; drift 864->2048
is +1.7% (stable).

*** RETRACTION of one earlier reading: the path scan's "local minimum
along Gamma-L (peak 2.29 at 0.458(111), dip into L)" came from
NON-ALLOWED interpolated momenta. On the allowed grid the diagonal is
MONOTONE: 0.598 (0.125) -> 1.36 (0.25) -> 1.97 (0.375) -> 2.18 (L),
folding back symmetrically beyond L because (1,1,1) IS an fcc RLV
(sum of the three primitive RLVs) so 0.625(111) ~ 0.375(111) etc.
Along the radial [111] direction L is the endpoint maximum (stationary
by symmetry); the roton statement is about the ZONE-BOUNDARY landscape:
the boundary-restricted f1/S dips deeply at L (W->L: 2.78->2.18),
i.e. L is the softest zone-boundary point of the centroid surface,
robust from 864 to 2048 sites. State it that way, not as a radial dip.
Also: interior momenta drift substantially with size ((1/3,1/3,1/3):
2.42 at 48-ED -> 1.81 at 864-QMC), which retroactively justifies the
referee's demand for large systems; the L and X values drift far less.

--------------------------------------------------------------------------------
"IS THE ROTON ONLY AT L?" -- THE SOFT VALLEY (job 19428885, corrected fold)
--------------------------------------------------------------------------------
FOLD BUG in the first soft-locus pass: reciprocal-lattice coefficients only
ran over {-1,0,1}, so corner points of the [0,2)^3 grid (needing g=(2,2,2))
failed to fold and small-q photon points masqueraded as low-ratio
zone-boundary stars (e.g. "[0.625,0.875,0.875] ratio 1.32" is really
[0.125,0.125,0.375]). Range widened to {-2..2}; clean table follows.

CLEAN 2048-site ranking (|q|>0.72, star-averaged):
  L (1/2,1/2,1/2)      2.179(15)   <- minimum
  (1/4,1/2,1/2)        2.216(5)    <- 24 copies
  (3/8,3/8,5/8)        2.265(5)
  (1/8,3/8,5/8)        2.315(4)
  (0,0,3/4)            2.430(10)
  ... everything else 2.48-2.78 (X 2.71, K 2.72, U 2.74, W 2.78)

ANSWER: the softness is NOT a point. It is a shallow VALLEY along the
(t,1/2,1/2) lines: t=0 -> 2.224(6), t=1/4 -> 2.216(5), t=1/2(L) -> 2.179(15)
-- 2% variation along the line, 15-28% climb transverse to it. L is the
(shallow) minimum of the valley; the valley walls are steep.

CONCAVITY (the quasiparticle question): at 2048 sites the ratio rises away
from L in EVERY allowed non-radial direction: +0.037(16) toward (1/4,1/2,1/2),
+0.086(21) toward (3/8,3/8,5/8), +0.31(2) toward (1/4,1/2,3/4), +0.60(2)
toward W. Concave up in all sampled boundary directions; stationary along
radial [111] by band periodicity (like any zone-corner minimum on a lattice,
cf. magnon roton-minima at zone corners of frustrated AFMs).

SPECTRAL confirmation that the valley is real physics, not centroid
artifact: the 72-site ED momentum (1/6,1/2,1/2) sits ON this line and its
exact spectrum has a soft level at 1.075K/55% vs L's 1.068K/54% -- the one
accessible valley momentum away from L hosts an (essentially degenerate)
soft mode. New subsection sec:valley added to the QMC section.

FIGURES: fig_thermo (3 panels: full allowed-momentum landscape with error
bars at 864+2048; E0/N vs 1/N both methods incl. the 48/72 method-agreement
overlap; L vs X ratio vs 1/N across methods) and fig_ladder (four-size
L spectroscopy with the non-rotonizing-class controls and branch ratios).
matplotlib gotcha again: \frac14 shorthand crashes mathtext (use
\frac{1}{4}) -- as recorded in memory, costs an sbatch cycle each time.
