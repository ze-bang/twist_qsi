# Response to the Referee

We thank the referee for a careful and constructive report. The
criticisms were, in our judgement, correct: in several places we had
quoted numbers whose definitions were not stated, and in one place we
presented as evidence something that is closer to an identity. We have
rewritten the manuscript accordingly and, following the referee's own
suggestion, resubmit it to Physical Review B as a longer article, which
gives the room the additional definitions and checks require.

Two corrections we found while responding are worth flagging at the
outset, because they change numbers in the previous version:

* The reported off-pole spectral weight, "13-38%", was **not
  reproducible**. It merged the upper end of the 48-site range with the
  lower end of the 64-site range, and it rested on a criterion (an
  energy window around a fitted Gaussian line) that we now agree is not
  a definition. We have replaced the criterion entirely; the corrected
  48-site range is 16-38%. Details under point 2.
* The absolute values of $f_1$ quoted in the text were a factor of two
  too large, from a double-counted plaquette sum in an analysis helper.
  The error was caught by a gate that should have printed 1 and printed
  exactly 2. All ratios, including the 3/4 result, were unaffected,
  because the factor cancels; only the absolute numbers were wrong, and
  they are corrected against the independently gated values.

---

### 1. "Roton-like" terminology and branch tracking

We accept that the previous version asserted a branch identification it
had not demonstrated, and we have separated two claims that were
conflated.

We now state plainly that **adiabatic branch tracking is impossible in
this model**, and why: since $E_\ell = S^z_\ell = \pm 1/2$ makes
$E_\ell^2$ a c-number, there is no bare electric stiffness to tune, and
therefore no continuous parameter connecting a free photon to the
interacting problem. Any claim of adiabatic ancestry would be
rhetorical. We make none.

What we do establish is a **channel identification**, made
basis-independent. The referee's concern applies with full force to our
earlier polarization statements, and in fact more strongly than the
report indicated: the two Gaussian transverse branches are *exactly*
degenerate at every momentum, so the singular-value basis inside that
subspace is arbitrary and a statement like "68% polarization 1" is
meaningless. We therefore computed the invariants instead --- the
polarization density matrix $M_n = A_n^\dagger A_n$, its purity
$\mathrm{tr}M^2/(\mathrm{tr}M)^2$, and the cross-level overlap
$\mathrm{tr}(M_AM_B)/(\mathrm{tr}M_A\,\mathrm{tr}M_B)$, all invariant
under rotations of the degenerate subspace.

Every bright level has purity $1.000000$, and the two brightest levels
at each momentum are mutually orthogonal (overlap $10^{-28}$ to
$10^{-31}$ where the little group forbids mixing; $3\times10^{-3}$ at
$L$). They are therefore the two transverse branches, identified with
no energy scale and no fitted parameter. The full table is now
Appendix B. The title has also been made more specific, to
"Roton-like softening of a transverse photon branch in quantum spin
ice".

### 2. The Gaussian benchmark and the photon overlaps

Fully accepted; this was underspecified. Appendix A now gives the
Gaussian construction explicitly: the quadratic Hamiltonian, the
alternating lattice curl, the normal modes as singular values of the
curl in the momentum block, the three validations (the gauge identity
$C d_0 = 0$, exactly two nonzero singular values per momentum, Bloch
orthonormality and probe/mode pairing), and the fact that **exactly one
number is fitted**, the overall scale $\sqrt{2UK}=0.5868K$, since $U$
is generated rather than bare. We also state the limitation this
creates: the benchmark cannot test an overall velocity renormalization,
only statements invariant under uniform rescaling of the branch.

On the "one-photon" classification, the referee's objection led us to
discard our previous criterion rather than defend it. Testing it, the
20% energy window counted the anomalous $L$ level *itself* as off-pole
(67.7% at $L$), precisely because it sits at half the Gaussian energy
--- which is circular in the unhelpful direction. The replacement uses
no energy scale at all: the photon descendants are the brightest level
in each transverse polarization channel, and the off-pole weight is
everything else. An independent variant (the two brightest levels)
gives identical numbers at every momentum. The corrected range is
16-38% on 48 sites, with 13% the 64-site minimum.

### 3. Finite-size and shape checks

We report four quantities at $L$ across sizes, as requested, and we
have to report one negative result honestly.

**Shape variation at fixed size is not available at these sizes.** We
enumerated every 48- and 64-site torus containing an $L$-type momentum.
Every alternative to $2\times2\times3$ and $2\times2\times4$ is one
unit cell thick in some direction, and in those the majority of
six-cycles close only by winding around the torus. The
$1\times3\times4$ cluster, for example, retains only 24 contractible
hexagons on 48 sites against 48 for $2\times2\times3$. Keeping only
the contractible ones leaves half the ring terms per site --- a
different Hamiltonian, not the same one in a different box. We now
state the admissibility criterion (the number of contractible hexagons
must equal the number of sites) and abandon the shape test explicitly
rather than presenting a misleading version of it.

Instead we extended the **size** sequence. This required rewriting the
state representation: every previous pipeline stored a configuration in
a single 64-bit word, which caps the calculation at 64 spins. States
are now pairs of 64-bit words with exact 128-bit lookup. The
implementation is validated against the 48-site cluster, which it
reproduces exactly ($E_0=-10.55601176$, $S(L)=0.37036$,
$\omega=1.0235/1.7303$).

[72-SITE RESULTS TO BE INSERTED]

We also emphasize a check already present but previously undersold: on
the 64-site cluster the anisotropy splits the $L$ star into two
inequivalent classes with the same $|\mathbf q|$ and the same Gaussian
energy but different ground-state structure factors, $S=0.369$ and
$S=0.227$. The mechanism predicts in advance which reconstructs. It
does: the first carries a level at $0.954K$ with 50.0% of the weight,
the second has its dominant level at $1.998K$ with 77% and no anomaly.

### 4. Circularity of the Feynman ratio

This was the sharpest point in the report and we accept it without
reservation. A new subsection states outright that $f_1/S$ *is* the
spectral centroid of the state created by the probe, so noting after
the fact that the centroid is low where a heavy low-lying pole exists
is a restatement and not evidence. We no longer use it that way.

The non-circular content is that both factors are properties of the
**ground state alone** --- $f_1$ from the double commutator, $S$ from
the equal-time correlator --- so the ratio can be evaluated before any
excited state is computed and used as a predictor. The manuscript now
rests the argument on the two places where it is used in that
direction: the 64-site class prediction above, and the trajectory under
the Rokhsar-Kivelson bias, where $f_1(L)$ is constant to better than
one percent while $S(L)$ grows and the level softens, so the
ground-state factor moves first and the spectrum follows.

### 5. Missing factor of $K$

Correct, and corrected. The equation now reads
$f_1 = (K/2N)\sum_p \langle W_p+W_p^\dagger\rangle \sum_\mu |g^\mu_p|^2$.
The factor had been set to unity in the code and dropped in
transcription.

### 6. "Neutron intensity"

Accepted. We were computing the sublattice-summed longitudinal
pseudospin correlator and calling it neutron intensity, which is not
defensible. The observable is now defined explicitly in its own
subsection, together with what is *not* included: the ionic form
factor, the $g$-tensor, the rotation from the four local
$\langle111\rangle$ frames, and the polarization projector. We note
that these factors are common to all eigenstates at a given $\mathbf q$
and cancel from the ratios we quote, and we correspondingly make no
claim about the absolute momentum dependence of neutron intensity. All
affected wording has been changed throughout.

### 7. Three-photon states

The referee is right that parity excludes only even photon number. We
have added an explicit kinematic test. Minimizing the total Gaussian
energy over all momentum-conserving partitions on the same torus gives,
at $L$, thresholds of $2.03K$, $4.03K$ and $5.55K$ for one, two and
three photons. The level lies at $1.02K$, a factor of 5.4 below the
three-photon threshold, and no three-photon state anywhere on the
cluster falls below $5.28K$ because the softest available photon
already costs $1.76K$. We state the limitation: this argument counts
the constituents at their Gaussian cost, so it excludes multi-photon
states of weakly renormalized photons.

### 8. Backflow decomposition

Now defined exactly, in Appendix C. Because the ring term is the entire
nonconstant Hamiltonian inside the ice manifold, the excitation energy
of any state decomposes into plaquette contributions with no remainder,
and grouping plaquettes by hexagon family gives a four-term
decomposition whose totals reproduce the variational and exact energies
exactly. The table is given. Of the $0.427K$ recovered, $0.301K$
(71%) comes from the family with $g_p(L)=0$ identically.

We also found that two different "single-mode" energies were being
conflated in the previous version, and have separated them: $f_1/S =
1.85K$ (the sublattice-summed probe, i.e. the centroid) and $1.450K$
(the lowest eigenvalue in the four-dimensional space of
sublattice-resolved probe states). The 29% figure refers to the
second, which is now stated.

### 9. Polarization splitting is not independent

The referee is correct and we now say so in the text, in the conclusion
and in Appendix B: the "polarization splitting" at $L$ and the
"roton-like softening" at $L$ are the same fact, since the softened
level is the lower of the two transverse branches and the splitting is
the size of the softening. It is reported because it makes contact with
the earlier semiclassical predictions, not as separate evidence.

### 10. Off-pole weight is not a thermodynamic continuum

Agreed; the caution is retained and sharpened. A finite cluster
produces discrete poles where the thermodynamic system may produce a
peak plus a continuum, and the robust statement is the redistribution
of weight rather than the pole count.

### 11. Reproducibility

All dependence on the unpublished companion manuscript has been
removed. The one place where it was load-bearing --- the exclusion of
torus-winding six-cycles --- is now justified inline, with the counts
for each cluster. A numerical methods appendix has been added
describing the enumeration, the sparse construction, the winding-sector
identification, the diagonalization at each size, and the three
standing gates. The complete set of scripts is included with the
submission.

### 12. Length and venue

We have followed the referee's suggestion and resubmit to Physical
Review B, where the additional definitions, appendices and checks can
be presented properly.

### 13. Smaller points

* **Is the 3/4 reduction exact or numerical?** Both, in different
  senses, and we now distinguish them. Resolved by hexagon family, the
  geometric factor at $L$ is $(96,96,96,0)$ against $(96,96,96,96)$ at
  $X$, so the reduction of the *geometric* factor is exactly 3/4. The
  reduction of $f_1$ itself is 0.777, because $f_1$ weights each
  plaquette by its own ring coherence, which is not uniform on an
  anisotropic cluster. We also now report that this geometric
  decoupling occurs at *every* $[111]$ momentum, including the smallest,
  where there is no anomaly --- so it is necessary but not sufficient,
  which strengthens rather than weakens the mechanism argument.
* **Error bars on the classical comparison.** Added, and supplemented
  by a stronger control that needs none. Exactly, on the same cluster,
  the equal-amplitude state over the same 9104 configurations gives
  $S(L)=0.297$ against the interacting $0.370$; measured as enhancement
  over the other momenta, 11% classical against 48% quantum. The
  2048-site Monte Carlo, binned into 30 blocks, gives $S(L)=0.2488(67)$,
  $S(X)=0.2538(73)$, $S(W)=0.2475(52)$, $S(K)=0.2467(46)$: a spread of
  0.007 against a typical error of 0.006, i.e. flat within resolution.
* **Sign convention for $V$.** Now stated once, explicitly: $V>0$
  penalizes flippable plaquettes, $V<0$ rewards them, the RK point is
  $V=+K$, scans run over $V/K \in [-0.5,+0.5]$, and the anomalous level
  softens toward negative $V$.
* **"Lowest eigenvalue" wording.** Corrected to make clear it is the
  lowest excitation within the diagonalized winding sector, not a claim
  about all sectors or the full spin model.
