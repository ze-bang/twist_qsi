# Paper C — Emergent electrostatics: RESULT IS A NULL (measured 2026-08-17)

> **STATUS: the central premise is falsified by its own test.** The quadratic
> law `dE(T) = |T|^2 / (2 eps_eff L)` fails at every coupling, cluster and
> virtual depth measured — **29 of 29 fits**, best max-relative-residual
> **0.215**, worst **0.441**. There is no weak-field window on either cluster in
> which `eps_eff` is defined, so `c` and `alpha` cannot be extracted from these
> hosts. `main.tex` still contains the draft as originally written, with a red
> banner in the abstract and a new Sec. "Falsification" reporting the measurement.
> **Decide the reframing before doing anything else with this draft.**
>
> Run: job `19155241`, `analysis/cubic16_sector_check.py`,
> log `analysis/cubic16_19155241.log`, data `analysis/cubic16_sector_check.json`.

## What the measurement found

**Transport labels (the join that was pending).**

| cluster | distinct \|T\|^2 | multiplicities | ice dims |
|---|---|---|---|
| cubic-16 | 0, 1, 2, 4 | 1, 6, 12, 6 | 12, 8, 2, 1 |
| FCC-32 | 0, 2, 4, 6, 8, 10, 16 | 1, 12, 6, 24, 12, 24, 6 | 396, 140, 72, 12, 6, 4, 1 |

This confirms the shell table printed in the draft and identifies FCC-32's
396-dimensional bottom sector (index 42) as `T = 0`.

**1. Quadraticity fails.** FCC-32, Jpm=-0.05, depth 1: the law needs
`dE(|T|^2=4) = 2 dE(|T|^2=2)`; measured ratio is **1.38**. Then `|T|^2` 4 -> 6
jumps by 3.5x and is flat through 8, 10, 16. The energies split into two
families — large sectors that still resonate (dim 396/140/72) and small ones
that are nearly frozen (dim 12/6/4/1) — with a step between. That is a lattice
ring-exchange model in its strong-field, discrete regime, not a Maxwell field.

**2. The isotropy test is vacuous on these clusters.** Every `|T|^2` value
corresponds to exactly one cubic point-group star; no two distinct stars share a
modulus. The exact within-shell degeneracies (worst spread 6.0e-14 over the
whole run) are therefore guaranteed by the space group and test nothing. This
was to have been the sharpest structural check and it is unavailable.

**3. Zero flux is not always the ground.** On cubic-16 the **exact** (levels=4)
ground sector is `|T|^2 = 1`, six-fold degenerate, at Jpm = -0.05 and -0.20; it
moves to `T = 0` only by -0.30. This is the long-standing "six degenerate ground
sectors [3,8,11,13,16,21]" of the campaign — now identified as a **flux-polarized
ground state**, not a resonance degeneracy. FCC-32 has its ground at `T = 0` at
every coupling and depth. The two clusters disagree qualitatively.

**4. Depth truncation is severe, and FCC-32 runs at the worst depth.**
On cubic-16 at Jpm=-0.30 the *identity of the ground sector* is a truncation
artifact (levels=1 says `|T|^2=1`; levels>=2 and exact say `T=0`). On FCC-32
levels 1 -> 2 at Jpm=-0.05 moves the first three shells by +34%, +31%, +21%, and
the ordering of the `|T|^2` = 6 and 8 shells **inverts**. All thirteen FCC-32
couplings are at levels=1.

## Three ways to reframe (pick one)

1. **The null paper.** "How small is too small for emergent electrodynamics?"
   Reports items 1-4 as the result. Genuinely useful to everyone doing QSI ED —
   it says which finite-cluster gauge observables are meaningful and which are
   not, and it kills the flux-sector route to `eps` on any cluster of this size.
   Cheapest: the data is in hand.
2. **Fold into Paper A or the canonical methods paper** as a section on
   finite-cluster limits of the gauge-sector interpretation. Loses the standalone
   result but strengthens a paper that is already sound.
3. **Push to a bigger host and retry.** The builder supports FCC (2,2,3) = 48
   sites; the weak-field window scales as the number of cells across, so a 2x2x3
   box helps only in one direction. This is the expensive option and, given that
   the small-`|T|` shells already fail by 21-26% at *both* existing sizes, is
   unlikely to change the answer without a much larger cluster.

My reading: option 1, with items 3 and 4 as the headline (a truncation-dependent
ground *sector*, and a flux-polarized exact ground state on cubic-16, are both
new and both matter to anyone reading finite-cluster QSI results).

## What remains LIVE and reusable

| Number | Source |
|---|---|
| 85 / 25 sector ground energies, all depths | `fcc32_sector_grounds/`, `analysis/cubic16_sector_check.json` |
| transport-label join, both clusters | this run |
| within-shell degeneracy 6.0e-14 (exact, but see item 2) | this run |
| depth drift +34/+31/+21% (FCC-32, levels 1->2) | this run |
| 't Hooft splitting x0.65 (16->32) at -0.05, x3.5 / x2.8 at -0.22 / -0.30 | `tier1_probes/README` |
| omega(Gamma/X/L) at +0.046 and -0.200 | `run_momentum_resolved_spectrum.py` |

## Reproduce

    sbatch analysis/run_cubic16_check.sbatch    # ~40 s of compute, whole node
    sbatch build.sbatch                         # rebuild the PDF

Note: `analysis/run_sector_labels.sbatch` (the FCC-32-only join) is now
subsumed by `cubic16_sector_check.py`, which does both clusters.
