# Paper A — Photon-number parity in quantum spin ice

**One-line thesis.** The global pseudospin flip `P` is the *charge conjugation*
of the emergent U(1) gauge theory, so photon-number parity is exactly conserved
at h=0; neutrons and Raman/ultrasound therefore measure **disjoint** sets of
gauge excitations, and a field switches the forbidden channel on as `h^2`.

Why this is the most experimentalist-facing of the three: every claim is exact
and cluster-independent, which is precisely what the rest of the campaign is
not. The FCC-32 spectrum is used only as a demonstration, never as a prediction.

## Status of every number in the draft

### LIVE (already computed, traceable)

| Number | Source |
|---|---|
| E/cosB complementarity table, both couplings, complement 1e-25..1e-35 | `campaign/run_flux_diagnostic.py`, reproduced in `paper/parity_section.tex` |
| momentum projector exact to 4e-13 | `campaign/run_momentum_resolved_spectrum.py` |
| 85 sectors <-> 85 magnetization tuples, bijection; within-sector spread exactly 0; commutator 0 vs unmasked control 55.9 | WP0b, `SIMULATION_PLAN.md` |
| q=0 masked 1.42e-29 vs periodic 0.3126 (Jpm=-0.20); 1.36e-30 vs 0.4879 (-0.05) | `paper/DATA_PROVENANCE.md`, Fig. 2 |
| 32 contractible of 128 six-cycles; out-of-block weight 0.0 with the filter, 10-28% without | `campaign/check_ring_exchange_closure.py` |
| omega(X) 1.5-2.2x, omega(Gamma) 3.5-6.1x cubic-16 -> FCC-32, ordering inverts | `campaign/run_momentum_resolved_spectrum.py` |
| [111] field sector staircase 25->19->16->9; naive delays crossover 2-3x | `campaign/outputs/tier1_probes/tier1_field_-0p050.json` |
| g4 even / g6 odd in Jpm; raw flux sign inverted for all Jpm<0 | `paper/flux_methods.tex` |

### PENDING (marked `\pending{}` in red in the PDF — do not remove without running)

1. **`Phi_-` (sin B) channel.** The whole positive-assignment section rests on it.
   Cheapest possible run: `campaign/run_flux_diagnostic.py` already builds
   `Phi(q) = sum_h e^{-iq.r_h} (W_h + W_h^dagger)` over the 32 contractible
   hexagons. Add the conjugate combination `i(W_h - W_h^dagger)` and emit a third
   weight column. Same eigenvectors, same masked block — no new diagonalization.
   Expected: exact complementarity with the S^zz-bright set.
2. **`Jpmpm != 0` spot check** at `(Jpm, Jpmpm)/Jzz = (-0.125, 0.085)`. The proof
   covers it (the pair-flip term maps to its h.c. under P), but S^z_tot is not
   conserved there, so the basis and the sector labels change. One point.
3. **`h^2` onset.** No solver in the repo has a Zeeman term wired into the
   Hamiltonian — `protocol.zeeman_band_term` is a diagnostic on the ice reference
   basis only. This is the largest of the three tasks: add `-h sum_i S^z_i` to
   `run_truncated_feshbach.build_hamiltonian` and recompute the two channels at
   a few small `h`. The `tier1_probes` field staircase shows the machinery
   partially exists on cubic-16.

### CLAIMED AS HYPOTHESIS, NOT RESULT

The two-photon bound state (Sec. "The missing probe"). The 19.4% / 18.5% deficit
across a 4.3x coupling change is real arithmetic on live numbers, but the
interpretation is blocked by `omega_odd(X)/omega(L) = 2.42, 2.04` against the
linear-dispersion value 1.155. The draft says so explicitly. Do not promote this
to the abstract without item 1 above plus a second cluster.

## Known weak points a referee will attack

- **"P is just the standard Z2 of the XYZ model."** True, and the draft says so.
  The novelty is the identification `P = C` of the emergent gauge field and the
  probe-complementarity table, not the symmetry itself. Keep that framing.
- **"Octupolar QSI photon being neutron-dark is known."** Partly — it is usually
  argued from the form factor / which pseudospin component the neutron couples
  to. The parity framing adds the *positive* statement (Raman/ultrasound sees the
  2-photon continuum, with the onset fixing the photon gap), which is the useful part.
- **Nothing is size-converged.** Handled by keeping every energy out of the claims.

## Build

    sbatch build.sbatch      # pdflatex / bibtex / pdflatex x2

`refs.bib` was drafted partly from memory — **verify every entry** before
submission, especially `SimonPatriKim2022`, `Desrochers2023`, `Poree2025`.
