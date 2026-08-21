# Superseded drafts (archived 2026-08-17, not deleted)

Three exploratory manuscripts, all replaced by `paper_loop_renormalization/`.
Kept because each contains reusable material and one recorded negative result.

- `paper_A_selection_rules/` — photon-number parity / probe complementarity.
  **Why dropped:** largely the standard one-magnon/two-magnon Raman selection
  rule transposed to the emergent photon, plus a q=0 theorem already derived in
  `2dcs_gauge` stage14. Reusable: the `sin B` probe observation (the linear
  magnetic channel `i(W - W^dagger)` is P-odd, unlike `cos B`), which remains
  unmeasured and would allow a positive photon assignment.
- `paper_B_material_verdict/` — the Ce2Hf2O7 25 mK null. **Why dropped:**
  absorbed as Sec. "The cerium pyrochlores" of the new paper, where it is a
  corollary of the renormalization rather than a standalone null. The entropy
  script `analysis/entropy_budget.py` is still the right instrument and is
  referenced by the new manuscript.
- `paper_C_emergent_maxwell/` — emergent permittivity from electric-flux
  sectors. **FALSIFIED by its own test** (job 19155241): the quadratic law
  fails 29/29 fits, best residual 0.215. See its README for the full null,
  including two findings worth keeping — the ground *sector* is virtual-depth
  dependent, and cubic-16's exact ground state is flux-polarized.
