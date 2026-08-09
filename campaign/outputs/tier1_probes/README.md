# Tier-1 probes: exact winding-free observables vs raw ED

Everything here comes from one instrument: per transport sector, the bordered
Hermitian operator `M_s = Pi_s H Pi_s` on `span(P_s) + Q` (whose ice-descended
eigenvalues equal the masked self-consistent roots to ~1e-13), sparse-solved
with eigenvectors.  Because `S^z` is diagonal in the Ising basis, every
longitudinal matrix element is exact — spinon dressing included — and
transitions are sector-diagonal.  Raw-ED panels use the lowest exact
eigenstates of the same `H` for contrast.

Points: `Jpm` = -0.05, -0.20, -0.30 (XXZ, Sz=0 block) and the Ce2Hf2O7 fit
point `(-0.125, 0.085)` (full even-parity space; every ice state has
`N_up = 8`, so the odd sector is irrelevant).

## Files

| file | contents |
|---|---|
| `tier1_<tag>.json/npz` | per-point results: sector bottoms, band levels + Klein momentum labels (G/Xx/Xy/Xz), ice weights, DSSF sticks at the four cluster momenta, q=0 sublattice rule weights, equal-time `<Sz_i Sz_j>` (multiplet-averaged, unsubtracted), hexagon-hexagon (Wilson) correlators, raw comparison |
| `tier1_field_-0p050.json` | [111] field sweep at `Jpm=-0.05`: raw ground line, winding-free band bottom, continuum edge, and the count of sectors still holding a root, vs `h` |
| `fig_tier1_sq_maps.*` | equal-time `S^zz(q)` in the (h,h,l) plane, winding-free vs raw, all four points |
| `fig_tier1_scales.*` | photon `w(X)` and spinon gap vs coupling; the photon/matter hierarchy ratio |
| `fig_tier1_rule.*` | the Gamma-point longitudinal selection rule, masked vs raw (log scale) |
| `fig_tier1_field.*` | field sweep energies + the field-driven sector staircase |
| `fig_tier1_thooft.*` | flux-sector splitting vs cluster size (cubic-16 exact, FCC-32 L=1/L=2) |

Producer: `campaign/make_tier1_figures.py` (figures) from data produced by the
session scripts `tier1_core.py` / `tier1_field.py` (copies staged here).

## Headline numbers

* Within-band longitudinal DSSF is dark (sticks <= 3e-5): emergent
  transversality.  The longitudinal photon signal is the *diffuse map*.
* q=0 rule: masked <= 9e-5 (weak coupling, dressing tail) down to 2e-27
  (CHO, machine zero) vs raw 0.26-0.60 — survives `Jpmpm` exactly.
* Pinch points: winding-free anisotropy at (002) is sharper than raw at every
  point, and the wf map *restructures* by -0.30 (weight moves to (110)-type
  spots) while raw barely changes — a falsifiable discriminator.
* Emergent hierarchy `w(X)/gap`: 0.006 (-0.05) -> 0.28 (-0.20); CHO sits at
  0.095 with `w(X) = 0.045 Jzz` (~26 mK for Ja=0.05 meV, host-coefficient
  caveat ~x2-3 downward).  At -0.30 the 6->2 restructured band has no photon
  reading -- its X states (0.234 above GS) are NOT a photon energy and the
  hierarchy point there is bandwidth-based (open marker in the figure).
* Consistency: `2.4*T_ring` from the band C(T) lands between the first
  cross-sector gap and `w(X)` (-0.05: 0.0030 vs 0.0028/0.0047) -- the C(T)
  peak samples the dense cross-sector gaps at the band bottom, the DSSF sees
  only what the probe couples to (nothing, longitudinally), and `w(X)` is one
  point of the dispersion.  Same spectrum, three different moments.
* 't Hooft: splitting *shrinks* 16->32 at -0.05 (ratio 0.65, deconfined
  signature) and *grows* at -0.22/-0.30 (3.5x/2.8x, L-truncation caveat) —
  the size-scaling changes sign between -0.05 and -0.22.
* [111] field: sector staircase 25->6; raw ED's artifact binding delays the
  low-field polarization crossover by a factor ~2-3 in `h`.

Caveats: one cluster except the 't Hooft panel; cubic-16 host coefficient
(~x3 on ring-scale quantities); the matched-winding-pair residue (19-42% on
peak positions) is not corrected in these observables; FCC-32 gaps at finite
L are lower-bound-converged (L1->L2 grows them).
