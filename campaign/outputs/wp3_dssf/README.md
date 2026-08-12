# WP3 sector-resolved longitudinal response — state-count hazard

**Do not mix files with different excited-state counts in one figure.**

The number of excited states summed per sector (`--nstates`) controls how much
of the inelastic weight is captured at each momentum.  At 24 states the sum is
a partial sum that captures different fractions at different momenta, and at
weak coupling — where the band is dense on the scale of the coupling — it can
return machine-zero weight at momenta that in fact carry weight.

That is exactly what happened.  The 24-state run reported the three `X` points
dark at `~1e-24`, which was read as a selection rule forbidding half the
Brillouin zone.  **That claim is retracted.**  A converged 120-state
recomputation at `Jpm/Jzz = -0.05` gives finite weight at every nonzero
momentum: `0.3093` at the four `L` points and `0.2378` at the three `X` points.
Only `Gamma` is forbidden, at `1.4e-30`, and that one is a theorem — the
transport label is in bijection with the `q=0` sublattice magnetisations, so
`P S^z_{q=0} P` is a c-number on every mask block.

## File status

| file | states | status |
|---|---|---|
| `dssf_fcc32_L1_-0p050_pp+0p000.json` | 120 | current |
| `dssf_fcc32_L1_-0p200_pp+0p000.json` | 120 | current |
| `STALE_24state_...-0p200_....json.quarantine` | 24 | **quarantined, do not plot** |
| `dssf_cubic16_L3_*.json` | 24 | gate runs only |

The quarantined file is retained rather than deleted because it is the
provenance of a retraction.  It is renamed out of the `dssf_*.json` pattern so
that a glob cannot pick it up.  Its 120-state replacement at
`Jpm/Jzz = -0.20` was completed by job `18871563` and records
`nstates_requested = 120` explicitly.

The two `dssf_cubic16_L3_*.json` files are also 24-state.  They are used only
for the `Gamma` suppression gate, which is insensitive to the state count
because the `Gamma` zero is exact; their nonzero-momentum weights should not be
plotted against FCC-32's 120-state values.

## The surviving momentum-space statement

At `Jpm/Jzz = -0.05`, 120 states, the two nonzero-momentum groups carry
`0.3093` and `0.2378`; at `-0.20` they carry `0.2988` and `0.2141`. These are
measurements on one cluster at `levels=1`, with only eight available momenta.
The anisotropy is not established as a flux-sector signature and should not be
presented as one.
