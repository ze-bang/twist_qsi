# Manuscripts

| Folder | Manuscript |
|---|---|
| `paper/` | *Removing Torus Artifacts Reveals Nonperturbative Loop Locality in Quantum Spin Ice* — the method/companion paper. |
| `paper_loop_renormalization/` | *A single renormalized energy denominator controls the nonperturbative loop expansion of quantum spin ice* — the physics paper. |
| `archive_superseded_2026-08-17/` | Three exploratory drafts, replaced. One (emergent permittivity) is a recorded falsification. |

## The physics paper in one paragraph

Perturbation theory gives every loop of perimeter `L` an amplitude
`K_L^(0) = 2 C(L-2, L/2-1) |Jpm|^(L/2) / Jzz^(L/2-1)`, built from `L/2-1`
energy denominators each set to `Jzz`. Measured nonperturbatively on
artifact-free clusters, the amplitudes obey `K_L = K_L^(0) mu^(L/2-1)` with a
single `mu` per coupling (spread 1-3.5% over L = 8, 10, 12, two virtual
depths): every denominator is renormalized `Jzz -> D_eff = Jzz/mu`. Two exact
sum rules — the second-order amplitude vanishes, and the first spectral moment
is exactly `12|Jpm|^3` — make `D_eff = sqrt(M_1/K_6)` a definition that equals
`Jzz` in leading perturbation theory, and it is intensive (16 vs 32 sites agree
to <10%). Measured `D_eff/Jzz = 1.17 -> 3.35` over `|Jpm|/Jzz = 0.03 -> 0.30`,
so the hexagon coupling is suppressed by `mu^2` — 4.3x at Ce couplings, 11x at
the strongest. Hence: the emergent photon is several times colder than the
standard estimate; the two specific-heat peaks cannot merge (the `|Jpm|^3`
growth that would merge them is cancelled by `mu^2`, and the merging seen in
both perturbation theory and naive periodic ED is an artifact); and in the Ce
pyrochlores the gauge feature sits at a few mK, below every temperature ever
measured, so a single broad hump is the expected signature. The decisive
experiment is the entropy below the observed hump, which must be
`(R/2)ln(3/2)`.

## Build

    sbatch paper_loop_renormalization/build.sbatch
