# Winding-free quantum spin ice campaign

This repository tests a finite-cluster protocol for removing noncontractible
ring exchanges from the gauge band of quantum spin ice (QSI).  The scientific
claim is deliberately narrower than earlier drafts: winding-free character
projection removes winding channels from a chosen, isolated band.  Whether
the resulting band approaches bulk QSI is a separate numerical question that
must pass order, character-grid, cluster-size, and sign-free QMC benchmarks.

## Active workflow

```bash
python -m pip install -e .
make all
```

The workflow currently establishes:

- exact removal of the second-order winding four-loop and retention of the
  third-order contractible hexagon on cubic-16 and FCC-32;
- an all-temperature construction that replaces only the winding-free gauge band
  while retaining the exact microscopic complement on cubic-16;
- a zero-temperature, four-sublattice-traced `Szz` comparison between the
  periodic and winding-free cubic-16 bands at all four allowed momenta;
- a zero-flux comparison at `(Jpm, Jpmpm)/Jzz = (0.046, 0)` to the heat-capacity and entropy
  curves of Huang, Deng, Wan, and Meng;
- explicit status gates for perturbative order `N`, character resolution `M`,
  isolated-band overlap, full-temperature stability, and cluster size.

The exact cubic-16 band is converged from `M=3` to `M=4` to 0.50% in centered
operator norm.  It improves the Huang heat-capacity comparison substantially,
but retains a finite-size discrepancy.  The remaining production campaign is
therefore focused on a fully FCC-32 Figure 1 and a Ce2Hf2O7 fit using the full
ABC/XYZ model.  The active fit code will not accept legacy order-three
ice-space curves as nonperturbative FCC-32 results.

## Nonperturbative production path

The intended production method is now the exact microscopic twisted-band
calculation, not the order-three block.  It constructs the ice-rule projector
`P_ice`, diagonalizes the explicitly sourced cubic-16 Hamiltonian, identifies
the isolated band continuously connected to `Ran(P_ice)`, and pulls that band
to the fixed ice reference space with the polar map.  It then averages the
operator over one common primitive `q = 2 delta` character grid.  Run the fast
`M=2` baseline with

```bash
make nonperturbative
```

The driver accepts `--jpmpm`; a nonzero value automatically selects the full
65,536-state cubic-16 basis and writes a coupling-qualified output.  The frozen
QMC thermodynamics and DSSF products remain the `Jpmpm=0` benchmark.

The converged cached `M=2,3,4` campaign is generated with

```bash
make nonperturbative-m4
```

This calculation is intentionally separate from `make all` because it is
substantially more expensive.  The order-three workflow remains a topology
unit test and perturbative reference only.

## Layout

- `src/qsi_campaign/`: reusable protocol and thermodynamics code.
- `campaign/`: one configuration, runner, data provenance, and generated
  validation products.
- `paper/`: PRL-length manuscript and Supplemental Material.
- `docs/projected_band_protocol_note.tex`: pedagogical, code-level derivation
  of the exact protocol and its perturbative topology diagnostic.
- `notes/`: retained geometry and full-band solvers used by the campaign.
- `legacy/`: superseded counterterm, masked-BW, and preliminary submission
  campaigns.  These are negative controls and historical records only.

See `SIMULATION_PLAN.md` for the concise mathematical definition and
pass/fail gates.  Run `make note` to build the standalone pedagogical guide.

The FCC-32 material stage is specified by
`campaign/fcc32_xyz_manifest.json`; the Ce2Hf2O7 curve contract and fit are in
`campaign/material_fit_config.json` and `campaign/run_ce2hf2o7_fit.py`.
