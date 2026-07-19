# Winding-clean quantum spin ice campaign

This repository tests a finite-cluster protocol for removing noncontractible
ring exchanges from the gauge band of quantum spin ice (QSI).  The scientific
claim is deliberately narrower than earlier drafts: transport-character
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
- an all-temperature construction that replaces only the cleaned gauge band
  while retaining the exact microscopic complement on cubic-16;
- a zero-flux comparison at `Jpm/Jzz = 0.046` to the heat-capacity and entropy
  curves of Huang, Deng, Wan, and Meng;
- explicit status gates for perturbative order `N`, character resolution `M`,
  isolated-band overlap, full-temperature stability, and cluster size.

The exact cubic-16 band is converged from `M=3` to `M=4` to 0.50% in centered
operator norm.  It improves the Huang heat-capacity comparison substantially,
but still fails the preregistered heat and entropy tolerances.  FCC-32
full-Hamiltonian and material fits are therefore future gates, not paper
claims.

## Nonperturbative production path

The intended production method is now the exact microscopic twisted-band
calculation, not the order-three block.  It diagonalizes the cubic-16
Hamiltonian in four translation sectors, pulls the lowest 90-state band into
the ice basis with the polar map, and averages that operator over one common
primitive `q = 2 delta` character grid.  Run the fast `M=2` baseline with

```bash
make nonperturbative
```

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
