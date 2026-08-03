# Winding-free quantum spin ice campaign

This repository develops and validates a finite-cluster protocol that removes
the noncontractible (winding) ring exchanges a periodic cluster invents from
the gauge physics of quantum spin ice (QSI).

## The method, current form

The optimized protocol is a projection, not an average.  Every ice
configuration carries a transport (polarization) label; a winding process
changes it by exactly half a box vector (`rho = w.L/2`), a contractible
process cannot change it at all.  The winding-free theory is therefore
obtained by **deleting every ice-block matrix element that connects
different transport sectors** — applied to the exact Feshbach map
`F(z) = PHP + PHQ (z - QHQ)^{-1} QHP`, because the bare ice block is
diagonal (both the physical hexagon and the winding artifact are emergent),
so the deletion must act on the folded map where those amplitudes become
explicit.  Twist/character averaging over the ice manifold is the historical
*derivation*, not the algorithm: the `M=2` character (corner) average equals
the parity mask exactly (verified entry by entry, and eight times cheaper),
and the continuous source average equals the full transport-sector block
projection imposed directly here at any coupling.  Full-range thermodynamics
comes from the two-sided fold (remove the highest-ice-weight raw levels,
insert the masked self-consistent roots); band spectra on clusters too large
for dense complements come from the multi-anchor interpolated
self-consistency and the per-sector bordered matrices.

The complete narrative — every construction, calibration gate, and recorded
dead end — is `paper/pedagogical.tex`.  Campaign status and the
remote-cluster phase goals are at the top of `SIMULATION_PLAN.md`.

Whether the winding-free band approaches bulk QSI remains a cluster-size
question that must pass order, cluster-size, and sign-free QMC benchmarks.

## The story

Periodic clusters invent a spurious four-site ring exchange: an
ice-preserving loop that closes only through the boundary, entering at
order `Jpm^2/Jzz` — *below* the physical six-site ring exchange at
`12|Jpm|^3/Jzz^2`.  Every naive finite-size ED of quantum spin ice
therefore measures the box, not the photon.  The central goal of this
repository is an ED technique that removes that artifact exactly, without
introducing new physics or destroying real physics.  Three chapters:

1. **Twist averaging over the ice manifold — tried, and provably
   insufficient.**  Phases attach to virtual routes, not loop topology; no
   observable-level twist scheme beats a surviving-weight floor of 1/3, and
   the measured peak still scales with the artifact power.  Retained as the
   falsification record (`campaign/twist_averaging.tex`,
   `outputs/twist_average_curves_m*.npz`).
2. **Polar isometry / character projection, equivalent to the parity
   mask — the current best method.**  The `M=2` character average of the
   polar-pullback band operator equals, entry for entry, the transport
   parity mask; the continuous source average equals the full
   transport-sector block projection.  In optimized form (see "The method"
   above) this is a deletion on the Feshbach map, exact at any coupling,
   and its full-spectrum embedding is the in-block counterterm `H + C`
   (equivalently the splice / the two-sided fold for thermodynamics).
3. **The coupling scans, tracking every peak.**  The `Jpm` line on
   cubic-16 and FCC-32 (winding-free ring branch, third-peak taxonomy,
   host coefficients `c16 = 1.070`, `c32 = 0.521`), and the
   `(Jpm, Jpmpm)` plane on cubic-16 (certified ratio ceiling ~0.10, one
   certified gauge-spinon merge cell, re-entrant dissolution boundary).
   The FCC-32 plane is the remote-cluster phase (`SIMULATION_PLAN.md`).

## Repository map

- `paper/` — `main.tex` (the paper) and `pedagogical.tex` (the complete
  narrative: every construction, calibration gate, and recorded dead end).
- `SIMULATION_PLAN.md` — campaign status, remote-phase goals, and the
  frozen microscopic protocol definition.
- `src/qsi_campaign/` — the library: downfolding (`downfold.py`), mask and
  band primitives (`protocol.py`, `exact_band.py`), materials
  (`material_scan.py`), space-group reduction (`point_group.py`).
- `campaign/` — one standalone script per instrument or study; the current
  production stack is listed in `campaign/README.md`, and
  `campaign/outputs/README.md` indexes every retained data directory with
  its producer and consumers.
- `notes/recompute_finite_size_artifact.py` — the cluster geometry builder
  every campaign script imports.
- `tests/` — unit tests for the library.

Removed material (legacy trees, frame-era sweeps, dead-end counterterm
catalogues, superseded twist caches) is recoverable from git history; the
conclusions live in `paper/pedagogical.tex`.

## Workflow

```bash
python -m pip install -e .
make all        # regenerates the frozen validation figures
```

Campaign scanners run standalone, one json+npz per parameter point,
parallel by grid point, with `OMP_NUM_THREADS`/`OPENBLAS_NUM_THREADS`
pinned (8 per job is calibrated); see `SIMULATION_PLAN.md` for measured
per-point costs.

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
