# Validation plan: a convergent projected-band protocol

## Scientific question

Periodic QSI clusters contain ice-preserving paths that close only through a
periodic image.  On cubic-16 the shortest such process flips four spins and
appears at order `Jpm^2/Jzz`, before the physical contractible six-spin ring
exchange at order `|Jpm|^3/Jzz^2`.  The task is to remove the former without
asserting that a low-order effective Hamiltonian is the microscopic model.

## Definition

Write `H(lambda) = H0 + lambda V` and let `P` project onto the ice manifold.
The canonical Schrieffer-Wolff/Kato gauge chooses an off-diagonal,
anti-Hermitian generator

```text
S_N(lambda) = sum_{n=1}^N lambda^n S_n,
P S_n P = Q S_n Q = 0,
Q exp(-S_N) H exp(S_N) P = O(lambda^(N+1)).
```

This convention removes the unitary ambiguity inside `P`.  The two blocks are

```text
H_P^(N) = P exp(-S_N) H exp(S_N) P,
H_Q^(N) = Q exp(-S_N) H exp(S_N) Q.
```

Resolve every completed row of `H_P^(N)` by its integer transported dipole
`q in Z^3`.  The `M^3` character average

```text
Pi_M[H_P] = M^-3 sum_{m in Z_M^3} H_P(theta_m),
theta_m = 2 pi m / M,
```

keeps `q = 0 mod M`.  At fixed order, choosing `M` above the largest possible
transport gives the exact zero-transport operator.  Otherwise the residual is
measured by the `M -> 2M` ladder.  The finite-order cleaned Hamiltonian is

```text
H_clean^(N,M) = exp(S_N) [Pi_M H_P^(N) + H_Q^(N)] exp(-S_N).
```

No counterterm is fitted.  No thermal observable is averaged over boundary
conditions.  Both operations were tested and rejected because they do not
project completed virtual paths.

## Exact-band limit

If an isolated band of dimension `dim(P)` exists, let `P_lambda` be its
spectral projector.  Kato parallel transport, equivalently the polar map of
`P P_lambda P`, defines a canonical unitary from `P` to `P_lambda`.  Pulling
the exact band back with this unitary resums the canonical series as
`N -> infinity`.  Loss of invertibility of `P P_lambda P` is a failed
isolated-band gate, not a reason to extrapolate the protocol.

For full-temperature observables the complement is retained exactly:

```text
spec(H_clean) = spec(Pi_M H_band) union spec(H outside the selected band).
```

Equivalently, partition-function moments of the bare band are subtracted from
the microscopic trace and clean-band moments are added.  This defines one
temperature-independent Hermitian Hamiltonian and preserves the spinon and
high-temperature sectors.

## Required gates

| Gate | Quantity | Pass criterion | Current status |
|---|---|---|---|
| topology | row-resolved transport | all winding four-loops removed; all contractible hexagons retained | passed at orders 2/3 on cubic-16 and FCC-32 |
| order | low-order rows reproduce exact-band weak-coupling mechanism | qualitative/topological agreement | order 2/3 is diagnostic only; no production `N` limit required |
| character | successive `M` centered operators and spectra | change below 5% | passed on exact cubic-16: `M=3 -> 4` is 0.498% |
| band | minimum ice overlap and separation from complement | nonsingular pullback and stable band identity | passed in fixed `Sz=0`; minimum overlap 0.762 at `Jpm/Jzz=0.046` |
| all temperature | high peak and entropy after exact-band replacement | stable high-temperature complement | high peak stable; entropy comparison fails |
| cluster | cubic-16 vs FCC-32 | controlled drift toward bulk | order-three low peak still drifts strongly |
| external | zero-flux QMC `C`, `S`, dynamics | cleaner approaches QMC and meets fixed tolerances | exact `M=4` improves heat RMSE 72.9% but heat and entropy both fail; dynamics pending |

Only observables whose gates pass can enter a physics claim.  FCC-32
material fitting and dynamics require full-Hamiltonian low-band extraction,
`N/M` convergence, and the zero-flux QMC gate first.

## Campaign stages

1. **Analytic/topological unit tests.** Enumerate loops and verify the
   zero-transport row selector on both clusters.
2. **Weak-coupling mechanism check.** Compare order 2/3 rows with the exact
   pulled-back band; do not use the truncated block for production curves.
3. **Character convergence.** Run matched exact-band grids with the primitive
   `q=2 delta` source.  The active `M=3,4` cubic-16 sequence passes at 0.498%.
4. **All-temperature check.** Combine the clean low band with exact ED on
   cubic-16 and FTLM/SLQ on larger clusters.
5. **Zero-flux benchmark.** At `Jpm/Jzz=0.046`, compare `C` and `S` with the
   vector thermodynamic curves of Huang et al.  Then match their QMC-SAC
   `Szz` and `S+-` at `T/Jzz=0.001, 0.04, 0.1` on the momentum paths actually
   represented by each finite cluster.
6. **FCC-32 deployment.** Compute the exact isolated band and stochastic
   complement.  Order-three ice-manifold data alone are not a deployment.
7. **Pi-flux/material application.** Only after stages 1-6, fit static and
   dynamic observables.  Ce2Hf2O7 is motivation, not a current result.

## Reproducibility

`campaign/run_nonperturbative.py --max-grid 4` writes the exact pulled-back
operators, thermodynamics, and gate report to `campaign/outputs/`; individual
source points are restartable under `campaign/cache/nonperturbative_points/`.
`campaign/run_validation.py` retains the order-three topology diagnostics.
`campaign/make_figures.py` reads only those products.  `legacy/` contains
rejected approaches and preliminary claims; active code never imports it.
