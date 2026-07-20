# Active campaign

The cubic-16 exact-band calculation and the cubic/FCC topology audit are frozen
method validation.  The remaining production campaign has two outputs:

1. a Figure 1 built entirely from converged FCC-32 exact-band XYZ results;
2. a Ce2Hf2O7 heat-capacity fit over converged FCC-32 model curves.

`fcc32_xyz_manifest.json` defines the Hamiltonian, grids, convergence gates,
and Figure 1 products.  The microscopic builder and cubic-16 driver now include
`Jpmpm=(Jb-Jc)/4` on the full spin basis because pair flips break the conserved
spin component.  FCC-32 has a 2,970-dimensional ice reference space, but the full
32-spin microscopic Hilbert space is much larger; legacy order-three matrices
on the ice space are topology diagnostics and are not production inputs.

`material_fit_config.json` records the data source, parameter domain, published
seeds, and model-curve location.  A valid curve in
`outputs/fcc32_xyz_grid/*.npz` must contain:

```text
method = "fcc32_exact_winding_free_xyz"
n_sites = 32
Ja_meV, Jb_meV, Jc_meV
temperature_K, heat_capacity_J_molCe_K
character_M, character_converged, complement_converged
```

Run `make material-fit` to rank available curves.  The driver rejects wrong
methods, clusters, and unconverged character or complement calculations.  The
current Smith et al. curve is a reproducible v5 raster extraction without
tabulated uncertainties, so its result is explicitly exploratory until author
data are supplied.

The earlier `run_validation.py`, `run_nonperturbative.py`, and `run_dssf.py`
remain the reproducible cubic validation path.  They do not produce an FCC-32
material fit.
