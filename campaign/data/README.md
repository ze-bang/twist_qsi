# External QMC benchmark provenance

`huang_2018_qmc_jpm_0p046.csv` is a vector extraction of the heat capacity
and entropy curves in Fig. 1(b) of:

C.-J. Huang, Y. Deng, Y. Wan, and Z. Y. Meng, *Dynamics of Topological
Excitations in a Model Quantum Spin Ice*, Phys. Rev. Lett. **120**, 167202
(2018), DOI: 10.1103/PhysRevLett.120.167202, arXiv:1707.00099.

This is the preferred external benchmark because the paper uses exactly the
repository convention

```text
H = -Jpm sum(S+S- + h.c.) + Jz sum SzSz
```

at `Jpm/Jz = 0.046`.  The thermodynamic simulation is described as a
continuous-time worm QMC calculation.  The same paper reports QMC-SAC dynamic
structure factors on `8 x 8 x 8` primitive cells at `T/Jz = 0.001, 0.04,
0.1`.

The arXiv source archive contains `fig1.pdf`.  Convert it to SVG and run:

```bash
pdftocairo -svg fig1.pdf fig1.svg
python extract_huang2018.py fig1.svg
```

The extractor reads the red heat-capacity paths and orange entropy paths
directly from the vector figure.  Axis calibrations use the vector tick
positions.  Estimated extraction uncertainties are `0.002` in `log10(T/Jz)`,
`0.0015` in `C/N`, and `0.002` in `S/N`.  These cover line width, axis
calibration, and interpolation of entropy onto the heat-capacity temperature
grid.  They do not replace QMC statistical errors, which are not tabulated in
the figure.

The copyrighted source figure is not redistributed.  The extracted curve is
used only for numerical verification and should be replaced by author data if
available.

# Ce2Hf2O7 material-fit input

`ce2hf2o7_smith2025_digitized.csv` contains the experimental and published
NLC-A traces extracted from Fig. 2(a) of Smith et al., *Phys. Rev. Lett.*
**135**, 086702 (2025), arXiv:2501.08327v5.  The source archive was downloaded
from arXiv and had SHA256
`e536b26af0088f6de2e5047bca18410a213b89c026b7f5d6f8c6fb3a8aedcedd`;
its `Figure2.png` had SHA256
`cc2f9ea660b8c26320557734d337c7513be6073be95143750412e0d6c87646a9`.

Regenerate the extraction with:

```bash
python campaign/data/extract_smith2025.py /path/to/Figure2.png
```

The raster does not provide the experimental uncertainties used in the
published NLC objective.  Active code may use these points for exploratory
curve ranking, but a submission-quality fit requires tabulated author data
with uncertainties.  Published parameter seeds are A = `(0.050, 0.021,
0.004)` meV, B = `(0.051, 0.008, -0.018)` meV, and the ordered-regime QMC
fit = `(0.046, -0.003, -0.010)` meV.
