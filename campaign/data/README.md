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
