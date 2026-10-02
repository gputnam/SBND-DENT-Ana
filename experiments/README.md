# experiments/

Ancillary and **deprecated** code kept for reference. These are **not**
maintained, **not** covered by the test suite, and not part of the core
analyses. They may reference data paths or methods that have been superseded.

To run one, work from the repo root so the top-level core modules and `crtlib`
resolve (the scripts insert the repo root onto `sys.path`); some also need the
optional extras (`pip install jax jaxlib` for the CRT optimizer).

| Subfolder | Contents |
|---|---|
| `crt_alignment/` | **Deprecated** CRT-to-TPC alignment path: JAX rigid-body optimizer (`align_crt.py`), bootstrap wall/deformation fits, the 3-D space-point df builders, `crtdisp` voxelizers, residual/centroid comparisons. The CRT-only `crtlib` helper modules live in `crt_alignment/crtlib_legacy/`. |
| `dx_closure/` | dx-bias forensics: inversion/regularization studies, the intervention campaign, toy closure, truth-hit A/B/C builders, angle–hit-time-bias analysis, 2D-vs-3D comparison. |
| `generators/` | Anode-piercing muon generators (CORSIKA / SingleGen) + their fcl. |
| `diagnostics/` | Per-pixel cathode/dt diagnostics, track/hit counters, filelist builder, GIF annotation, misc one-off plots. |

The provenance of these studies is summarized in `../docs/history/CHANGELOG.md`.
