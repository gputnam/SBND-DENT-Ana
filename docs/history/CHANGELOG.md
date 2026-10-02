# Method history

The distortion method evolved through several epochs. The detailed per-epoch
reports in this folder are kept verbatim for provenance; this file indexes them
and summarizes the arc. The **current** method is the two-iteration spatial map
→ spline → E-field layer → combined SCE ROOT, with the in-situ x-ramp
correction and per-plane anode selection (v10c), re-run on expanded statistics
(v9 / v11). See `../../README.md` and `../../CLAUDE.md` for how to run it.

## Arc (newest first)

- **v11 respin** — cathode + distortion re-run augmented with the 2026-Summer
  DENT-Study SAM sample. `V11_RESPIN_REPORT.md`.
- **v10 / v10c — in-situ x-ramp correction.** The residual dx over-measurement
  is a per-TPC linear ramp in drift distance (a hit-time-bias effect). Resolved
  by a joint (shared-ε) ramp fit refit each voxelizer iteration
  (`fit_dx_ramp.py`, `--x-ramp-correct hits --x-ramp-joint`), plus a measured
  mean-cathode-offset term (`--x-ramp-cathode-offset`) and per-plane anode
  selection. `DX_INVERSION_V2_REPORT_mc.md`, `DX_CLOSURE_CAMPAIGN_mc.md`.
- **dx-closure campaign** — systematic intervention matrix proving the fake-dome
  is track-correlated (field-independent hit-finder bias); closure = base +
  calibrated damping or the in-situ ramp. `DX_CLOSURE_CAMPAIGN_mc.md`.
- **MC closure** — end-to-end validation vs the simulated truth (dual per-TPC
  map); dy/dz recover truth to ~0.07–0.09 cm RMS; dx limited by the
  double-anode's restricted x angular range. `DISCREPANCY_REPORT_mc.md`,
  `INVERSION_DRIFT_STUDY_mc.md`, `CATHODE_DRIFT_MC.md`, `COMMON_OFFSET_STUDY.md`.
- **v9** — same pipeline re-run on the expanded Run2 crossingmuon statistics
  (547 M hits / 250 k tracks).
- **v8 — combined SCE ROOT** — forward + numerically-inverted backward
  (`crtlib.spline.invert_displacement_grid`) + E-field, assembled by
  `build_sce_root.py`.
- **v7 — two-iteration outlier rejection** in `plot_doubleanode_2d_voxels.py`
  (`--iterations 2 --iter-resid-cm 4`).
- **v6 — E-field layer** derived from the spatial map by the MicroBooNE
  technique + a Gauss's-law check (`build_efield_2d_voxels.py`, `drift_model.py`).
- **v5 — the 2-D double-anode method** replacing the CRT / 3-D space-point path.

## Handoff notes

- `HANDOFF_double_anode_df.md` — the per-hit double-anode DataFrame schema.
- `HANDOFF_SCE_AND_CATHODE_V6.md`, `sbnd_cathode_v6_handoff.md` — SCE ROOT +
  cathode v6 ASCII specifications for the SBND services.

## Dropped: CRT workflow

The CRT-to-TPC alignment / `crtdisp` field-distortion path is **deprecated**
(the active method uses wire/tick positions only). Its scripts and the JAX-based
rigid-body optimizer now live under `experiments/crt_alignment/` for reference.
