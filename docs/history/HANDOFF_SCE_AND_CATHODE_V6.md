# SBND MC inputs from cathode-study products

Two pipelines + their deliverables. Authored 2026-05-12 from the
voxel/CSV analysis products in this directory.

## Part 1 — SCE forward-displacement TH3F (DONE)

Source: `doubleanode_maps_v4_20cm_singleg/voxel_data.npz` (20 cm voxel
grid of LSQ-inverted 3-D displacements from the CRT-free double-anode
sample, ~10k voxels, 9573 finite).

Builder: `build_sce_input.py` (this directory).

Deliverable:
```
/exp/sbnd/data/users/gputnam/efield-study/sce_input_doubleanode/
  sbnd_sce_fwd_doubleanode.root      # 9 TH3F histograms, 49x49x61
  fit_diagnostics/{dx,dy,dz}/png/x{NNN}.png  # 49 per-component slices
  fit_diagnostics/{dx,dy,dz}/{dx,dy,dz}.gif  # animated GIFs
  outlier_report.csv                 # 59 voxels dropped (X-neighbor MAD)
  spline_knots.json                  # the chosen knot vectors
  fit_summary.csv                    # per-component chi2/dof
```

What the script does (matches the approved plan):
1. Loads the npz, masks bad voxels (`count<200`, `cond>300`).
2. X-neighbor MAD outlier filter (k=5, abs-floor=2 cm): drops 59 voxels.
3. Splits the grid at x=0 into TPC0 (X<0) and TPC1 (X>0) halves — fits
   two independent tensor-product cubic B-spline meshes, no continuity
   across the cathode.
4. Enforces `Δ̂(x=±200) = 0` in all 3 components by dropping the
   anode-side X basis-function column from the design matrix (clamped
   cubic property: only the boundary basis function is non-zero at the
   endpoint).
5. Weighted LSQ fit with 1/σ² weights (σ_floor = 0.5 cm) AND Tikhonov
   ridge regularization (λ=1.0) — the ridge term tames the
   under-determined spline coefficients with only 10 voxel X-positions
   per half. (CLI: `--ridge-lambda`.)
6. Default knot counts: `--nx-knots 3 --ny-knots 5 --nz-knots 9`. Picked
   via `--diagnostic-knots` scan (smaller knots over-smoothed; bigger
   knots oscillated wildly even with the ridge).
7. Evaluates on the SCE 49×49×61 grid (bin centres at -200..+200 in X,Y
   and 0..500 in Z, 8.333 cm voxels), applies the TPC0 X-sign flip so
   that `SpaceChargeServiceSBND::GetPosOffsets` (with its `corr=-1` for
   `x<0`) reproduces the measured displacement direction.
8. Writes TH3F (float32) via `uproot.writing.identify.to_TH3x` —
   confirmed loadable back as `Model_TH3F_v4`. Forward Displacement gets
   the fitted values; TrueBkwd_Displacement_* and True_ElecField_* are
   zero (we have no backward map and no separate E-field measurement).

To use it in SBND MC, point the SCE service at the new file:
```
services.SpaceCharge: @local::sbnd_spacecharge
services.SpaceCharge.InputFilename: "/exp/sbnd/data/users/gputnam/efield-study/sce_input_doubleanode/sbnd_sce_fwd_doubleanode.root"
services.SpaceCharge.EnableSimSpatialSCE: true
services.SpaceCharge.RepresentationType: "Voxelized_TH3"
```

Verified values (uproot read-back):
- All 6 boundary X-planes (x=±200) are 0 in every component.
- Stored ranges: dx [-3.7, +2.6], dy [-2.3, +4.3], dz [-4.5, +3.1] cm.
- TPC0 stored value at x=-100, y=0, z=250 is -0.32 cm; loader output
  (corr=-1) = +0.32 cm — matches the physical signed measurement.

To re-tune knots / ridge / outlier filter, see `build_sce_input.py
--help`. The diagnostic-knots mode prints χ²/dof + extrapolation
ranges without writing the ROOT file.

## Part 2 — Cathode v6 ASCII (script ready; awaiting drift_df rebuild)

### Status

- `build_cathode_v6.py` (this directory) is implemented and tested.
- A **preview** v6 file built from the InTime-only CSVs lives at:
  ```
  /exp/sbnd/data/users/gputnam/efield-study/cathode_v6_doubleanode_intime_preview/
    sbnd_cathode_v6_drift.txt
    png/{mu_TPC0,mu_TPC1,x_neg,x_pos,thickness,consistency}.png
  ```
- The **drift_df rebuild** (~225k file union: InTime+Run1+Run2 cross-mu)
  is running in the background — see `drift_df_full_v2.log` for
  progress. ETA ~9-25 hours from 2026-05-12 17:05. Output:
  `/exp/sbnd/data/users/gputnam/efield-study/drift_df/drift_df_full_v2.parquet`.

### Steps remaining (after the drift_df rebuild finishes)

Run:
```bash
./post_drift_df_cathode_v6.sh
```

That script:
1. Runs `plot_cathode_heatmap.py --pixel-size 2 --mode combined` on the
   new parquet, writing
   `/exp/sbnd/data/users/gputnam/efield-study/cathode_study_2cm_v2/`.
2. Parses the new empirical `v_drift = 4m/<dt0+dt1>` from the heatmap
   log.
3. Runs `build_cathode_v6.py --centring global --panel-half-thickness 1.5`
   with that v_drift on the new CSVs, writing
   `/exp/sbnd/data/users/gputnam/efield-study/cathode_v6_doubleanode/`.

### What `build_cathode_v6.py` does

1. Loads `cathode_fit_results_tpc{0,1}.csv` (200 × 250 native pixel grid
   at 2 cm).
2. Per-TPC: drop `mu_err > 10 tick` (failed fits), then 8-neighbor MAD
   filter (k=5, abs-floor=20 ticks). Typical: drop ~10% of pixels.
3. Per-TPC: fit a 2-D `LSQBivariateSpline` (cubic) with 8 y-knots, 12
   z-knots, weights `1/mu_err²` (floor 2 ticks).
4. Evaluate each spline on the v6 160 × 200 grid (Y∈[-200,+200],
   Z∈[0,500], 2.5 cm bins).
5. Tick → cm conversion: `δx = (2/3) · v_drift · 0.5 us · (mu - mu_ref)`,
   with:
   - `--centring global` (default): `mu_ref = 200 / (v_drift · 0.5 us)`,
     the universal nominal drift duration for a flat cathode at x=0.
     Gives the absolute geometric cathode displacement, preserving any
     per-TPC bias (clock offset, field uniformity etc.).
   - `--centring per_tpc`: `mu_ref = median(mu)` per TPC; centres each
     view at 0. Use when only the local cathode bumps matter and you
     want the global offset removed.
   - Sign: `x_TPC0 = +(2/3)·v·dt·(mu0 - mu_ref)`,
     `x_TPC1 = -(2/3)·v·dt·(mu1 - mu_ref)`. Both estimate the same
     physical δx_c. (Derivation: `v ~ √E = √(V/L)`, so `mu = L^(3/2)`,
     `dmu/mu_nom = (3/2)·δL/L_nom`, `δx = (2/3)·v·δt`.)
6. NMIN mask: each v6 pixel needs ≥5 finite native pixels within ±5 cm
   to be filled; otherwise `nan`.
7. v6 mapping (user-selected): `x_neg = x_TPC0 - panel_half_thickness`,
   `x_pos = x_TPC1 + panel_half_thickness`. The half-thickness offset
   (default 1.5 cm) gives the panel a ~3 cm "thickness corridor"
   matching the existing endpoint-based v5/v6 file. Without it the two
   per-TPC estimates would overlap and the v6 loader's "x_neg ≤ x ≤
   x_pos" inside-cathode test would degenerate.
8. Writes the ASCII file in exactly the format documented in
   `sbnd_cathode_v6_handoff.md`.

### Why a `--panel-half-thickness` flag?

The hit-based drift-time analysis measures the cathode plane (a single
x position per (y,z), per TPC view), not the full panel thickness. The
existing v6 file came from a separate endpoint-percentile analysis that
captures the panel structure (~3 cm thick) directly. To produce a v6
file that the C++ loader can use as a meaningful "inside-cathode" test,
we add a fixed ±half-thickness offset around our measured cathode plane.

If you want a thin (delta-function) cathode set `--panel-half-thickness
0.0`. Expect 50%+ of pixels to have `x_neg > x_pos` then — the loader
will treat those as "outside the cathode", which is the safe fallback.

### Verification checklist (from sbnd_cathode_v6_handoff.md)

Once the final file exists, the script writes diagnostic PNGs that let
you eyeball:
- nan-cell count parity between x_neg and x_pos
- central pixels |x| < 4 cm
- thickness panel (`x_pos - x_neg`) ~0.5 - 3 cm with support-bar voids
- consistency plot (`x_TPC0 + x_TPC1`): if this is close to zero
  everywhere, the two TPC views agree. Non-zero residual is a real
  per-TPC bias.

## Tasks tracker

- [x] Build SCE forward-displacement ROOT from doubleanode v4 voxel map.
- [/] Union filelist + rebuild drift_df parquet (running in background;
      `drift_df_full_v2.log`).
- [ ] Run `plot_cathode_heatmap.py` on the new parquet at 2 cm pixels.
      Triggered by `post_drift_df_cathode_v6.sh` step 1.
- [ ] Build cathode v6 ASCII from new CSVs. Triggered by
      `post_drift_df_cathode_v6.sh` step 2.

## Files in this directory created for this task

- `build_sce_input.py` — Part 1 builder.
- `build_cathode_v6.py` — Part 2 builder.
- `post_drift_df_cathode_v6.sh` — kicks off the cathode rebuild once
  drift_df_full_v2.parquet exists.
- `filelist_drift_full.txt` — union of InTime + Run1 + Run2 crossingmuon
  (225,494 files) used by the drift_df rebuild.
- `HANDOFF_SCE_AND_CATHODE_V6.md` — this file.
