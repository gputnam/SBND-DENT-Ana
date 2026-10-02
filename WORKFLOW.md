# Pipeline map

Condensed data-flow for both analyses. Exact commands: `README.md`.
Boxes are scripts; arrows carry the named artifact.

## Field distortion (2-D double-anode → SCE ROOT)

```
caloskim ntuples
  │  make_doubleanode_2d_df_sbnd.py        (xrootd; per-hit raw plane/tpc/wire/time)
  ▼
doubleanode_2d_df/*.parquet
  │  optimize_anode_cut.py  ──► config/cut_tuning/anode_cut.json
  │  plot_doubleanode_2d_voxels.py         (voxel LSQ; 2-iter; anode-extrapolate; joint x-ramp)
  ▼
voxel_data.npz  (delta_mu_all = spatial distortion per voxel)
  │  fit_doubleanode_2d_spline.py --write-sce-root
  ├──────────────► sbnd_sce_fwd_*.root     (forward displacement)
  ▼
spline_knots.json + spline_fit.npz
  │  build_efield_2d_voxels.py             (MicroBooNE g=dΔ/dx → dE; drift_model v(E))
  ▼
efield voxel_data.npz
  │  fit_efield_2d_spline.py --write-sce-root ──► sbnd_sce_efield_*.root
  ▼
  │  build_sce_root.py   (forward + numerically-inverted backward + E-field)
  ▼
sbnd_sce_doubleanode_2d_*.root   ◄── sbnd_spacecharge.InputFilename
```

Validation: `check_gauss_2d.py` (∇·E), `compare_2d_vs_truth.py` /
`plot_1d_profiles_vs_truth.py` (MC closure), `compare_iter1_iter2.py`.

## Cathode distortion (drift velocity + cathode ASCII)

```
caloskim ntuples
  │  make_driftV_df_sbnd.py                (per-track collection start/end times)
  ▼
drift_df.parquet
  │  plot_cathode_heatmap.py               (per-(y,z) CB cathode-tick fit)
  │  fit_driftv_cb.py                      (drift velocity, CB fit of AC tick lengths)
  ▼
cathode_fit_results_tpc{0,1}.csv
  │  cathode_step1_tick_to_cm.py           (tick → cm, (2/3)·v·dt)
  ▼
tpc{0,1}_displacement.npz
  │  cathode_step2_spline.py               (per-TPC 2-D ridge B-spline; crtlib.spline)
  ▼
tpc{0,1}_spline.npz
  │  cathode_step3_v6.py                   (NMIN-masked v6 grid)
  ▼
sbnd_cathode_v6_drift.txt   ◄── SBND CathodeFilter loader
```
