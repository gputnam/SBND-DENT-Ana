# v11 respin — SBND cathode + field-distortion, augmented with the 2026Summer DENT-Study sample

**Date:** 2026-07-06
**Scope:** Fold a new SAM definition of Run-2 crossing-muon calibration ntuples into
both linked studies (cathode / drift-velocity and 2D double-anode field-distortion),
augmenting the existing parquet datasets, and re-run the full measurement chain as
the **v11** epoch (distortion) / **cathode v5-heatmap, v7-ASCII** epoch. Method and
parameters are unchanged from v10c/AC-v3 — only more input data.

---

## 1. Input sample (SAM definition → filelist)

```
data_SBND2026A_Gen2_Run2_crossingmu_2026Summer_DENT_Study_v10_14_02_04_histreco2_sbnd
```

- Queried via the SBND SAM HTTP API (`samsbnd.fnal.gov:8483`) — the cvmfs
  `sam_web_client` v1_8 is Python-2 and fails on this AlmaLinux-9 host, and
  `samweb.fnal.gov` does not resolve here. Builder: `build_filelist_v11.py`.
- **Files in definition: 22,358.** Production subdir
  `.../SBND2026A/v10_14_02_04/Gen2_Run2_crossingmu_2026Summer_DENT_Study/calib/crossingmuon/`,
  runs 20228–20233 (June 2026), **13 unique runs**.
- **Genuinely new: 22,358 / 22,358 (0 overlap)** with the existing 258,225-file
  union (`filelist_drift_full.txt`) — a distinct production stream from the
  Dec-2025 `gen2_crossingmu-Run2` files already in v9.

Filelists written (study dir): `filelist_v11_defn.txt` (all 22,358),
`filelist_v11_new.txt` (22,358 new, used for the builds).

---

## 2. Track / hit statistics

### New augmentation sample (the 22,358 files)

| Quantity | Value |
|---|---:|
| **Double-anode-crossing tracks** | **166,357** |
| double-anode hits | 378,761,438 |
| events with a DA track | 166,327 |
| DA tracks / event | 1.00 |
| DA hits / track | 2,277 |
| **Cathode-crossing tracks** (AC-through, collection plane, dt₀&dt₁>2400) | **305,985** |
| events with an AC-through track | 299,845 |
| AC tracks / (AC event) | 1.02 |
| AC tracks / (all crossing-µ events) | 0.276 |
| drift-df pool (all AC-candidate tracks) | 5,308,807 |
| drift-df events | 1,107,161 |
| drift-df tracks / event | 4.80 |

This is a crossing-muon-**enriched** sample: ~7.4 double-anode tracks per file vs
~0.97/file in the v9 union — hence 22 k files add 67 % more DA tracks.

### Augmented union (existing + new)

| Dataset | Existing | New | **Augmented** |
|---|---:|---:|---:|
| Double-anode tracks | 249,733 (v9) | 166,357 | **416,090** |
| Double-anode hits | 547,024,448 (v9) | 378,761,438 | **925,785,886** |
| Drift-df tracks | 52,636,587 (v3) | 5,308,807 | **57,945,394** |

(The voxelizer independently confirmed 416,090 DA tracks in the union.)

---

## 3. What was run

Pipeline = the current v10c data procedure (joint ramp + measured
mean-cathode-offset + per-plane anode selection + anode-chord fix) **plus** the v9
E-field / Gauss / combined-SCE-ROOT layers, so the deliverable is the full
forward + numerically-inverted-backward + E-field SCE ROOT. Cathode = the AC-v3
workflow + the stepped v6-family ASCII pipeline, retargeted to the v11 inputs.
Calibration (per-plane anode offsets + drift velocity) **re-measured** on the
augmented sample.

### Key results

**Distortion (v11):** union 416,090 tracks, 412,531 used (99.1 %), 879,962,561 hits
accumulated (3.91 % dropped at |r|>4 cm), **9,698 voxels inverted** (9,609 kept
after cond≤100/res≤2 cm). Applied joint ramp eps = −0.428 cm/m → v_eff 0.15599 cm/µs
(residual +0.020); measured mean cathode-offset c = +0.068 ± 0.038 cm (diagnostic;
implied per-TPC v_DA 0.15593/0.15604). Spline RMS(spline−measured): dx 0.387/0.378,
dy 0.203/0.131, dz 0.210/0.158 cm (TPC0/TPC1) — tighter than v9 from the extra stats.
E-field |E| ∈ [459, 594] V/cm (mean 500); dE/E0 dEx [−0.187,+0.167] dEy
[−0.080,+0.069] dEz [−0.101,+0.065]. Gauss div(E) mean +0.19/+0.21 V/cm² (ρ ≈
+1.7×10⁵ e⁻/cm³ — same positive-ion picture as v6/v9). Forward∘backward closure
median 0.0012 / **RMS 0.0084** / max 0.086 cm; corr(Fwd_X, Bkwd_X) = −0.998 (TPC0) /
−0.999 (TPC1) vs reference −0.998.

**Per-plane anode offsets** (re-measured, augmented union) → `anode_hit_times_data_v11/anode_peak_offset.json`:
TPC0 +1.52/−2.80/0, TPC1 +1.95/−2.57/0 ticks (mid-induction fires early — same
pattern as v10).

**Cathode (v11):** CB drift velocity (`driftv_cb_data_v4`) 3-plane TPC0 0.15577 /
TPC1 0.15653 / TPC-avg 0.15615 / v(sum/400) 0.15623 cm/µs (cathode displaced toward
TPC1, same as AC-v3); pixel-dtsum v = 0.15628 (median). Cathode-displacement spline
RMS 0.232 cm (TPC0) / 0.186 cm (TPC1) at 80×120 knots λ=5. Final v6-grid ASCII:
x_neg ∈ [−4.056,+0.570], x_pos ∈ [+0.053,+4.297] cm, 0 pixels with x_neg>x_pos.

---

## 4. SBND Monte-Carlo input files (the deliverables)

| File | Use |
|---|---|
| `/exp/sbnd/app/users/gputnam/cathode-study/sce_input_doubleanode_2d/sbnd_sce_doubleanode_2d_v11.root` | **Space-charge input.** Combined forward + numerically-inverted backward + E-field, all from the v11 augmented maps. Point `sbnd_spacecharge.InputFilename` here. |
| `/exp/sbnd/data/users/gputnam/efield-study/cathode_pipeline/step3_v11/sbnd_cathode_v7_drift.txt` | **Cathode model** (drift-time-based ASCII) for the SBND CathodeFilter loader. |

(The three intermediate/forward-only distortion ROOTs below are kept for record.)

---

## 5. Complete output inventory

### Study dir `/exp/sbnd/app/users/gputnam/cathode-study/`

**Filelists & glue scripts**
- `filelist_v11_defn.txt`, `filelist_v11_new.txt`
- `build_filelist_v11.py`, `assemble_df_v11.sh`, `count_v11_tracks.py`,
  `concat_drift_v4.py`
- `run_cathode_heatmap_v5.py`, `run_ac_v4_workflow.sh`,
  `run_cathode_pipeline_v11.sh`, `run_pipeline_data_v11.sh`

**Distortion outputs**
- `doubleanode_2d_maps_20cm_v11/` — voxel maps + npz (iter1..iter4) + per-X-slice
  PNG/PDF + dx/dy/dz + stacked GIFs
- `doubleanode_2d_spline_20cm_v11/` — spatial spline 3-pane plots + GIFs + `spline_fit.npz` + `spline_knots.json`
- `efield_2d_maps_20cm_v11/`, `efield_2d_spline_20cm_v11/`, `efield_2d_gauss_20cm_v11/` — E-field voxel map / spline / Gauss div(E) maps + GIFs
- `compare_iter1_iter2_v11/`, `iter_profiles_v11/` — iteration diagnostics
- `compare_2d_vs_truth_v11_dual/` — vs dual-map truth (reference-only for data)
- `profiles_1d_v11_dual/`, `profiles_1d_v11_dual_spline/` — 1D profile grids
- `anode_hit_times_data_v11/` — per-plane anode-peak offsets + `anode_peak_offset.json`

**SCE ROOTs** (`sce_input_doubleanode_2d/`)
- `sbnd_sce_doubleanode_2d_v11.root` — **combined deliverable**
- `sbnd_sce_fwd_spline_doubleanode_2d_v11.root` — forward (spline)
- `sbnd_sce_fwd_voxel_doubleanode_2d_v11.root` — forward (direct voxel)
- `sbnd_sce_efield_doubleanode_2d_v11.root` — E-field intermediate

**Cathode outputs**
- `ac_ticklength_data_v4/` — AC tick-length CB fit (png/pdf/npz)
- `driftv_cb_data_v4/` — Crystal-Ball drift velocity (`driftv_cb_fits.png`, `driftv_cb.csv`)

### Data area `/exp/sbnd/data/users/gputnam/efield-study/`

**Parquets**
- `doubleanode_2d_df_v11_new/` — new-only double-anode hits (4 shards, ~3.9 GB, 378.8 M hits)
- `doubleanode_2d_df_v11/` — **augmented union** (10 shards: 6 v9 symlinks + 4 new)
- `drift_df/drift_df_v11_new.parquet` — new-only drift df (984 MB, 5.31 M tracks)
- `drift_df/drift_df_full_v4.parquet` — **augmented union** (57.9 M tracks)

**Cathode**
- `cathode_study_2cm_v5/` — heatmap CB fits + `cathode_fit_results_tpc{0,1}.csv` + pixel-dtsum
- `cathode_pipeline/step1_displacement_v11/`, `step2_spline_v11/`, `step3_v11/`
- `cathode_pipeline/step3_v11/sbnd_cathode_v7_drift.txt` — **cathode ASCII deliverable**
  (`sbnd_cathode_v6_drift.txt` is the byte-identical script-named copy)

### Run logs (`/tmp/v11_logs/`)
`build_doubleanode_v11.log`, `build_drift_v11.log`, `pipeline_data_v11.log`,
`ac_v4_workflow.log` (tqdm/stderr kept local to avoid the NFS BlockingIOError).

---

## 6. Notes / provenance

- v_drift ramp **start** kept at 0.15532; the joint-ramp correction makes the map
  largely input-velocity-independent (converged v_eff 0.15599). The fresh CB v
  (0.15615 3-plane avg) is recorded but not fed back.
- The dual-map-truth comparison is **reference-only** for data (no ground truth);
  kept for continuity with the MC closure plots.
- v9 / v10c distortion ROOTs and the v3/v6 cathode products are left intact for
  comparison.
