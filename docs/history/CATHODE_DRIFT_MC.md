# MC cathode-crossing study: drift velocity + cathode positions (2026-06-09)

Ran the existing cathode-crossing / drift-velocity workflow on the MC
(`DataDrivenDetectorModel_2026`, full ~9,542 files → `drift_df_mc.parquet`,
**2,107,926** tracks; ~690k anode-cathode through-going per plane). Goal:
measure the MC drift velocity and cathode positions directly, and validate
against truth.

## 1. Drift velocity — the headline (corrects an earlier inference)

Measured geometrically from cathode crossers, `v = 400 cm / ⟨(dt₀+dt₁)·0.5 µs⟩`
(independent of any assumed v_drift):

| plane | v_drift [cm/µs] | N (dt>2400 both TPC) |
|---|---|---|
| 0 | 0.15654 | 689,801 |
| 1 | 0.15662 | 653,842 |
| 2 | 0.15650 | 719,337 |
| **mean** | **≈ 0.15655** | |

**The MC drift velocity is ≈ 0.1565 cm/µs ≈ sim-nominal (0.15627), NOT the data
value 0.15532.** This **refutes** the earlier indirect inference
(`INVERSION_DRIFT_STUDY_mc.md`) that the MC was built with the data v_drift.
That inference came from the double-anode dx-bias scan crossing slope = 1 at
0.15532 — but the cathode crossers show 0.15532 is **not** the true v_drift; it
was merely the value that *coincidentally cancels* the double-anode dx
magnification (dx is hyper-sensitive to v_drift via the weak-x leverage).

**Implications:**
- The double-anode MC maps reconstructed at 0.15627 used essentially the
  **correct** v_drift (within 0.2%). The dx over-measurement is the **intrinsic
  method magnification**, not a v_drift error.
- The **anisotropic x-ridge** is therefore the right fix for the dx bias; the
  "reconstruct at 0.15532" recommendation is **withdrawn** (it would use a wrong
  v_drift and only works by coincidence).
- (`print_drift_velocity.py` returns 0.15627 at SBND sim defaults; the per-track
  median here is ~0.15618, the mean 0.15655 — a mildly right-skewed drift-time
  distribution.)

## 2. Cathode positions

`run_cathode_heatmap_mc.py` (tuned DT_WINDOW/DT_BIN/MIN_COUNTS) →
`cathode_study_mc/cathode_fit_results_tpc{0,1}.csv` (per-(y,z) 2-cm pixel
cathode drift-time Crystal-Ball peak; 41.5k/50k pixels fit per TPC — the gun
illuminates (y,z) broadly). `cathode_step1_tick_to_cm.py --v-drift 0.15655` →
per-TPC apparent cathode displacement `x_cathode(y,z)`:

- TPC0: mean **+0.39 cm**, scatter 0.15 cm; TPC1: mean **−0.39 cm**, scatter
  0.17 cm (anti-symmetric, as the per-TPC sign convention dictates).
- Localized −1 to −2 cm features at z≈255, y≈±50/±150 — the **cathode
  support-frame** positions (real structure / CB-fit pathology, same family as
  the data "cathode-mesh-frame stripes").

### Truth comparison (`compare_cathode_vs_truth.py`)
The dual SCE map's forward Δx is **exactly anti-symmetric and pinned to 0 at the
cathode** (Δx(x=0,y,z) ≡ 0; it grows ~linearly to ±0.45 (E) / ±0.68 (W) by
|x|=83 cm). So the **true cathode displacement is zero — the MC cathode is
undisplaced** at nominal x=0.

The measured ±0.39 cm uniform offset is therefore **not** a real displacement:
it is a v_drift-**estimator** artifact in `mu_nominal`. `mu_nominal` used the
*mean* v (0.15655), but the cathode dt *peak* (mode) sits at ~2562.7 ticks =
v≈0.1561 — a +7.6-tick (mean−mode) gap → ±0.39 cm. Using the peak-consistent v
nulls the offset. The residual ~0.15 cm scatter is the method's systematic floor
(no truth structure at the cathode to recover, since Δx≡0 there).

**Net:** the workflow validates cleanly — MC v_drift ≈ sim-nominal, and the MC
cathode is flat at x=0 (the method's absolute zero is set by the v-estimator
choice to ~±0.4 cm, with ~0.15 cm pixel scatter).

## Outputs
```
/exp/sbnd/data/users/gputnam/efield-study/drift_df_mc.parquet  (2.1M tracks, 468 MB)
cathode_study_mc/      cathode_fit_results_tpc{0,1}.csv + cathode-tick heatmaps
cathode_mc_step1/      tpc{0,1}_displacement.npz + displacement/mu heatmaps
compare_cathode_vs_truth_mc/  per-TPC compare heatmaps + cathode_truth_summary.csv
build_driftdf_mc.log, cathode_heatmap_mc.log
```
New scripts: `run_cathode_heatmap_mc.py` (MC analog of run_cathode_heatmap_v3),
`compare_cathode_vs_truth.py`. `make_driftV_df_sbnd.py`,
`plot_cathode_heatmap.py`, `cathode_step1_tick_to_cm.py` reused unmodified.
