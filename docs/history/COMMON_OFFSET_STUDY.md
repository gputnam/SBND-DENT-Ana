# Common measured−truth offset, SCE-ON vs SCE-OFF (2026-06-22)

**Observation (user):** the double-anode field-distortion map shows the *same*
offset between expectation (truth) and measurement whether SCE is ON or OFF.

**Setup.** Apples-to-apples — both maps produced by the identical v10jc pipeline
(vdrift 0.15627, 4 iters, `--anode-extrapolate`, joint ramp + cathode-offset):
- SCE-ON  = `doubleanode_2d_maps_20cm_mc_full_v10jc` vs the dual MC truth.
- SCE-OFF = `doubleanode_2d_maps_20cm_nosce` vs zero truth (`truth ≡ 0`, so
  residual = measured directly — no truth-sampling step at all).
Analysis: `study_common_offset.py` → `offset_study/` (table + `offset_profiles.png`).

## Finding 1 — the offset is real, field-independent, and lives in dy and dx

Residual (measured − truth) means, common voxels (9,887):

| comp | region | SCE-OFF mean | SCE-ON mean | ON−OFF |
|---|---|---:|---:|---:|
| dx | TPC0 | −0.065 | −0.061 | +0.004 |
| dx | TPC1 | +0.055 | +0.068 | +0.013 |
| dy | TPC0 | −0.014 | −0.011 | +0.002 |
| dy | TPC1 | −0.011 | −0.010 | +0.001 |
| dz | TPC0 | +0.000 | −0.021 | **−0.022** |
| dz | TPC1 | +0.001 | −0.031 | **−0.031** |

- **dy:** a coherent **−0.012 cm** (≈120 µm) bias, the same sign/size in both
  TPCs and (ON−OFF ≈ 0) the same with SCE on or off → field-independent.
- **dx:** a per-TPC **antisymmetric ≈ ∓0.06 cm** offset, reproduced on/off.
- **dz is NOT a common offset:** SCE-OFF dz ≈ 0; the SCE-ON dz −0.026 appears
  *only when a real dz field exists* (ON−OFF ≈ −0.03, correlated with the truth)
  → a mild **underestimate of real dz signal**, a separate effect, not the
  common artifact.

Voxel-by-voxel `residual_ON ~ residual_OFF` correlation is low (r ≈ 0.1–0.3):
only the **structured means** (the offsets) reproduce, not the per-voxel noise —
i.e. it is a coherent systematic, not shared statistical fluctuation.

## Finding 2 — the offset grows with drift distance (the diagnostic clue)

`offset_profiles.png` (residual vs x/y/z, SCE-ON red over SCE-OFF blue lie on
top of each other for dx and dy):

- **dx vs x:** an antisymmetric S-ramp — 0 at the cathode (x=0), ±0.15 cm at the
  anodes (x=±200). This is exactly the residual the **joint** ramp correction
  leaves behind: it removes the TPC-*common* (symmetric) drift ramp but, by
  design, leaves the per-TPC-*antisymmetric* part (which it attributes to a real
  per-TPC v_drift difference). In MC there is no real difference → pure artifact.
- **dy vs x:** a symmetric **dome** — ≈0 at both anodes, ≈−0.02 cm at the cathode.
  The bias **scales with drift distance** (max at the far end of the drift).
- dz vs z (SCE-ON only): ramps −0.05→+0.02, tracking the real field — the
  separate signal-underestimate, not present SCE-OFF.

**This shape is the key.** A *static* geometric offset (a wire-position
calibration error, a mis-set reference chord, a real plane displacement) would be
**flat in x**. The measured offset instead **vanishes at the anode and grows
toward the cathode** — it is accumulated *during the drift*, i.e. it is a
**per-hit, drift-time-dependent** effect, not a fixed geometry offset.

## Possibilities considered

| hypothesis | predicts | verdict |
|---|---|---|
| **Reco hit-time (PeakTime) bias** — angle×drift-dependent hit-finder/deconvolution bias (DX_INVERSION_V2_REPORT_mc.md), field-INDEPENDENT | drift-distance-dependent fake displacement, rectified by the rank-1 LSQ into dx (big) + dy/dz (small couplings); identical on/off | **PRIMARY — matches everything**: same on/off, x-dome/ramp shape, dx dominant. Prior A/B/C forensics: timing-clean slope 0.85 vs reco-hit-time 1.98. |
| Static wire-geometry / `offset_cm` calibration error | **flat-in-x** uniform offset | Disfavored as the dominant cause (shape is a dome, not flat); a small constant sub-component can't be excluded. |
| Anode-reference chord bias | ~constant reference shift → **flat-in-x** | Disfavored (same flat-in-x argument); `--anode-extrapolate` already removes the known inset. |
| Real detector/plane displacement vs nominal geometry | flat-in-x, present on/off | Disfavored by shape; would also be a *real* offset, but cannot make a drift-dependent dome. |
| v_drift scale error | drift-linear dx ramp | Already absorbed by the joint ramp; the leftover antisymmetric dx is what remains. Not a dy cause. |

**Why the X-only correction doesn't remove dy/dz:** the in-situ ramp correction
(`--x-ramp-*`) rebuilds only the hit **drift-x**; it is X-only by construction.
The same underlying hit-time bias couples into dy/dz through the 3-plane LSQ
`n̂` geometry, and nothing in the pipeline touches those components — so the
~−0.012 cm dy dome survives untouched in both samples.

## Conclusion

The "same offset on/off" is a **field-independent reconstruction artifact**, not a
property of the SCE field or its truth map. Its drift-distance dependence (zero at
the anode, growing to the cathode) identifies it as the documented **reco hit-time
(PeakTime) bias**: the dx ramp is its large, partly-corrected face; the residual
per-TPC-antisymmetric dx (±0.15 cm) and the uniform **dy dome (−0.012 cm)** are the
parts the X-only joint-ramp correction leaves behind. dz shows no common offset;
its SCE-ON −0.026 is a separate, small underestimate of the *real* dz signal.

**(Related, separately field-independent:** the CB drift velocity reads
v(sum)=0.15606 vs truth 0.15630, **−0.15%**, the same ON and OFF — the
"effective-path-length floor": finite track angle + reco sampling bias the AC
tick-length slightly high → v slightly low. Geometric, not field-related.)

### If we want to remove it
- Extend the in-situ correction beyond X (a dy/dz analog of the ramp), or
- Since it is field-independent, subtract an **MC-derived additive dy/dz template**
  (it transfers to data exactly because it does not depend on the field), or
- Attack it at the source — calibrate the reco PeakTime vs angle×drift
  (`angle_hit_time_bias.py`) and rebuild hit times before voxelizing.

Outputs: `study_common_offset.py`, `offset_study/{offset_profiles.png,offset_table.csv}`.
