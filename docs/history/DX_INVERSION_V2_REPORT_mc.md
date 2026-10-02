# dx inversion v2: the MC double-anode X-offset gap, resolved (2026-06-10)

## Executive summary

Against the CORRECT benchmark — the as-applied SCE displacement field,
voxelized from the MC truth (`apparent_truth_map_mc_v2.npz`,
`delta_applied`; y/z-averaged dx profile ±0.47/0.71 cm, voxel RMS 0.48) —
the measured double-anode dx is

    measured  =  1.0 x (real applied dome)  +  ~1.0 x (fake dome from
                 reco-hit peak-time bias)   +  small geometry terms,

i.e. the famous "factor ~2". Established by running the SAME tracks/wires
through the full pipeline with different hit times (A/B/C forensics):

| variant (all with the chord fix) | dx slope vs applied | dx RMS |
|---|---:|---:|
| A: nominal-drift times (timing-clean) | **0.852** | 0.573 |
| B: reco hit times (production content) | **1.983** | 0.840 |

The pipeline itself measures a clean displacement signal at ~85%; the
doubling is injected by the reconstructed hit times.

Two pipeline issues were found and fixed along the way; several
mitigations of the hit-time dome were implemented and quantified (none
removes it fully — it is track-correlated); and the practical closure is a
calibrated dx suppression:

| production-subset map vs applied benchmark | dx slope | dx RMS | dy RMS | dz RMS |
|---|---:|---:|---:|---:|
| baseline | 2.142 | 0.794 | 0.127 | 0.167 |
| + `--anode-extrapolate` | 1.977 | 0.734 | 0.126 | 0.167 |
| + hit-time LUT | 1.892 | 0.659 | 0.126 | 0.165 |
| **extrap + dx x 0.5** | **0.989** | **0.279** | 0.126 | 0.167 |
| x_ridge 0.1 | 1.126 | 0.216 | 0.155 | 0.175 |
| x_ridge 0.2 | 0.776 | 0.179 | 0.172 | 0.187 |

The previously "arbitrary" x-ridge λx≈0.1 lands at slope 1.13 vs the
applied field — it was approximately the right answer; the decomposition
above is the reason why (it damps the ×2 by ~half).

### Full-statistics X-only angular selection (tan θ_xw > 0.8) — 2026-06-10

Per the user's direction, the production-style measurement was re-run at
full statistics (13 shards, 1.18B hits accumulated, 570k tracks) with the
X-ONLY angular selection: dx per voxel solved from only the hits with
tan(θ_xw) = |ds/dx| ≥ 0.8 (104.0M hits, 8.8%; where the reco-hit-time
bias is flat), conditional on the all-hits dy/dz
(`plot_doubleanode_2d_voxels.py --min-abs-slope`, dual-accumulator
implementation), plus `--anode-extrapolate`. Outputs
`*_mc_full_thxw08*`; memory-calibrated to 2 accumulation workers
(~37 GB/worker peak).

Results — dy/dz: the best of the entire study (vs applied field: slope
0.977/0.967, RMS 0.054/0.047 cm; vs dual map: r = 0.998/0.997). dx:
**slope +1.964 vs the applied field (RMS 0.72 cm), r = 0.884 vs the dual
map** — statistically unchanged from the no-cut value (1.98). At full
statistics the angular selection does NOT remove the fake dome: the
reco-time bias that survives in the high-θ_xw subset is track-correlated
rather than angle-mean driven, consistent with the LUT and
high-leverage-estimator results. dx coverage with the cut: 8,400/9,984
voxels. The 1D expectation-vs-measurement grids
(`profiles_1d_mc_full_thxw08_dual{,_spline}/`) show the measured dx dome
sitting ×~2 above the per-TPC truth curves while dy/dz track them.

## Simulation/ntuple conventions — verified in code

This investigation initially went wrong by misreading the truth branches;
the verified facts (sbncode v10_14_02 / sbndcode v10_14_02):

* `TrackCaloSkimmer_module.cc:775-785` — `truehits.p` is the raw
  `sim::IDE` position = the **displaced** (drift-frame) deposit position.
* `:564-578` — `p_scecorr` = `GetCalPosOffsets`-corrected = ≈ the **true**
  position (good to the backward map's inversion accuracy, ~0.01–0.1 cm).
* `:822-825` — `truehits.time` and `tdrift` are **recomputed** as nominal
  drift from `p.x` (`ConvertXToTicks`, `|p.x − plane.x|/DriftVelocity()`)
  — they are NOT simulated times.
* SBND WireCell sim (`simparams.jsonnet:16`): **constant**
  `drift_speed: 1.563 mm/us` — there is NO field-dependent drift velocity
  in detsim; the drift-time distortion derives entirely from the
  X-displacement of the deposits.
* SCE enters at the g4 stage (`g4_enable_doublespacecharge_sbnd.fcl`):
  `EnableSimSpatialSCE: true` shifting deposits by the forward offsets of
  `SCEoffsets_SBND_E500_DoubleCharge_voxelTH3.root` (double space-charge
  map; the as-applied field, not the Gen1 nominal). `EnableSimEfieldSCE`
  affects recombination only.
* Footnote: WireCell drifts at 1.563, reco assumes 1.5627 mm/µs — a 0.04%
  mismatch worth ~0.07 cm linear at full drift.

So the correct dx benchmark = the as-applied spatial dome (deposits are
shifted; arrival time = nominal drift from the shifted position). The
benchmark map is built by `build_apparent_truth_map.py` from truth-matched
hits: `delta_applied = (p − p_scecorr)` binned at the true position
(plus an "apparent" dx layer from matched reco times for diagnostics).

## What generates the fake ×1 dome: reco peak-time bias

`angle_hit_time_bias.py` (`angle_bias_mc/`), from truth-matched hits:
τ = PeakTime − nominal-drift time is strongly **track-angle and
drift-distance dependent**: ±0.3–0.6 cm equivalent at |ds/dx| ≲ 0.5
(drift-parallel tracks → long per-wire integration windows,
Δx/wire = d_w/|ds/dx|; diffusion broadens with drift), flat and small for
|ds/dx| ≳ 1. In θ_xw terms (tan θ_xw = ds/dx, n_x = −sin θ_xw): the bias
lives at small θ_xw, and the LSQ rectification strength is sin²θ_xw.

How it becomes a dome: the naive n_x²-weighted absorption is only
±0.08 cm — but the low-slope hits push their bias into the fitted Δy/Δz
(n_y·n_x cross-moments), and the per-voxel 3×3 LSQ converts the angle
pattern into dx at full strength (measured directly: the per-plane
resid-vs-n_x slopes are ±0.92–1.04, identical in all three planes —
`diagnose_dx_mc/`).

**Mitigations implemented and quantified** (B-variant, vs applied):

| mitigation | dx slope | comment |
|---|---:|---|
| none (B) | 1.983 | |
| binned τ LUT (`--hit-time-corr`; |slope| or signed identical) | 1.896 | removes only the binned mean; the dome is within-bin, track-correlated (overlapping-pulse deconvolution errors along drift-parallel tracks) |
| per-hit cut |slope|≥0.5 (`--min-abs-slope`) | 1.977 | low-slope hits carry little n_x leverage; cutting them doesn't remove the absorbed bias; dy/dz degrade (RMS 0.56/0.43) |
| per-hit cut |slope|≥1.0 | 1.896 | stats collapse (1,215 voxels), dy/dz destroyed |
| hi-leverage conditional dx (dy/dz from all hits, dx from |n_x|>0.45/0.707 residuals) | 1.78/1.64 | blocked by the τ contamination of the yz-block via n_y·n_x cross-moments |
| E-variant (true-time chord + reco-time hits, `--chord-source-dir`) | dome grows to ±1.8 | the reco-time chord partially common-modes the hit bias — artifact is per-hit, not chord machinery |

The full removal would need per-track time-correlation modeling (a joint
fit with per-track nuisances), or a finer τ parameterization
(θ_xw, dw/dx, drift) — both documented as future directions.

## Pipeline fixes (adopt)

1. **`--anode-extrapolate`** — the selected anode hits sit up to the
   40-tick window inside the detector but their view coords were assigned
   to x=±200, tilting the chord by (inset × track slope). Proven in the
   toy closure (`toy_closure_2d.py`: synthetic hits, known field, real
   pipeline code — dx slope 0.34 → **0.979** with the fix; exact bias
   formula unit-tested in `test_anode_extrapolate.py`). On the real MC:
   2.14 → 1.98. Unconditionally correct.
2. `reconstruct_anodes` robustness for tracks lacking collection hits in
   a TPC (needed for non-production parquets).

## Per-component recommendation

* **dy/dz**: clean everywhere (RMS 0.13–0.17 baseline; 0.085/0.097 with
  the boundary-constrained global spline solve,
  `fit_global_spline_from_accum.py` — anode BC Δ=0 + cathode BC Δx=0,
  full per-voxel covariance).
* **dx**: artifact-dominated at ×~2 of the real signal. Options measured:
  scale ×0.5 after the extrap fix (slope 0.989/RMS 0.279), x_ridge 0.1
  (1.126/0.216). The fake component is field-independent (it's a
  hit-finder property) while the real part scales with the SCE field — so
  for DATA a multiplicative calibration is only valid to the extent the
  data field ≈ MC field; an additive MC-derived fake-dome template, or
  the in-situ angle-binned consistency check, are the more defensible
  data-side treatments.

## Tooling index

```
diagnose_dx_offsets.py          per-voxel x_est distributions (gallery), per-plane
                                dx(x) profiles, resid-vs-n_x, leverage; also the
                                hi-leverage conditional-dx estimator (--nx-min)
measure_truth_dx_mc.py          truth decomposition profiles (NOTE: p/p_scecorr
                                roles per the conventions above)
compare_hit_times_mc.py         reco-vs-true hit-time profiles
angle_hit_time_bias.py          tau(angle, drift) + --write-lut
build_truthhit_parquets_mc.py   A/B/C parquets (production selection replicated)
compare_abc_maps.py             A/B/C overlays (--suffix for variants)
build_apparent_truth_map.py     applied + apparent truth benchmark maps
toy_closure_2d.py               synthetic full-pipeline closure
fit_global_spline_from_accum.py boundary-constrained global solve (S1)
score_dx_strategies.py          unified scoring (--apparent-truth)
test_anode_extrapolate.py       chord-fix unit test
new voxelizer flags: --anode-extrapolate, --hit-time-corr, --min-abs-slope,
                     --iter-correct-positions/--iter-relax, --plane-t0-json,
                     --chord-source-dir (forensic)
```

History note: an earlier revision of this report claimed an independent
"drift-velocity distortion layer" in the simulation based on
`truehits.time`/`tdrift`; that was wrong (those branches are nominal-drift
recomputations) and is retracted — see the conventions section.
