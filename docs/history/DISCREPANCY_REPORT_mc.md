# MC closure discrepancies — investigation report (2026-06-09)

Investigation of two discrepancies seen in the MC measured-vs-truth comparison
(`compare_2d_vs_truth_mc_full`, full ~9,542-file sample):

- **(A)** TPC1's measured **dy and dz are too large** (vs the SCE-map truth),
  more so than TPC0.
- **(B)** measured **dx is too large in both TPCs**.

Five hypotheses were explored in parallel (statistics/selection, voxelizer
geometry & code, drift/x-reconstruction, per-TPC timing & anode reconstruction,
comparison-code audit), plus a decisive cross-check against the MC's own truth
trajectory. **No analysis or pipeline code was changed.**

---

## Headline conclusions

| symptom | verdict | root cause |
|---|---|---|
| **(A) TPC1 dy/dz ~1.45× large** | **NOT our bug** | The **simulation actually applied ~1.5× more SCE displacement in TPC1** than the (symmetric) SCE map specifies. Our 2D method measures the *applied* field correctly. The "discrepancy" is benchmarking against the **map** instead of the **as-applied** displacement. Lives in sbndcode `SpaceChargeServiceSBND` per-TPC application. |
| **(B) dx ~2.8× large, both TPCs** | **Method limitation (expected)** | Intrinsic **geometric magnification** of the double-anode method: the straight anode-to-anode chord can't represent the curved-in-x distortion, and near-drift-parallel tracks give n̂ almost no x-leverage, so the rank-1 inversion amplifies. Partly also the same per-TPC sim asymmetry (dx applied 1.5× larger in TPC1). |

The key evidence is that **two independent measurements agree**: our reconstructed
voxel map and the MC's own truth trajectories both show the TPC1 displacement is
~1.5× the TPC0 displacement.

---

## The decisive test (symptom A): the simulation applied an asymmetric field

The MC ntuple carries the *actually-simulated* trajectory both before and after
SCE: `truth.p.traj.{x,y,z}` (true G4) and `truth.p.traj_sce.{x,y,z}`
(SCE-distorted, used for charge deposition). Their difference **is** the forward
displacement the simulation applied — the genuine ground truth, independent of
our reconstruction.

Measured directly (6 files, ~8.8k trajectory points per TPC, true-x used to
assign TPC):

| comp | applied RMS TPC0 | applied RMS TPC1 | **TPC1/TPC0** |
|---|---|---|---|
| dx | 0.217 | 0.322 | **1.49** |
| dy | 0.586 | 0.904 | **1.54** |
| dz | 0.519 | 0.757 | **1.46** |

Our 2D voxel-map gain (measured/SCE-map slope, full sample):

| comp | TPC0 | TPC1 | **TPC1/TPC0** |
|---|---|---|---|
| dy | 0.94 | 1.38 | **1.48** |
| dz | 0.96 | 1.42 | **1.48** |

The measured TPC1/TPC0 ratio (1.48) matches the *simulated* TPC1/TPC0 ratio
(1.46–1.54). **Our method faithfully measures the field the simulation actually
applied.** Crucially, vs the **as-applied** displacement (not the map), the TPC1
inflation disappears: both TPCs sit at ~0.94× — i.e. the method is TPC-symmetric
and ~unbiased (a small ~6% under-measurement in dy/dz, the same in both TPCs).

The SCE **map** (`SCEoffsets_SBND_E500_voxelTH3.root`, what we benchmarked
against) is exactly cathode-symmetric (truth_TPC0 = mirror(truth_TPC1) to
0.0000 cm). So the asymmetry is introduced **when sbndcode applies the map**,
per-TPC — almost certainly the same per-TPC convention/scale family as the
`TrueFwd_Displacement_X` sign-flip already documented in CLAUDE.md.

---

## What was ruled out for symptom A (and how)

All hypotheses below were tested quantitatively and **rejected** as the cause of
the TPC1 dy/dz inflation:

1. **Statistics / hit-count asymmetry** — TPC0/TPC1 are symmetric to ~1%: hits
   per voxel (median 119k vs 120k), voxels inverted (4993 vs 4991), cond (5.6 vs
   5.9), position-σ (0.0025 cm both), iter2 drop (0.51% both). The gain is **flat
   vs hit count**.
2. **Wire-geometry constants / n̂ / inverter code** — provably TPC-symmetric:
   every `(what_y,what_z)` is exactly unit-normalized (`√(wy²+wz²)=1.000`), `|n̂|=1`
   in both TPCs (`plot_doubleanode_2d_voxels.py:287-289`), `slope_cm=0.30000`
   identical, the rank-1 inverter (`crtlib/voxel_invert.py`) has no per-TPC term.
   A plane-drop or n̂ sign-flip is mathematically **incapable** of producing a
   gain ≠ 1 (the LSQ is self-consistent).
3. **Anode-tick / drift timing** — collection anode tick peak (419.5) and cathode
   tick (2985) are identical TPC0 vs TPC1; the shared `t0=418.5`, `v_drift=0.15627`
   do not bias TPC1 differently.
4. **Induction-anode (y,z) selection** — selected-minus-Tc tick, FOM, 3-plane
   fraction all symmetric.
5. **Track angular population** — `|slope|`, transverse span, n̂ transverse
   magnitude all symmetric.
6. **Comparison / truth-sampling code** — `compare_2d_vs_truth.py` and
   `plot_1d_profiles_vs_truth.py` audited; an **independent** re-derivation
   (fresh `RegularGridInterpolator`, no project helpers) reproduces k_TPC1≈1.40,
   k_TPC0≈0.94, so the inflation is in the measured `delta_mu_all`, not the plots.

The two **discriminating** observations that point at the simulation (not our
code):

- **The gain is flat across drift distance** (TPC1 dy: 1.36→1.40 from |x|=10→150
  cm; dz: 1.40→1.43). A reconstruction drift/timing error would *grow* with
  drift — it doesn't. A pure constant scale is consistent only with the field
  itself being scaled.
- The **isotropic** scaling (equal dy, dz; off-diagonals ≈0) is the signature of
  a scalar field scale, not a geometry/projection error.

---

## Symptom B (dx too large, both TPCs)

This is dominated by the **method's intrinsic x-limitation**, independent of (A):

- Best-fit measured dx = ~2.8 × map-truth dx (r=0.92). Both are domes that vanish
  at the anodes; the measured dome is ~2.5–3× taller.
- **Mechanism:** the 3-D reference is the straight chord between the two
  anode-pierce points (`plot_doubleanode_2d_voxels.py:278-293`), forced to ~0 dx
  at both anode endpoints, so it cannot represent the curved-in-x distortion.
  Double-anode tracks are near-parallel to drift, so n̂ has a small x-component
  (`<n_x²>/<n_y²>`≈0.26); the rank-1 inversion is ~4× weaker in x
  (`cov_xx/cov_yy`≈3.9) and **amplifies** the chord-vs-truth mismatch.
- It is a **coherent bias, not noise** (per-voxel σ_x≈0.004 cm, cond≈5.6 — both
  tiny). Not v_drift (0.15627 is the sim-nominal value; the 0.6% data/sim
  difference is far too small and gives the wrong shape). Not 20-cm voxel
  averaging (a 1.3% effect on the truth).
- Part of the TPC1-vs-TPC0 dx difference (2.3 vs 2.8) is the **same per-TPC sim
  asymmetry** as (A): the applied dx is 1.49× larger in TPC1.

dy/dz, which the method constrains well, are accurate (slopes ~0.94 vs the
as-applied field, r≈0.98) — so the dx limitation is specific to the poorly
constrained drift axis, as already noted in CLAUDE.md ("the ~0.4 cm dx RMS floor
is intrinsic").

---

## Recommended next steps (not yet done — awaiting go-ahead)

1. **Re-benchmark against the as-applied field**, not the map: build a "truth"
   map from `truth.p.traj_sce − truth.p.traj` (or `p_scecorr`) and re-run
   `compare_2d_vs_truth`. Expectation: TPC1 inflation vanishes; method shown
   TPC-symmetric and ~6% low in dy/dz; dx still magnified (method limit).
2. **Localize the sbndcode per-TPC SCE-application asymmetry** in
   `SpaceChargeServiceSBND` (the per-TPC X-flip / offset convention is the prime
   suspect; the ~1.5× scale suggests a per-TPC factor applied to y/z too).
3. **For the dx magnification:** consider a curved (per-hit) reference instead of
   the straight anode-to-anode chord, or restrict dx claims — this is a method
   limitation, documented, not a bug.

---

## Resolution — confirmed against the dual (per-TPC) map (2026-06-09)

The discrepancy is fully resolved by benchmarking against the **per-TPC dual
map** `SCEoffsets_SBND_E500_dualmap_voxelTH3.root` (separate `_E`/`_W`
histograms; **E↔x<0/TPC0, W↔x>0/TPC1**). The W map is uniformly **1.50×** the
E map (X 0.36/0.54, Y 0.71/1.06, Z 0.60/0.90 RMS) — exactly the asymmetry the
simulation applied and the method measured.

Re-running `compare_2d_vs_truth` (and `plot_1d_profiles_vs_truth`, now with two
per-TPC expectation curves) against the dual map:

| comp | r (voxel) | RMS (voxel) | per-TPC RMS (T0/T1) |
|---|---|---|---|
| dy | **0.9966** | **0.091 cm** | 0.071 / 0.107 |
| dz | **0.9962** | **0.070 cm** | 0.059 / 0.080 |
| dx | 0.919 | 0.72 cm | 0.36 / 0.35 (method magnification, now symmetric) |

dy/dz RMS roughly **halved** vs the single map (0.20→0.09, 0.18→0.07) and the
TPC1 inflation is **gone** — measured tracks each TPC's own expectation at
~0.91–0.93× (the small, TPC-symmetric under-measurement). The per-TPC 1D
amplitudes: truth TPC1/TPC0 = 1.50, measured tracks each
(dy 1.06/1.55, dz 1.04/1.54). **Conclusion confirmed: our reconstruction is
faithful; the symmetric single map was simply the wrong benchmark.**

Outputs: `compare_2d_vs_truth_mc_full_dual{,_spline}/`,
`profiles_1d_mc_full_dual{,_spline}/`. `sample_truth_at` auto-detects dual maps.

## Evidence index

- Measured map: `doubleanode_2d_maps_20cm_mc_full/voxel_data.npz`;
  comparison `compare_2d_vs_truth_mc_full/residual_summary.csv`.
- Reconstruction: `plot_doubleanode_2d_voxels.py:260-303`,
  `crtlib/voxel_invert.py:239-349`, `wire_geometry_sbnd.json`.
- Truth ground-truth branches: `truth.p.traj.{x,y,z}`,
  `truth.p.traj_sce.{x,y,z}` (and `truth.p.truehitsN.p_scecorr.{x,y,z}`) in the
  MC `hists_*Reco2*.root` ntuples.
- Map truth: `SCEoffsets_SBND_E500_voxelTH3.root` (cathode-symmetric).
