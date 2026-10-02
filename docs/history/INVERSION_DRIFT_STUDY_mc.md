> **CORRECTION (2026-06-09, see `CATHODE_DRIFT_MC.md`):** the direct
> cathode-crossing measurement gives MC v_drift ≈ **0.15655 cm/µs ≈ sim-nominal**,
> NOT 0.15532. So the "MC was built with the data v_drift 0.15532" claim below is
> **refuted** — 0.15532 merely *coincidentally cancels* the double-anode dx
> magnification (dx is hyper-sensitive to v_drift). The reconstruction at 0.15627
> used essentially the correct v_drift; the dx bias is the intrinsic method
> magnification, and the **anisotropic x-ridge is the proper fix** (the
> "reconstruct at 0.15532" recommendation is withdrawn). The regularization
> study below stands as-is.

# dx bias: inversion/regularization study + drift-velocity verification (2026-06-09)

Goal: reduce the **dx measurement bias** of the 2D double-anode method (measured
dx ~2× the truth, given the weak x-direction handle), and verify whether the
**drift velocity** contributes. Benchmark = the per-TPC dual map
`SCEoffsets_SBND_E500_dualmap_voxelTH3.root`. Study done on the subset
accumulator (`accumulator_mc_subset.npz`, dumped via the new
`plot_doubleanode_2d_voxels.py --dump-accumulator`); strategies tested offline
on the per-voxel normal equations M=Σn̂n̂ᵀ, v=Σn̂m (`explore_inversion.py`).

## Headline

1. **Drift velocity IS a culprit — strongly.** dx is hyper-sensitive to
   `v_drift`: the slope (measured/truth) crosses **1.0 at v_drift ≈ 0.15532**
   (the data-measured value) and is **2.0 at 0.15627** (sim-nominal, which we
   used for the MC). dy/dz are essentially v_drift-insensitive.
2. **Best inversion strategy: anisotropic x-ridge** (penalize only dx),
   λx ≈ 0.1–0.15: dx RMS **0.80 → 0.17–0.22 cm (≈4×)**, slope 2.0 → ~1.0,
   while dy/dz are nearly untouched — and it is **robust to v_drift error**.

## 1. Drift-velocity verification (`explore_inversion_mc/dx_bias_vs_vdrift.png`)

Re-reconstructing the subset at several v_drift (tick→x = anodeₓ ∓ (t−t0)·0.5µs·v):

| v_drift | dx slope | dx RMS | dy slope | dz slope |
|---|---|---|---|---|
| 0.1450 | −10.9 | 8.1 | 0.99 | 1.01 |
| 0.1500 | −4.9 | 4.1 | 0.96 | 0.98 |
| **0.15532 (data)** | **+1.02** | **0.53** | 0.93 | 0.95 |
| 0.15627 (sim, used) | +2.03 | 0.89 | 0.93 | 0.94 |
| 0.1650 | +10.7 | 5.8 | 0.87 | 0.88 |

**The dx bias is dominated by v_drift.** A 0.6 % v_drift change
(0.15627→0.15532) flips the dx slope by 2× — a ~150× amplification, because the
straight anode-to-anode reference pins dx=0 at the anodes, so a v_drift error
produces a drift-distance-proportional x-error that the weak x-leverage
amplifies. The prior assertion that "v_drift is too small (0.6 %) to matter" was
wrong: it ignored this leverage amplification. dy/dz, which the method
constrains well, are flat vs v_drift (~0.93).

**Why 0.15532?** The MC was produced with the **"DataDrivenDetectorModel"**,
which uses the **data-measured** drift velocity (0.15532 cm/µs — our cathode
result, the value CLAUDE.md says the spatial pipeline should use). For the MC
run we picked sim-nominal 0.15627; that mismatch *is* the dx bias. **Re-run the
MC reconstruction at 0.15532** to remove it (dx slope→1, RMS 0.89→0.53).

## 2. Inversion / regularization strategies (`.../dx_bias_vs_regularization.png`, `strategy_metrics.csv`)

Per-voxel solve δ=(M+R)⁻¹v on the SAME voxel set (cond≤100, n≥30), scored vs the
dual truth. Baseline = production (ridge 1e-6, ~none):

| strategy | dx slope | dx RMS | dy RMS | dz RMS |
|---|---|---|---|---|
| baseline | 2.04 | 0.795 | 0.135 | 0.171 |
| iso_ridge λ=0.1 | 1.06 | 0.211 | **0.257** | **0.225** |
| **x_ridge λx=0.10** | **1.10** | **0.215** | 0.164 | 0.179 |
| **x_ridge λx=0.15** | 0.90 | **0.173** | 0.174 | 0.185 |
| svd_floor f=0.5 | 1.03 | 0.263 | 0.192 | 0.196 |
| svd_trunc (drop small eig) | 0.01 | 0.485 | 0.261 | 0.257 |
| dx_zero (δx≡0) | — | ~0.48 | — | — |

- **Anisotropic x-ridge wins.** `R = diag(λx·n, 0, 0)` shrinks only the
  poorly-constrained dx. At λx≈0.1–0.15 it cuts dx RMS ~4× and slope→~1, with
  dy/dz RMS rising only 0.02–0.03 cm.
- **Isotropic ridge** reaches the same dx but **harms dy/dz** (RMS 0.135→0.257) —
  it over-regularizes the well-measured directions. Avoid.
- **SVD-truncation / dx-zero** over-kill dx (set it ~0) → RMS 0.48, *worse* than a
  regularized dx: keeping a shrunk dx beats discarding it.
- **Robustness:** x-ridge is computed at the *wrong* v_drift (0.15627) yet
  recovers dx RMS 0.17 — it suppresses the v_drift-induced amplification too, and
  gives *lower* dx RMS than the v_drift fix alone (0.17 vs 0.53) because it also
  damps the per-voxel scatter.

## Recommendations (not yet applied to production)

1. **Reconstruct the MC at v_drift = 0.15532** (the data-driven value the MC was
   built with) — removes the coherent dx bias. (Equivalent: the production
   spatial pipeline already specifies 0.15532; only the MC run used 0.15627.)
2. **Add an anisotropic x-ridge** `λx≈0.1` to `invert_scalar_accumulator`
   (`R = diag(λx·n,0,0)`) — a cheap, additive change that cuts dx RMS ~4×,
   preserves dy/dz, and makes dx **robust to v_drift miscalibration**. Best
   combined with #1.
3. Do **not** use isotropic ridge or SVD-truncation (collateral dy/dz damage /
   over-kill).

## Artifacts
- `explore_inversion.py` (strategy scan), `plot_doubleanode_2d_voxels.py
  --dump-accumulator` (new, additive), `accumulator_mc_subset.npz`.
- `explore_inversion_mc/`: `strategy_metrics.csv`, `dx_bias_vs_regularization.png`,
  `dx_bias_vs_vdrift.png`.
