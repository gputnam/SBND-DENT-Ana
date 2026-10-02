"""Investigate the residual offset (measured - truth) that is COMMON to the
SCE-ON (mc_full_v10jc, dual truth) and SCE-OFF (nosce, zero truth) double-anode
maps. Apples-to-apples: both produced by the identical v10jc pipeline
(vdrift 0.15627, 4 iters, anode-extrapolate, joint ramp + cathode-offset).

Outputs to offset_study/:
  - per-component / per-TPC offset table (printed + offset_table.csv)
  - voxel-by-voxel regression residual_ON vs residual_OFF (slope/intercept/r)
  - profiles of each residual component vs x, y, z (SCE-ON and SCE-OFF overlaid)
    -> offset_profiles.png
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import compare_2d_vs_truth as C
import build_sce_input as B

OFF = "doubleanode_2d_maps_20cm_nosce/voxel_data.npz"          # SCE-OFF
ON = "doubleanode_2d_maps_20cm_mc_full_v10jc/voxel_data.npz"    # SCE-ON
DUAL = "SCEoffsets_SBND_E500_dualmap_voxelTH3.root"
OUT = "offset_study"
COND, RES, MINC = 100.0, 2.0, 30
os.makedirs(OUT, exist_ok=True)
COMPS = ("dx", "dy", "dz")

# --- load measured maps (same quality cuts) ---
meas_off, vx, vy, vz, s_off = C.load_measured_voxels(OFF, COND, RES, MINC)
meas_on, vx2, vy2, vz2, s_on = C.load_measured_voxels(ON, COND, RES, MINC)
assert np.allclose(vx, vx2) and np.allclose(vy, vy2) and np.allclose(vz, vz2)

# --- truth: 0 for SCE-OFF, dual map sampled at voxel centers for SCE-ON ---
truth_on = C.sample_truth_at(vx, vy, vz, DUAL, unflip_x=False)   # raw convention
res_off = meas_off - 0.0
res_on = meas_on - truth_on

tpc0 = (vx < 0)[:, None, None]
both = np.isfinite(res_off).all(-1) & np.isfinite(res_on).all(-1)

print(f"SCE-OFF kept {s_off['n_keep']}, SCE-ON kept {s_on['n_keep']}, "
      f"common voxels {int(both.sum())}\n")

# --- per-component / per-TPC offset table ---
rows = []
print(f"{'comp':4s} {'region':5s} {'off_mean':>9s} {'off_rms':>8s} "
      f"{'on_mean':>9s} {'on_rms':>8s} {'on-off':>8s}  {'slope':>6s} "
      f"{'icpt':>7s} {'r':>6s}")
for i, c in enumerate(COMPS):
    for name, reg in (("all", np.ones_like(tpc0)),
                      ("TPC0", tpc0), ("TPC1", ~tpc0)):
        m = both & reg[..., 0] if reg.shape[-1] == 1 else both & reg[..., 0]
        m = both & np.broadcast_to(reg, both.shape)
        a = res_off[..., i][m]
        b = res_on[..., i][m]
        # regression b ~ slope*a + icpt
        if len(a) > 2 and np.std(a) > 0:
            slope, icpt = np.polyfit(a, b, 1)
            r = np.corrcoef(a, b)[0, 1]
        else:
            slope = icpt = r = np.nan
        rows.append((c, name, a.mean(), a.std(), b.mean(), b.std(),
                     b.mean() - a.mean(), slope, icpt, r))
        print(f"{c:4s} {name:5s} {a.mean():>9.4f} {a.std():>8.4f} "
              f"{b.mean():>9.4f} {b.std():>8.4f} {b.mean()-a.mean():>8.4f}  "
              f"{slope:>6.3f} {icpt:>7.4f} {r:>6.3f}")
    print()

import csv
with open(os.path.join(OUT, "offset_table.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["comp", "region", "off_mean", "off_rms", "on_mean", "on_rms",
                "on_minus_off", "slope_on_vs_off", "intercept", "r"])
    w.writerows(rows)

# --- profiles of each residual component vs x, y, z ---
axes = {"x": (vx, 0), "y": (vy, 1), "z": (vz, 2)}
fig, ax = plt.subplots(3, 3, figsize=(15, 11))
for ri, c in enumerate(COMPS):
    for ci, (an, (av, axis)) in enumerate(axes.items()):
        a = ax[ri, ci]
        for res, lab, col in ((res_off, "SCE-OFF (resid=meas)", "tab:blue"),
                              (res_on, "SCE-ON (meas-truth)", "tab:red")):
            r = res[..., ri].copy()
            r[~both] = np.nan
            prof = np.nanmean(r, axis=tuple(k for k in range(3) if k != axis))
            a.plot(av, prof, "-o", ms=3, color=col, label=lab)
        a.axhline(0, color="k", lw=0.8, ls="--")
        a.set_title(f"{c} residual vs {an}")
        a.set_xlabel(f"{an} [cm]")
        a.set_ylabel(f"{c} resid [cm]")
        a.grid(alpha=0.3)
        if ri == 0 and ci == 0:
            a.legend(fontsize=8)
fig.suptitle("Common residual offset: SCE-ON vs SCE-OFF (identical v10jc pipeline)",
             fontsize=13)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "offset_profiles.png"), dpi=120)
print(f"\nwrote {OUT}/offset_profiles.png + offset_table.csv")
