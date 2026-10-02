"""Compare the MEASURED MC cathode-displacement map to the TRUE cathode
distortion (the SCE forward Delta_x at the cathode, x->0).

The cathode-crossing workflow (`plot_cathode_heatmap.py` ->
`cathode_step1_tick_to_cm.py`) produces, per TPC, an apparent cathode
displacement `x_cathode(y,z)` (tpc{0,1}_displacement.npz, key `x`). In SBND MC
the true cathode distortion is the forward SCE Delta_x evaluated at the cathode
plane. We use the per-TPC **dual** map (E = x<0/TPC0, W = x>0/TPC1) sampled at
x ~ 0.

Outputs per TPC: 3-pane heatmap [measured | truth | measured-truth], plus a
correlation/scale summary and CSV. Always prints a per-TPC sign/scale table
(measured vs +/- truth) so the cathode-method sign convention and the (2/3)
calibration factor can be read off (slope ~1 => calibration holds in MC).

Usage:
    python compare_cathode_vs_truth.py --step1-dir cathode_mc_step1 \
        --truth-root SCEoffsets_SBND_E500_dualmap_voxelTH3.root \
        --out compare_cathode_vs_truth_mc
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import uproot
from scipy.interpolate import RegularGridInterpolator as RGI

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_sce_input as B

HEAT = matplotlib.colormaps["RdYlBu_r"].copy()
HEAT.set_bad("white")


def truth_cathode_slice(truth_root, suffix, yc, zc, cathode_x):
    """Sample TrueFwd_Displacement_X_<suffix> at (cathode_x, yc, zc)."""
    sx, sy, sz = B.sce_bin_centers()
    with uproot.open(truth_root) as f:
        key = f"TrueFwd_Displacement_X_{suffix}"
        if key not in {k.split(";")[0] for k in f.keys()}:
            key = "TrueFwd_Displacement_X"          # single-map fallback
        g = f[key].values().astype(np.float64)
    Y, Z = np.meshgrid(yc, zc, indexing="ij")
    pts = np.stack([np.full(Y.size, cathode_x), Y.ravel(), Z.ravel()], axis=1)
    return RGI((sx, sy, sz), g, method="linear", bounds_error=False,
               fill_value=np.nan)(pts).reshape(len(yc), len(zc))


def panel(ax, grid, yc, zc, vr, title, label):
    ext = (zc[0], zc[-1], yc[0], yc[-1])
    im = ax.imshow(grid, origin="lower", aspect="auto", extent=ext, cmap=HEAT,
                   vmin=-vr, vmax=vr, interpolation="nearest")
    ax.figure.colorbar(im, ax=ax, extend="both", label=label)
    ax.set_title(title); ax.set_xlabel("Z [cm]"); ax.set_ylabel("Y [cm]")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--step1-dir", default="cathode_mc_step1")
    ap.add_argument("--truth-root",
                    default="SCEoffsets_SBND_E500_dualmap_voxelTH3.root")
    ap.add_argument("--out", default="compare_cathode_vs_truth_mc")
    ap.add_argument("--cathode-x", type=float, default=0.0,
                    help="x [cm] at which to sample the truth (cathode plane)")
    ap.add_argument("--vrange", type=float, default=2.0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    rows = []
    for tpc, suffix in ((0, "E"), (1, "W")):
        npz = os.path.join(args.step1_dir, f"tpc{tpc}_displacement.npz")
        d = np.load(npz)
        meas = d["x"].astype(np.float64)            # (ny,nz) cathode disp [cm]
        yc, zc = d["y_centers"], d["z_centers"]
        truth = truth_cathode_slice(args.truth_root, suffix, yc, zc,
                                    args.cathode_x)

        ok = np.isfinite(meas) & np.isfinite(truth)
        m, t = meas[ok], truth[ok]
        t_use = truth
        truth_std = float(np.std(t)) if t.size else np.nan
        offset = float(np.mean(m)) if m.size else np.nan      # systematic offset
        scatter = float(np.std(m)) if m.size else np.nan      # spatial scatter
        rms = float(np.sqrt(np.mean((m - t) ** 2))) if m.size else np.nan
        if t.size > 5 and truth_std > 0.02:
            r = float(np.corrcoef(m, t)[0, 1])
            slope = float(np.sum(m * t) / np.sum(t * t))
        else:
            r = np.nan; slope = np.nan                        # truth pinned ~0
        print(f"\n  TPC{tpc} ({suffix}): n={m.size}  "
              f"truth(@cathode) mean={np.mean(t):+.4f} std={truth_std:.4f} cm "
              f"(~0 => SCE pins the cathode)")
        print(f"    measured: mean offset={offset:+.3f} cm  spatial scatter="
              f"{scatter:.3f} cm  rms(meas-truth)={rms:.3f} cm  "
              f"r={r if np.isnan(r) else round(r,3)}")
        rows.append(dict(tpc=tpc, suffix=suffix, n=int(m.size),
                         truth_mean=float(np.mean(t)), truth_std=truth_std,
                         meas_offset_cm=offset, meas_scatter_cm=scatter,
                         rms_vs_truth_cm=rms, r=r, slope=slope))

        fig, axes = plt.subplots(1, 3, figsize=(20, 5.5))
        panel(axes[0], meas, yc, zc, args.vrange,
              f"measured cathode $\\Delta_x$ (TPC{tpc})", "x [cm]")
        panel(axes[1], t_use, yc, zc, args.vrange,
              f"truth {suffix} $\\Delta_x$ @ x={args.cathode_x:g}", "x [cm]")
        panel(axes[2], meas - t_use, yc, zc, args.vrange,
              f"meas - truth (rms {rms:.2f} cm)", "diff [cm]")
        fig.suptitle(f"MC cathode displacement vs truth, TPC{tpc}  "
                     f"(r={r:.3f}, slope={slope:.2f})")
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, f"cathode_compare_tpc{tpc}.png"),
                    dpi=120)
        plt.close(fig)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out, "cathode_truth_summary.csv"), index=False)
    print("\nsummary:")
    print(df.to_string(index=False), flush=True)
    print(f"\nwrote {args.out}/", flush=True)


if __name__ == "__main__":
    main()
