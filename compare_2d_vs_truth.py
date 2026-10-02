"""Compare the MEASURED double-anode distortion map to the TRUE distortions.

In SBND MC the "true" spatial distortion is the forward displacement stored
in the nominal SCE input ROOT the simulation used
(`TrueFwd_Displacement_{X,Y,Z}`, a 49x49x61 TH3F grid). This script overlays
the measured per-voxel map against that truth:

  * per X-slice, per component: a 3-pane figure
        [ measured | truth | measured - truth ]
    (+ per-component GIFs and a stacked dx/dy/dz GIF)
  * a correlation summary: truth-vs-measured scatter per component with the
    voxel-wise correlation r, mean residual and RMS.
  * residual_summary.csv: per-component (and per-TPC) n_voxels, r,
    mean_resid_cm, rms_resid_cm.

Two modes:
  --mode voxel  (default): sample the truth TH3F at each measured voxel
                center (RegularGridInterpolator, linear). Measured map comes
                from the raw voxel npz, OR from a fitted spline via
                --use-spline <spline_fit.npz>.
  --mode th3 :  compare two SCE-grid TH3F files directly (no interpolation):
                --fwd-root (our forward SCE ROOT) vs --truth-root.

X sign convention
-----------------
`SpaceChargeServiceSBND` stores TrueFwd_Displacement_X with a per-TPC flip
(x<0/TPC0 stored as -Delta_x); `build_sce_input.write_th3f_root` reproduces
that on write. The raw measured voxel map (`delta_mu_all[...,0]`) is in a
per-TPC convention that empirically matches the *stored* reference X, so by
default NO un-flip is applied (`--truth-x-unflip off`). The script ALWAYS
prints a per-TPC median + dx-correlation table under BOTH conventions so the
correct handedness can be confirmed for any given map. Y and Z never flip.

Usage:
    python compare_2d_vs_truth.py --input doubleanode_2d_maps_20cm_mc/voxel_data.npz
    python compare_2d_vs_truth.py --input .../voxel_data.npz --use-spline .../spline_fit.npz
    python compare_2d_vs_truth.py --mode th3 --fwd-root .../sbnd_sce_fwd_*.root
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
from scipy.interpolate import RegularGridInterpolator

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_sce_input as B
from stitch_gifs import stitch_one

COMPONENTS = ("dx", "dy", "dz")
TRUTH_NAMES = ("TrueFwd_Displacement_X", "TrueFwd_Displacement_Y",
               "TrueFwd_Displacement_Z")
HEATMAP = matplotlib.colormaps["RdYlBu_r"].copy()
HEATMAP.set_bad("white")


# ----------------------------------------------------------------------------
# loading
# ----------------------------------------------------------------------------
def load_measured_voxels(npz_path, cond_max, max_res, min_counts):
    """Measured map from the raw voxel npz, NaN where the cuts fail.

    Mirrors fit_doubleanode_2d_spline.load_and_cut (cond + worst-axis
    resolution) and additionally drops voxels below min_counts."""
    d = np.load(npz_path)
    mu = d["delta_mu_all"].astype(np.float64)            # (nx,ny,nz,3)
    cov = d["delta_cov_all"].astype(np.float64)          # (nx,ny,nz,6)
    cond = d["cond_all"]
    cnt = d["count_all"]
    vx = B.voxel_centers(d["x_edges"])
    vy = B.voxel_centers(d["y_edges"])
    vz = B.voxel_centers(d["z_edges"])
    res = np.sqrt(np.maximum(cov[..., :3], 0.0)).max(axis=-1)
    finite = np.isfinite(mu).all(axis=-1)
    keep = (finite & np.isfinite(cond) & (cond <= cond_max)
            & np.isfinite(res) & (res <= max_res) & (cnt >= min_counts))
    meas = np.where(keep[..., None], mu, np.nan)
    stats = dict(n_inverted=int(finite.sum()), n_keep=int(keep.sum()))
    return meas, vx, vy, vz, stats


def load_measured_spline(spline_npz):
    """Measured map from a fitted spline (spline_mu, masked by keep_mask)."""
    s = np.load(spline_npz)
    spl = s["spline_mu"].astype(np.float64)              # (nx,ny,nz,3)
    keep = s["keep_mask"]
    xe = s["x_edges"]
    vx = B.voxel_centers(xe)
    meas = np.where(keep[..., None], spl, np.nan)
    # y/z centers are not stored in spline_fit.npz; recover from shape on the
    # standard SCE-grid-independent voxel grid -> caller supplies them.
    return meas, vx, keep, dict(n_keep=int(keep.sum()))


def read_th3(path, names):
    """Read named TH3F as float64 arrays (indexed [x,y,z]) + bin centers."""
    with uproot.open(path) as f:
        keys = {k.split(";")[0] for k in f.keys()}
        arrs = []
        for n in names:
            if n not in keys:
                raise KeyError(f"{n} missing from {path} (have {sorted(keys)})")
            arrs.append(f[n].values().astype(np.float64))
    return np.stack(arrs, axis=-1)                       # (nx,ny,nz,3)


# ----------------------------------------------------------------------------
# truth sampling / convention
# ----------------------------------------------------------------------------
def xflip(arr_x, xc):
    """Multiply the X-component grid by -1 where the x-center is < 0 (TPC0)."""
    return np.where((np.asarray(xc) < 0)[:, None, None], -arr_x, arr_x)


def is_dual_map(truth_root):
    """True if the ROOT carries per-TPC `_E`/`_W` maps (dual map) instead of a
    single `TrueFwd_Displacement_*`."""
    with uproot.open(truth_root) as f:
        keys = {k.split(";")[0] for k in f.keys()}
    return ("TrueFwd_Displacement_X" not in keys
            and "TrueFwd_Displacement_X_E" in keys)


def _interp(grid, sx, sy, sz, pts, shape):
    return RegularGridInterpolator((sx, sy, sz), grid, method="linear",
                                   bounds_error=False, fill_value=np.nan
                                   )(pts).reshape(shape)


def sample_truth_at(vx, vy, vz, truth_root, unflip_x):
    """Sample truth TrueFwd_* at voxel centers -> (nvx,nvy,nvz,3).

    Handles both the single symmetric map (`TrueFwd_Displacement_*`) and the
    per-TPC dual map (`*_E` for x<0/East/TPC0, `*_W` for x>0/West/TPC1): each
    half-map is interpolated over the full grid and selected per voxel by the
    sign of x (no cathode blending)."""
    sx, sy, sz = B.sce_bin_centers()
    X, Y, Z = np.meshgrid(vx, vy, vz, indexing="ij")
    pts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    shape = (len(vx), len(vy), len(vz))
    east = (np.asarray(vx) < 0)[:, None, None]   # TPC0
    dual = is_dual_map(truth_root)
    truth = np.full((*shape, 3), np.nan)
    with uproot.open(truth_root) as f:
        for c, comp in enumerate("XYZ"):
            if dual:
                gE = f[f"TrueFwd_Displacement_{comp}_E"].values().astype(np.float64)
                gW = f[f"TrueFwd_Displacement_{comp}_W"].values().astype(np.float64)
                tE = _interp(gE, sx, sy, sz, pts, shape)
                tW = _interp(gW, sx, sy, sz, pts, shape)
                val = np.where(east, tE, tW)     # E for x<0, W for x>0
            else:
                grid = f[f"TrueFwd_Displacement_{comp}"].values().astype(np.float64)
                if c == 0 and unflip_x:
                    grid = xflip(grid, sx)
                val = _interp(grid, sx, sy, sz, pts, shape)
            if c == 0 and unflip_x and dual:
                val = xflip(val, vx)
            truth[..., c] = val
    return truth


def convention_table(meas, truth_stored_x, truth_unflip_x, vx):
    """Print per-TPC median + dx-correlation for both X conventions."""
    mx = meas[..., 0]
    tpc0 = (np.asarray(vx) < 0)
    print("\n  X sign-convention check (per-TPC dx medians, cm):")
    print(f"    {'region':6s} {'measured':>10s} {'truth(raw)':>11s} "
          f"{'truth(unflip)':>14s}")
    for name, m in (("TPC0", tpc0), ("TPC1", ~tpc0)):
        mm = np.nanmedian(mx[m])
        tr = np.nanmedian(truth_stored_x[m])
        tu = np.nanmedian(truth_unflip_x[m])
        print(f"    {name:6s} {mm:>10.3f} {tr:>11.3f} {tu:>14.3f}")
    for label, tx in (("raw   ", truth_stored_x), ("unflip", truth_unflip_x)):
        a, b = mx.ravel(), tx.ravel()
        ok = np.isfinite(a) & np.isfinite(b)
        r = np.corrcoef(a[ok], b[ok])[0, 1] if ok.sum() > 2 else np.nan
        rms = np.sqrt(np.nanmean((a[ok] - b[ok]) ** 2)) if ok.sum() else np.nan
        print(f"    dx vs truth({label}): r={r:+.4f}  rms={rms:.3f} cm")
    print("  (default uses --truth-x-unflip; pick the higher-r convention)\n",
          flush=True)


# ----------------------------------------------------------------------------
# plots
# ----------------------------------------------------------------------------
def comp_split(arr, cnt_mask=None):
    return {c: arr[..., i] for i, c in enumerate(COMPONENTS)}


def make_slice_plots(meas, truth, vx, vy, vz, out, vrange, no_gif):
    """Per-X-slice 3-pane [measured | truth | meas-truth], + GIFs."""
    ye = np.append(vy - (vy[1] - vy[0]) / 2, vy[-1] + (vy[1] - vy[0]) / 2)
    ze = np.append(vz - (vz[1] - vz[0]) / 2, vz[-1] + (vz[1] - vz[0]) / 2)
    extent = (ze[0], ze[-1], ye[0], ye[-1])
    nx = len(vx)
    M = comp_split(meas)
    T = comp_split(truth)
    by_comp = {}
    for ci, comp in enumerate(COMPONENTS):
        cdir = os.path.join(out, comp, "png")
        pdir = os.path.join(out, comp, "pdf")
        os.makedirs(cdir, exist_ok=True)
        os.makedirs(pdir, exist_ok=True)
        files = []
        for ix in range(nx):
            gm, gt = M[comp][ix], T[comp][ix]
            diff = gm - gt
            if not (np.isfinite(gm).any() or np.isfinite(gt).any()):
                continue
            fig, axes = plt.subplots(1, 3, figsize=(20, 6))
            for ax, (label, grid) in zip(axes[:2],
                                         [("measured", gm), ("truth", gt)]):
                im = ax.imshow(grid, origin="lower", aspect="equal",
                               extent=extent, cmap=HEATMAP,
                               vmin=-vrange, vmax=vrange,
                               interpolation="nearest")
                fig.colorbar(im, ax=ax, extend="both",
                             label=f"$\\Delta_{{{comp[1]}}}$ [cm]")
                ax.set_title(label)
                ax.set_xlabel("Z [cm]"); ax.set_ylabel("Y [cm]")
            im = axes[2].imshow(diff, origin="lower", aspect="equal",
                                extent=extent, cmap=HEATMAP,
                                vmin=-vrange, vmax=vrange,
                                interpolation="nearest")
            fig.colorbar(im, ax=axes[2], extend="both",
                         label="measured - truth [cm]")
            dv = diff[np.isfinite(diff)]
            axes[2].set_title(f"measured - truth  (N={dv.size}, "
                              f"mean={dv.mean() if dv.size else 0:+.2f}, "
                              f"rms={dv.std() if dv.size else 0:.2f} cm)")
            axes[2].set_xlabel("Z [cm]"); axes[2].set_ylabel("Y [cm]")
            hw = (vx[1] - vx[0]) / 2
            fig.suptitle(f"double-anode $\\Delta_{{{comp[1]}}}$ vs truth, "
                         f"X in [{vx[ix]-hw:+.0f}, {vx[ix]+hw:+.0f}] cm")
            fig.tight_layout()
            tag = f"x{int(round(vx[ix])):+04d}"
            p = os.path.join(cdir, f"{tag}.png")
            fig.savefig(p, dpi=110)
            fig.savefig(os.path.join(pdir, f"{tag}.pdf"))
            plt.close(fig)
            files.append(p)
        by_comp[comp] = files

    if not no_gif:
        gif_paths = {}
        for comp, files in by_comp.items():
            if not files:
                continue
            gif = os.path.join(out, comp, f"{comp}.gif")
            B.make_gif(files, gif, duration_ms=200)
            gif_paths[comp] = gif
            print(f"  GIF: {gif}", flush=True)
        if len(gif_paths) == 3:
            stitch_one(gif_paths["dx"], gif_paths["dy"], gif_paths["dz"],
                       os.path.join(out, "dxyz_stack.gif"),
                       "double-anode 2D vs truth (dx/dy/dz)")


def correlation_summary(meas, truth, vx, out, vrange):
    M = comp_split(meas)
    T = comp_split(truth)
    tpc0 = (np.asarray(vx) < 0)[:, None, None]
    rows = []
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    for ax, comp in zip(axes, COMPONENTS):
        a = M[comp].ravel()
        b = T[comp].ravel()
        ok = np.isfinite(a) & np.isfinite(b)
        aa, bb = a[ok], b[ok]
        ax.scatter(bb, aa, s=4, alpha=0.3, color="navy")
        lim = vrange * 1.6
        ax.plot([-lim, lim], [-lim, lim], "r-", lw=1)
        if aa.size > 2:
            r = np.corrcoef(aa, bb)[0, 1]
            resid = aa - bb
            ax.set_title(f"$\\Delta_{{{comp[1]}}}$: N={aa.size}, r={r:.4f}, "
                         f"<meas-truth>={resid.mean():+.3f}, "
                         f"rms={resid.std():.3f} cm")
        ax.set_xlabel(f"truth $\\Delta_{{{comp[1]}}}$ [cm]")
        ax.set_ylabel(f"measured $\\Delta_{{{comp[1]}}}$ [cm]")
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.grid(alpha=0.3)
        # CSV rows: all + per TPC
        for reg, mask in (("all", np.ones_like(tpc0, bool)),
                          ("TPC0", tpc0), ("TPC1", ~tpc0)):
            ma = np.broadcast_to(mask, M[comp].shape)
            a2 = M[comp][ma].ravel(); b2 = T[comp][ma].ravel()
            o2 = np.isfinite(a2) & np.isfinite(b2)
            a2, b2 = a2[o2], b2[o2]
            if a2.size > 2:
                rr = float(np.corrcoef(a2, b2)[0, 1])
                rows.append(dict(component=comp, region=reg, n_voxels=int(a2.size),
                                 r=rr, mean_resid_cm=float((a2 - b2).mean()),
                                 rms_resid_cm=float((a2 - b2).std())))
    fig.suptitle("double-anode displacement: measured vs truth (per-voxel)")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "correlation_summary.png"), dpi=120)
    fig.savefig(os.path.join(out, "correlation_summary.pdf"))
    plt.close(fig)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, "residual_summary.csv"), index=False)
    print("\nresidual summary:")
    print(df[df.region == "all"].to_string(index=False), flush=True)


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default="doubleanode_2d_maps_20cm_mc/voxel_data.npz",
                    help="measured voxel_data.npz")
    ap.add_argument("--truth-root", default=B.REFERENCE_SCE_ROOT_DEFAULT,
                    help="nominal SCE ROOT carrying the true TrueFwd_*")
    ap.add_argument("--out", default="compare_2d_vs_truth_mc")
    ap.add_argument("--mode", choices=("voxel", "th3"), default="voxel")
    ap.add_argument("--use-spline", default=None,
                    help="(voxel mode) compare a fitted spline_fit.npz instead "
                         "of the raw voxel map")
    ap.add_argument("--fwd-root", default=None,
                    help="(th3 mode) our forward SCE ROOT to compare to truth")
    ap.add_argument("--cond-max", type=float, default=100.0)
    ap.add_argument("--max-resolution-cm", type=float, default=2.0)
    ap.add_argument("--min-counts", type=int, default=30)
    ap.add_argument("--truth-x-unflip", choices=("off", "on"), default="off",
                    help="un-flip the per-TPC X sign of the truth (default off; "
                         "see the printed convention table)")
    ap.add_argument("--vrange", type=float, default=4.0)
    ap.add_argument("--no-gif", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    unflip = (args.truth_x_unflip == "on")

    if args.mode == "th3":
        if not args.fwd_root:
            ap.error("--mode th3 requires --fwd-root")
        sx, sy, sz = B.sce_bin_centers()
        vx, vy, vz = sx, sy, sz
        meas = read_th3(args.fwd_root, TRUTH_NAMES).copy()  # stored convention
        truth_stored = read_th3(args.truth_root, TRUTH_NAMES)
        # Our forward ROOT had the write-time per-TPC X flip applied
        # (build_sce_input: fwd[x<0,...,0]*=-1) on top of the measured map.
        # Un-flip it to recover the measured-convention map, which matches the
        # truth's RAW stored X (see the voxel-mode convention table). Truth is
        # then handled exactly like voxel mode (raw by default, --truth-x-unflip
        # to flip it).
        meas[..., 0] = xflip(meas[..., 0], sx)
        truth = truth_stored.copy()
        truth_flipped_x = xflip(truth_stored[..., 0], sx)
        if unflip:
            truth[..., 0] = truth_flipped_x
        truth_unflip_x = truth_stored[..., 0] if unflip else truth_flipped_x
        print(f"[th3] measured={args.fwd_root}\n      truth={args.truth_root}")
    else:
        if args.use_spline:
            meas_s, vx, keep, st = load_measured_spline(args.use_spline)
            d = np.load(args.input)
            vy = B.voxel_centers(d["y_edges"]); vz = B.voxel_centers(d["z_edges"])
            meas = meas_s
            print(f"[voxel/spline] {args.use_spline}: {st['n_keep']} voxels kept")
        else:
            meas, vx, vy, vz, st = load_measured_voxels(
                args.input, args.cond_max, args.max_resolution_cm,
                args.min_counts)
            print(f"[voxel] {args.input}: {st['n_keep']}/{st['n_inverted']} "
                  f"voxels kept after cuts")
        truth = sample_truth_at(vx, vy, vz, args.truth_root, unflip)
        # for the convention table, also build the other-handedness truth-X
        truth_other = sample_truth_at(vx, vy, vz, args.truth_root, not unflip)
        truth_unflip_x = truth_other[..., 0]

    # convention diagnostic (always)
    convention_table(meas, truth[..., 0], truth_unflip_x, vx)

    make_slice_plots(meas, truth, vx, vy, vz, args.out, args.vrange, args.no_gif)
    correlation_summary(meas, truth, vx, args.out, args.vrange)
    print(f"\nwrote {args.out}/", flush=True)


if __name__ == "__main__":
    main()
